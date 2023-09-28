import functools
import os
import time
from tqdm import tqdm
import tiktoken
from functools import partial
from types import SimpleNamespace
from typing import List, Optional
from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import tyro
from datasets import load_dataset
from flax import jax_utils, traverse_util
from flax import struct
from flax.core import freeze
from flax.training import common_utils, orbax_utils
from flax.training.train_state import TrainState
from rich.pretty import pprint
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter

from model import GPT

@dataclass
class GPTHParams:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    use_bias: bool = False
    dtype: Optional[str] = jnp.float32

@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "nanax"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    cuda: bool = True
    """Whether to use cuda if available."""
    distributed: bool = False
    "whether to use `jax.distirbuted`"
    run_name: tyro.conf.Suppress[str] = None
    """TO BE FILLED: a unique name of this run"""

    base_model: str = "gpt2"
    """the name of the pretrained model to use"""
    dataset_name: str = "openwebtext" # "tiny_shakespeare" 
    """the name of the dataset to use for labels in `https://huggingface.co/datasets/vwxyzjn/lm-human-preferences`"""
    lr: float = 1e-3
    """the maximum learning rate"""
    beta1: float = 0.9
    """the first momentum decay hp for AdamW"""
    beta2: float = 0.99
    """the second momentum decay hp for AdamW"""
    weight_decay: float = 1e-2
    """the weight decay for AdamW"""
    eps: float = 1e-5
    """the epsilon for AdamW"""
    anneal_lr: bool = True
    """whether or not to use warmup cosine decay lr schedule"""
    warmup_steps: int = 100
    """number of update steps to warmup for"""
    decay_steps: int = 5000
    """number of update steps to decay for"""
    end_lr: float = 1e-4
    "the minimum learning rate"
    gradient_accumulation_steps: int = 1
    """The number of steps to accumulate before performing an update"""
    world_size: tyro.conf.Suppress[int] = None
    """the number of processes to use"""
    local_batch_size: tyro.conf.Suppress[int] = 64
    """the per rank batch size"""
    batch_size: tyro.conf.Suppress[int] = None
    """the batch size across all ranks"""
    effective_batch_size: tyro.conf.Suppress[int] = None
    """the effective batch size by way of gradient accumulation"""
    total_update_steps: int = 5000
    """total number of optimization steps"""
    eval_frequency: int = 250
    """How often to evaluate during training"""
    eval_iterations: int = 1 #200
    """Number of validation batches per evaluation"""
    print_output_freq: int = 10
    """How often to print"""
    save_path: str = "models/"
    """Where to save the model"""

    # distributed settings
    local_rank: int = 0
    """the rank of this process"""
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that script will use"
    learner_devices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the devices that script will use"""
    global_learner_decices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    """the total devices (across all nodes and machines) that script will use"""
    gpt2_hparams: GPTHParams = field(default_factory=GPTHParams)

@struct.dataclass
class Batch:
    input_tokens: jnp.array
    target_tokens: jnp.array

def load_hf_dataset(args):
    """Loads a HuggingFace dataset and saves it to disk using np.memmap"""

    def process_shakespear(data, split: str):
        chars = sorted(list(set(data[split]['text'][0])))
        vocab_size = len(chars)
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
        train_data = np.array(encode(data[split]['text'][0]), dtype=np.uint16)
        val_data = np.array(encode(data[split]['text'][0]), dtype=np.uint16)
        return train_data, val_data, vocab_size, encode, decode

    def process_webtext(dataset, dataset_dir):
        split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
        split_dataset['validation'] = split_dataset.pop('test')

        # https://github.com/huggingface/datasets/issues/5536#issuecomment-1712221963
        class TikTokenFactory:
            def __init__(self):
                self._enc = None
                self.eot_token = None

            def encode_ordinary(self, text):
                if self._enc is None:
                    self._enc = tiktoken.get_encoding("gpt2")
                    self.eot_token = self._enc.eot_token
                return self._enc.encode_ordinary(text)
        
        # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
        def process(enc, example):
            enc = tiktoken.get_encoding("gpt2")
            ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
            ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
            # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        tokenized = split_dataset.map(
            partial(process, TikTokenFactory()),
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
        )

        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = f'{split}.bin'
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(os.path.join(dataset_dir, filename), dtype=dtype, mode='w+', shape=(arr_len,))

            print(f"writing {filename}...")
            idx = 0
            for example in tqdm(dset):
                arr[idx : idx + example['len']] = example['ids']
                idx += example['len']
            arr.flush()

        # train.bin is ~17GB, val.bin ~8.5MB
        # train has ~9B tokens (9,035,582,198)
        # val has ~4M tokens (4,434,897)

        # to read the bin files later, e.g. with numpy:
        # m = np.memmap('train.bin', dtype=np.uint16, mode='r')

    # determine if .bin files already exist
    cwd = os.getcwd()
    model_dir = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(cwd, 'data', model_dir, args.dataset_name)

    # download the data if you haven't already
    if not os.path.exists(dataset_dir):
        os.makedirs(os.path.join(cwd, 'data', model_dir, args.dataset_name))
        dataset = load_dataset(args.dataset_name)
        if args.dataset_name == 'tiny_shakespeare':
            process_shakespear(dataset, 'train')
        elif args.dataset_name == 'openwebtext':
            process_webtext(dataset)

    train_data = np.memmap(os.path.join(dataset_dir, 'train.bin'))
    val_data = np.memmap(os.path.join(dataset_dir, 'validation.bin'))
    vocab_size = tiktoken.get_encoding("gpt2").max_token_value
    
    return train_data, val_data, vocab_size


def get_dataloader_iter(rng, dataset, args, train=True):
    """Get iteration of dataloader."""
    block_size = args.gpt2_hparams.block_size
    shape = (
        args.total_update_steps if train else args.eval_iterations,
        args.batch_size * args.gradient_accumulation_steps if train else args.batch_size
    )
    # idxs = jax.random.randint(rng, shape=shape, minval=0, maxval=len(dataset)-block_size, dtype=jnp.int64)
    idxs = np.random.randint(low=0, high=len(dataset)-block_size, size=shape)
    batch_shape = (args.gradient_accumulation_steps, args.batch_size, block_size) if train else (args.batch_size, block_size)
    for idx in idxs:
        x = jnp.stack([dataset[i : i+block_size] for i in idx]).reshape(batch_shape)
        y = jnp.stack([dataset[i+1 : i+1+block_size] for i in idx]).reshape(batch_shape)
        yield Batch(input_tokens=x, target_tokens=y)

def train_step(state, batch, rng):
    # should pmean be taken inside the minibatch step or after all minibatches
    
    def loss_fn(params, mb, rng):
        logits = state.apply_fn(params, mb.input_tokens, rngs={"dropout": rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, mb.target_tokens).mean()
        accuracy = (logits.argmax(-1) == mb.target_tokens).mean()
        return loss, accuracy

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    def minibatch_step(cumulative_state, minibatch):
        loss, accuracy, grads, rng = cumulative_state
        step_key, rng = jax.random.split(rng)
        (mini_loss, mini_accuracy), mini_grads = value_and_grad(state.params, minibatch, step_key)
        loss, accuracy, grads = jax.tree_map(jnp.add, (loss, accuracy, grads), (mini_loss, mini_accuracy, mini_grads))
        return (loss, accuracy, grads, rng), None

    # accumulate gradients
    init_state = (0., 0., jax.tree_map(jnp.zeros_like, state.params), rng)
    (loss, accuracy, grads, rng), _ = jax.lax.scan(
        f=minibatch_step,
        init=init_state,
        xs=batch)
    # sum -> mean
    loss, accuracy, grads = jax.tree_map(lambda x: x/args.gradient_accumulation_steps, (loss, accuracy, grads))
    # mean across learners
    loss, accuracy, grads = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), (loss, accuracy, grads))
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss, "accuracy": accuracy}

def val_step(state, batch):
    
    def loss_fn(params):
        logits = state.apply_fn(params, batch.input_tokens, train=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch.target_tokens).mean()
        accuracy = (logits.argmax(-1) == batch.target_tokens).mean()
        return loss, accuracy

    loss, accuracy = loss_fn(state.params)
    loss = jax.lax.pmean(loss, axis_name="batch")
    accuracy = jax.lax.pmean(accuracy, axis_name="batch")
    return {"loss": loss, "accuracy": accuracy}

def train(args: Args):
    if args.distributed:
        jax.distributed.initialize(
            local_device_ids=range(len(args.learner_device_ids)),
        )

    args.world_size = jax.process_count()
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_decices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    pprint({"global_learner_decices": global_learner_decices})
    args.global_learner_decices = [str(item) for item in global_learner_decices]
    args.learner_devices = [str(item) for item in learner_devices]
    args.batch_size = int(args.local_batch_size * len(local_devices) * args.world_size)
    args.effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    args.local_rank = jax.process_index()

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    if args.local_rank == 0:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            wandb.run.log_code(".")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, train_iter_rng = jax.random.split(rng, 3)

    train_data, val_data, vocab_size = load_hf_dataset(args)
    train_iter = get_dataloader_iter(train_iter_rng, train_data, args)
    args.gpt2_hparams.vocab_size = vocab_size

    gpt = GPT(args.gpt2_hparams)
    x = jnp.zeros((args.batch_size, args.gpt2_hparams.block_size), dtype=jnp.int32)
    params = gpt.init(init_rng, x, train=False)
    print(gpt.tabulate(init_rng, x, train=False))

    lr_schedule = optax.warmup_cosine_decay_schedule(
        0.0, args.lr, args.warmup_steps, args.decay_steps, args.end_lr
    )
    optim_fn = lambda weight_decay: optax.adamw(
        lr_schedule if args.anneal_lr else args.lr,
        b1=args.beta1,
        b2=args.beta2,
        weight_decay=weight_decay,
        eps=args.eps
    )
    optimizer = optax.multi_transform(
        transforms={'decay': optim_fn(args.weight_decay), 'no_decay': optim_fn(0.0)},
        param_labels=traverse_util.path_aware_map(
                lambda path, x: 'decay' if path[-1] in ('kernel',) else 'no_decay',
                params),
    )
    state = TrainState.create(
        apply_fn=gpt.apply,
        params=params,
        tx=optimizer
    )
    state = jax_utils.replicate(state)
    p_train_step = jax.pmap(train_step, axis_name='batch')
    p_val_step = jax.pmap(val_step, axis_name='batch')
    
    print(f"Starting training on {len(train_data)} tokens, vocabulary size: {vocab_size}, effective batch size: {args.effective_batch_size}")
    for global_step, train_batch in enumerate(train_iter):
        rng, step_rng = jax.random.split(rng)
        train_batch, step_rng = common_utils.shard([train_batch, step_rng])
        state, train_metrics = p_train_step(state, train_batch, step_rng)
        
        writer.add_scalar("train/lr", lr_schedule(global_step).item())
        train_metrics = common_utils.get_metrics([train_metrics])
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, global_step)
        
        if global_step % args.eval_frequency == 0:
            val_iter = get_dataloader_iter(step_rng[0], val_data, args, train=False)
            val_metrics_list = []
            for val_batch in val_iter:
                val_batch = common_utils.shard(val_batch)
                val_metrics = p_val_step(state, val_batch)
                val_metrics_list.append(val_metrics)
            
            val_metrics = common_utils.get_metrics(val_metrics_list)
            for key, value in val_metrics.items():
                value = value.mean()
                val_metrics[key] = value
                writer.add_scalar(f"validation/{key}", value, global_step)
            print(f"global_step: {global_step} test/accuracy: {val_metrics['accuracy']:.3f}")
        if global_step % args.print_output_freq == 0:
            print(f"global_step: {global_step} train/accuracy: {train_metrics['accuracy'].item():.3f}")

    state = jax_utils.unreplicate(state)

    if args.save_path and args.local_rank == 0:
        ckpt = {"gpt_model": state, "args": vars(args)}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(ckpt)
        orbax_checkpointer.save(args.save_path+'-'+run_name, ckpt, save_args=save_args, force=True)

    if args.local_rank == 0 and args.track:
        wandb.finish()

# TODO: create model presets for shakespear and webtext
# TODO: setup webtext preprocessing (folloing garcia)
# TODO: setup world class dataloader
# TODO: read through tri dao implmentation, learn about optimizations, make them for yourself
# TODO: go and polish model.py (understand to the point of conversational)
# TODO: setup saving best only during training after a certain point
# TODO: create presets for different gpt sizes
# TODO: setup option to do periodic sampling along with evaluation

if __name__ == "__main__":
    args = tyro.cli(Args)
    train(args)