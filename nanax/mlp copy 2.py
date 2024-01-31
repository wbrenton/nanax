import os
import time
import tyro
import rich
from functools import partial
from types import SimpleNamespace
from dataclasses import dataclass, field, asdict
from typing import List, Sequence, Callable, Optional

import jax
import flax
import optax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from clu import metrics, metric_writers, periodic_actions
from tensorboardX import SummaryWriter

from utils.jax_data import prefetch
from utils.clu_data import load_dataset
from utils.helpers import parse_activation_fn

# batch size the is the number of example per gradient update (1024)
# local batch size is the number of example per process (1024 / 2 = 512)
# micro_batch_size is the number of examples per gradient step (512 / 8 = 64)
# effective_batch_size is the number of examples that actually get passed forward
# with 8 devices per process, 
# this means you would have gradient accumulation step of 8 if you have 8 devices


@dataclass
class Args:
    # common args
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    "the name of this experiment"
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    "if toggled, this experiment will be tracked with Weights and Biases"
    wandb_project_name: str = "nanax"
    "the wandb's project name"
    wandb_entity: Optional[str] = None
    "the entity (team) of wandb's project"
    run_name: tyro.conf.Suppress[str] = None
    "TO BE FILLED: a unique name of this run"

    # training args
    dataset_name: str = "mnist"
    "the name of the huggingface dataset"
    cache_dataset: bool = True
    "whether to store the dataset in memory"
    num_prefetch: int = 2
    "the number of batches to place in an on device buffer"
    n_classes: int = 10
    "the number of classes of the dataset"
    eval_every: int = 1000
    "the frequency of updates to evaluate the model"
    learning_rate: float = 0.01
    "the learning rate of the optimizer"
    batch_size: int = 128
    "number of examples per gradient update"
    local_batch_size: tyro.conf.Suppress[int] = None
    "the batch size of each process"
    num_epochs: int = 10
    "number of epochs to train"
    hiddens: Sequence[int] = (300, 100)
    "the hidden sizes of the MLP"
    activation_fn_str = 'relu'
    "the activation function of the MLP"

    # distributed settings
    world_size: tyro.conf.Suppress[int] = None
    "the number of processes"
    local_rank: int = 0
    "the rank of this process"
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    "the device ids that script will use"
    learner_devices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    "the devices that script will use"
    global_learner_devices: tyro.conf.Suppress[int] = None  # real type is `List[str]`
    "the total devices (across all nodes and machines) that script will use"

@flax.struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")
    accuracy: metrics.Average.from_output("accuracy")

    @classmethod
    def to_str(cls, metrics: metrics.Collection, n: int = 4):
        """Converts metrics to a string with n decimal places."""
        metrics = metrics.write_ready()
        format_specifier = f"{{k}}: {{v:.{n}f}}"
        return " ".join([format_specifier.format(k=k, v=v) for k, v in metrics.items()])

    def write_ready(self):
        return self.unreplicate().compute()


class MLP(nn.Module):
    """A simple MLP model."""
    n_classes: int
    hiddens: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, x: jax.Array):
        x = x.reshape((x.shape[0], -1))
        for hidden in self.hiddens:
            x = nn.Dense(hidden)(x)
            x = self.activation_fn(x)
        logits = nn.Dense(self.n_classes)(x)
        return logits

def loss_fn(params, batch):
    logits = state.apply_fn(params, batch['image'])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=batch['label']))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(batch['label'], -1))
    return loss, dict(loss=loss, accuracy=accuracy)

def train_step(state, batch):
    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, outputs = grad_fn(state.params, batch)
    grads = jax.lax.pmean(grads, axis_name="batch")
    state = state.apply_gradients(grads=grads)
    return state, Metrics.gather_from_model_output(**outputs, axis_name="batch")

def evaluation_step(state, batch):
    _, outputs = loss_fn(state.params, batch)
    return state, Metrics.gather_from_model_output(**outputs, axis_name="batch")

if __name__ == "__main__":
    args = tyro.cli(Args)

    # distributed training
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    args.world_size = jax.process_count()
    if args.world_size > 1:
        jax.distributed.initialize(local_device_ids=args.learner_device_ids) # TODO: assert this is correct
    learner_devices = [local_devices[d_id] for d_id in args.learner_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(args.world_size)
        for d_id in args.learner_device_ids
    ]
    rich.pretty.pprint({"global_learner_devices": global_learner_devices})
    args.global_learner_devices = [str(item) for item in global_learner_devices]
    args.learner_devices = [str(item) for item in learner_devices]
    args.local_rank = jax.process_index()
    args.local_batch_size = args.batch_size // args.world_size

    # logging
    console = rich.console.Console(force_terminal=True)
    args.run_name = f"{args.exp_name}_{args.dataset_name}_{args.seed}_{time.time()}"
    writer = SimpleNamespace()  # dummy writer
    if args.local_rank == 0:
        rich.pretty.pprint(args)
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=args.run_name,
                monitor_gym=True,
                save_code=True,
            )
        workdir = os.path.join("runs", args.run_name)
        writer = metric_writers.create_default_writer(workdir, asynchronous=False)
        writer.write_hparams(asdict(args))

    # seeding
    local_seed = args.seed + args.local_rank * 100003
    rng = jax.random.PRNGKey(local_seed)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    # load data
    train_dataset, test_dataset = load_dataset(args, data_rng)
    batch = next(iter(train_dataset))
    images = jnp.ones(batch['image'].shape[1:], jnp.float32)

    # initialize train state
    activation_fn = parse_activation_fn(args.activation_fn_str)
    network = MLP(args.n_classes, args.hiddens, activation_fn)
    state = TrainState.create(
        apply_fn=network.apply,
        params=network.init(init_rng, images),
        tx=optax.adam(args.learning_rate),
    )
    state = flax.jax_utils.replicate(state)

    # TODO: move this be resued as a loop util
    def tracked_step(state, metrics, batch, step_fn, hooks):
        if not metrics:
            metrics = flax.jax_utils.replicate(Metrics.empty())

        with jax.profiler.StepTraceAnnotation(
            step_fn.__name__, step_num=state.step.item()
        ):
            state, step_metrics = step_fn(state, batch)
            writer.write_scalars(step=step, scalars=step_metrics.write_ready())
            metrics = metrics.merge(step_metrics)

        for hook in hooks:
            hook(state.step.item())

        return state, metrics

    def tracked_step_loop(state, dataset, step_fn):
        metrics = flax.jax_utils.replicate(Metrics.empty())
        for _ in range(10):
            state, step_metrics = step_fn(state, next(dataset))
            metrics = metrics.merge(step_metrics)
            writer.write_scalars(step=step, scalars=step_metrics.write_ready())
        return state, metrics

    # distributed functions
    p_train_step, p_evaluation_step = map(
        lambda fn: jax.pmap(
            fn, 
            axis_name="batch",
            devices=learner_devices,
            donate_argnums=(0,)
        ), [train_step, evaluation_step]
    )

    # setup train and evaluation loops
    train_step_fn = partial(
        tracked_step,
        step_fn=p_train_step,
        hooks=[
            periodic_actions.Profile(logdir=workdir),
            periodic_actions.ReportProgress(
                num_train_steps=train_dataset.cardinality().numpy(),
                writer=writer,
                every_steps=100,
            ),
        ]
    )
    evaluation_fn = partial(
        tracked_step_loop,
        step_fn=p_evaluation_step,
        dataset=prefetch(test_dataset, 2)
    )

    steps = train_dataset.cardinality().numpy()
    print(f"Training for {steps} steps")
    train_metrics, eval_metrics = None, None
    for step, batch in enumerate(prefetch(train_dataset, 2)):
        state, train_metrics = train_step_fn(state, train_metrics, batch)

        if (step % args.eval_every == 0) or (step == steps - 1):
            state, eval_metrics = evaluation_fn(state)
            console.print(f"Update step: {step}: Train {Metrics.to_str(train_metrics)} | Test {Metrics.to_str(eval_metrics)}")