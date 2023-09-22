import os
import time
from functools import partial

import jax
import optax
import numpy as np
import jax.numpy as jnp
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from flax.core.frozen_dict import freeze, unfreeze
from datasets import load_dataset
import pickle

from model import CLIP, CLIPConfig

n_layer = 6
n_head = 4
n_embd = 128
n_embd_proj = 64
patch_size = 7
num_classes = 10
block_size = 6
dropout = 0.1
use_bias = True
batch_size = 128
img_shape = (28, 28)
learning_rate = 3e-4
max_iters = 500000
eval_interavl = 1 # 1000
eval_iters = 1 # 100

dataset = 'mnist'
out_dir = 'out'
wandb_log = False
wandb_project = "nanax"
exp_name = "CLIP" + f"-{dataset}-" + str(time.time())
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging

int_label_to_str_label = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
                  5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
# pad with '*'
int_label_to_str_label = {k: "".join(v + '*'*(block_size - len(v))) for k, v in int_label_to_str_label.items()}
text = [value for value in int_label_to_str_label.values()]
text = " ".join(text)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
int_label_to_int_enc = {k: [stoi[c] for c in v] for k, v in int_label_to_str_label.items()}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
data = np.array(encode(text), dtype=np.uint16)

def int_encode_class_label(example):
    """ 
    Converts integer class label (0) into str class label ("zero") 
    then integer encodes str to be used with learnable token embedding
    """
    example['label'] = int_label_to_int_enc[int(example['label'])]
    return example

def load_data(name: str):
    ds = load_dataset(name)
    n_classes = len(ds['train'].features['label'].names)
    ds = ds.with_format("jax")
    ds = ds.shuffle()

    def create_iter(dataset):
        while True:
            for i, batch in enumerate(dataset.iter(batch_size=batch_size)):
                batch['count'] = i
                images, int_labels = batch['image'], batch['label']
                texts = np.array([int_label_to_int_enc[int(label)] for label in int_labels])
                images = jnp.expand_dims(images, axis=-1) if len(images) != 4 else images
                yield images, texts

    train_iter = create_iter(ds['train'])
    test_iter = create_iter(ds['test'])

    return train_iter, test_iter, n_classes

def save_best(current, best, params, start_save_pct=0.10):
    if iter > max_iters * start_save_pct:
        if current < best:
            save_path = os.path.join(out_dir, exp_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(save_path, "wb") as f:
                pickle.dump(params, f)
                best = current
        return best

if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)
    init_rng, dropout_rng, mask_rng = jax.random.split(rng, 3)
    ds_train, ds_test, n_classes = load_data(dataset)
    img_x, text_x = next(ds_train)

    config = CLIPConfig(
        img_shape=img_shape,
        patch_size=patch_size,
        vocab_size=vocab_size,
        block_size=block_size,
        n_classes=n_classes,
        n_patch=(img_shape[0]//patch_size)**2,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_embd_proj=n_embd_proj,
        dropout=dropout,
        use_bias=use_bias,
    )

    model = CLIP(config)
    params = model.init(rng, img_x, text_x, train=False)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )

    def cross_entropy_loss(logits, labels, axis=0):
        return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=axis), axis=axis)

    def loss_fn(params, img, text, train=True, rngs=None):
        logits = train_state.apply_fn(params, img, text, train=train, rngs=rngs)
        labels = jnp.arange(logits.shape[0])
        loss_i = cross_entropy_loss(logits, labels, axis=0)
        loss_t = cross_entropy_loss(logits, labels, axis=1)
        loss = (loss_i / loss_t) / 2
        return loss.mean()

    grad_fn = jax.value_and_grad(loss_fn)
    eval_fn = jax.jit(partial(loss_fn, train=False))

    @jax.jit
    def update(train_state, img, text, rng):
        rng, dropout_rng = jax.random.split(rng)
        rngs={"dropout": dropout_rng}
        loss, grads = grad_fn(train_state.params, img, text, rngs=rngs)
        updated_train_state = train_state.apply_gradients(grads=grads)
        return updated_train_state, loss, rng

    def evaluate(train_state, mask_rng):
        losses = []
        for _ in range(eval_iters):
            img_y, text_y = next(ds_test)
            loss = eval_fn(train_state.params, img_y, text_y)
            losses.append(loss)
            return np.mean(losses)

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=exp_name, config=config)

    best_val_loss = jnp.inf
    for iter in range(max_iters):
        img_x, text_x = next(ds_train)
        train_state, train_loss, rng = update(train_state, img_x, text_x, rng)

        if iter % eval_interavl == 0:
            val_loss = evaluate(train_state, rng)
            best_val_loss = save_best(val_loss, best_val_loss, train_state.params)
            print(f"{iter}: train_loss {train_loss:.4f} val_loss {val_loss:.3f}")

            if wandb_log:
                wandb.log({"iter": iter, "train/loss": train_loss, "val/loss": val_loss})
