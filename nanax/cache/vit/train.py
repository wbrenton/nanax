import os
import time
from functools import partial

import jax
import optax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as orbax
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from datasets import load_dataset

from model import ViTConfig, ViT

n_layer = 6
n_head = 4
n_embd = 128
patch_size = 4
num_classes = 10
dropout = 0.1
use_bias = True
batch_size = 128
learning_rate = 3e-4
max_iters = 10000
eval_interavl = 500
eval_iters = 200

dataset = 'mnist'
out_dir = 'out'
wandb_log = True
wandb_project = "nanax"
wandb_run_name = "ViT" + f"-{dataset}-" + str(time.time())
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} # will be useful for logging

def load_data(name: str):
    ds = load_dataset(name)
    n_classes = len(ds['train'].features['label'].names)
    ds = ds.with_format("jax")
    ds = ds.shuffle()

    def create_iter(dataset):
        while True:
            for i, batch in enumerate(dataset.iter(batch_size=batch_size)):
                batch['count'] = i
                xb, xy = batch['image'], batch['label']
                xb = jnp.expand_dims(xb, axis=-1) if len(xb) != 4 else xb
                yield xb, xy

    train_iter = create_iter(ds['train'])
    test_iter = create_iter(ds['test'])

    return train_iter, test_iter, n_classes

checkpoint_path = os.path.join(out_dir, 'checkpoint')
checkpoint_manager = orbax.CheckpointManager(
    checkpoint_path,
    checkpointers=orbax.Checkpointer(orbax.PyTreeCheckpointHandler()),
    options=orbax.CheckpointManagerOptions(
        max_to_keep=2,
        best_fn=lambda c: c['val_loss'],
        best_mode='min',
        keep_checkpoints_without_metrics=False,
        create=True,
    ),
)

ds_train, ds_test, n_classes = load_data(dataset)

rng = jax.random.PRNGKey(0)
init_rng, dropout_rng = jax.random.split(rng, 2)
xb, _ = next(ds_train)
img_shape = xb.shape[1:]

config = ViTConfig(
    img_shape=img_shape,
    patch_size=patch_size,
    n_classes=n_classes,
    n_patch=(img_shape[0]//patch_size)**2,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    use_bias=use_bias,
)

vit = ViT(config)
vit_params = vit.init(rng, xb, train=False)
train_state = TrainState.create(
    apply_fn=vit.apply,
    params=vit_params,
    tx=optax.adam(learning_rate),
)

def loss_fn(params, x, y, train=True, rngs=None):
    logits = train_state.apply_fn(params, x, train=train, rngs=rngs)
    one_hot = jax.nn.one_hot(y, n_classes)
    loss = -jnp.sum(one_hot * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    acc = jnp.mean(y == jax.nn.softmax(logits).argmax(-1))
    return loss.mean(), acc

grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
eval_fn = jax.jit(partial(loss_fn, train=False))

@jax.jit
def update(train_state, x, y, rng):
    rng, dropout_rng = jax.random.split(rng)
    rngs={"dropout": dropout_rng}
    (loss, acc), grads = grad_fn(train_state.params, x, y, rngs=rngs)
    train_state = train_state.apply_gradients(grads=grads)
    return train_state, loss, acc, rng

def evaluate(train_state):
    losses = []
    accuracies = []
    for _ in range(eval_iters):
        xb, xy = next(ds_test)
        loss, acc = eval_fn(train_state.params, xb, xy)
        losses.append(loss)
        accuracies.append(acc)
        return np.mean(losses), np.mean(accuracies)

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

for iter in range(max_iters):
    xb, xy = next(ds_train)
    train_state, train_loss, train_acc, rng = update(train_state, xb, xy, rng)
    if iter % eval_interavl == 0:
        val_loss, val_acc = evaluate(train_state)
        print(f"{iter}: train_loss {train_loss:.4f} val_loss {val_loss:.3f} train_acc {train_acc:.4f} val_acc {val_acc:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/accuracy": train_acc, 
                "val/accuracy": val_acc})

checkpoint = ({'state': train_state, 'config': config})
checkpoint_manager.save(step=iter, items=checkpoint, save_kwargs=orbax_utils.save_args_from_target(checkpoint))