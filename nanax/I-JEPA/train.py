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

from model import iJEPA, iJEPAConfig

n_layer = 6
n_head = 4
n_embd = 128
predictor_n_embd = 64
patch_size = 7
n_enc_masks = 1
n_pred_masks = 4
num_classes = 10
dropout = 0.1
use_bias = True
batch_size = 512
img_shape = (28, 28)
learning_rate = 3e-4
ewa = 0.996
max_iters = 100000
eval_interavl = 500
eval_iters = 10

dataset = 'mnist'
out_dir = 'out'
wandb_log = False
wandb_project = "nanax"
exp_name = "I-JEPA" + f"-{dataset}-" + str(time.time())
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

def conditional_save(current, best, params, start_pct):
    if iter > max_iters * start_pct:
        if current < best:
            with open(os.path.join(out_dir, exp_name), "wb") as f:
                pickle.dump(params, f)
                best = current
    return best

if __name__ == "__main__":

    rng = jax.random.PRNGKey(0)
    init_rng, dropout_rng, mask_rng = jax.random.split(rng, 3)
    ds_train, ds_test, n_classes = load_data(dataset)
    xb, xy = next(ds_train)

    config = iJEPAConfig(
        img_shape=img_shape,
        patch_size=patch_size,
        n_classes=n_classes,
        n_patch=(img_shape[0]//patch_size)**2,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        predictor_n_embd=predictor_n_embd,
        dropout=dropout,
        use_bias=use_bias,
    )

    model = iJEPA(config)
    params = model.init(rng, xb, mask_rng, train=False)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate),
    )

    def loss_fn(params, x, mask_rng, train=True, rngs=None):
        h, z = train_state.apply_fn(params, x, mask_rng, train=train, rngs=rngs)
        loss = jnp.abs(h - z).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    eval_fn = jax.jit(partial(loss_fn, train=False))

    @jax.jit
    def update(train_state, x, y, rng):
        rng, mask_rng, dropout_rng = jax.random.split(rng, 3)
        rngs={"dropout": dropout_rng}
        loss, grads = grad_fn(train_state.params, x, mask_rng, rngs=rngs)
        updated_train_state = train_state.apply_gradients(grads=grads)
        updated_params = unfreeze(updated_train_state.params)
        updated_params['params']['target_encoder'] = optax.incremental_update(
                updated_train_state.params['params']['target_encoder'],
                train_state.params['params']['target_encoder'],
                ewa)
        updated_train_state = updated_train_state.replace(
            params=freeze(updated_params))
        return updated_train_state, loss, rng

    def evaluate(train_state, mask_rng):
        losses = []
        for _ in range(eval_iters):
            xb, xy = next(ds_test)
            loss = eval_fn(train_state.params, xb, mask_rng)
            losses.append(loss)
            return np.mean(losses)

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=exp_name, config=config)

    best_val_loss = jnp.inf
    for iter in range(max_iters):
        xb, xy = next(ds_train)
        train_state, train_loss, rng = update(train_state, xb, xy, rng)

        if iter % eval_interavl == 0:
            val_loss = evaluate(train_state, rng)
            print(f"{iter}: train_loss {train_loss:.4f} val_loss {val_loss:.3f}")
            best_val_loss = conditional_save(val_loss, best_val_loss, train_state.params)

            if wandb_log:
                wandb.log({"iter": iter, "train/loss": train_loss, "val/loss": val_loss})