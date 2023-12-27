import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import os
import tyro
from rich.pretty import pprint
from dataclasses import dataclass, field
from typing import List, Sequence, Callable

import jax
import optax
import numpy as np
import jax.numpy as jnp
from flax import struct
from flax import linen as nn
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

import nanax_utils.xla_determinism
from nanax_utils.datasets import get_dataset, sample_batches

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int = 1
    track: bool = False
    wandb_project_name: str = "nanax"
    dataset: str = "mnist"
    n_classes: int = 10
    run_name: tyro.conf.Suppress[str] = None
    tensorboard_dir: str = "tuning_logs/mnist/"
    
    learning_rate: float = 0.01
    batch_size: int = 128
    n_epochs: int = 10
    hiddens: Sequence[int] = (300, 100)
    activation_fn = jax.nn.relu

    world_size: int = None
    distributed: bool = False
    local_rank: int = 0
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    global_learner_devices: List[int] = field(default_factory=lambda: [0])

@struct.dataclass
class Metrics:
    loss: float
    accuracy: float


class MLP(nn.Module):
  """A simple MLP model."""
  n_classes: int
  hiddens: int
  activation_fn: Callable

  @nn.compact
  def __call__(self, images: jax.Array):
    x = images.astype(jnp.float32) / 255.
    x = x.reshape((x.shape[0], -1))
    for hidden in self.hiddens:
        x = nn.Dense(hidden)(x)
        x = self.activation_fn(x)
    logits = nn.Dense(self.n_classes)(x)
    return logits


def loss_fn(params, batch):
    logits = state.apply_fn(params, batch.image)
    one_hot = jax.nn.one_hot(batch.label, args.n_classes)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch.label)
    return loss, accuracy

@jax.jit
def train_epoch(state, dataset, rng):

    def train_step(state, batch):
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, Metrics(loss=loss, accuracy=accuracy)

    batches = sample_batches(dataset, args.batch_size, rng)
    state, epoch_metrics = jax.lax.scan(train_step, state, batches)
    return state, epoch_metrics

@jax.jit
def test_epoch(state, dataset, rng):

    def test_step(state, batch):
        loss, accuracy = loss_fn(state.params, batch)
        return state, Metrics(loss=loss, accuracy=accuracy)

    batches = sample_batches(dataset, args.batch_size, rng)
    state, epoch_metrics = jax.lax.scan(test_step, state, batches)
    return state, epoch_metrics

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.run_name = f"{args.exp_name}_{args.dataset}_{args.seed}"
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
        pprint(args)
    writer = SummaryWriter(f"{args.tensorboard_dir}/")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{rng}|{value}|" for rng, value in vars(args).items()])),
    )
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    (train_ds, test_ds), n_classes, n_batches = get_dataset(args.dataset, args.batch_size, rng)
    assert n_classes == args.n_classes, "Dataset does not match n_classes"

    network = MLP(args.n_classes, args.hiddens, args.activation_fn)
    state = TrainState.create(
        apply_fn=network.apply,
        params=network.init(init_rng, train_ds.image[0][None,]),
        tx=optax.adam(args.learning_rate),
    )

    last = lambda x: jax.tree_map(lambda y: y[-1].item(), x)
    mean = lambda x: jax.tree_map(lambda y: y.mean().item(), x)
    for epoch in range(args.n_epochs):
        rng, train_rng, test_rng = jax.random.split(rng, 3)
        state, train_metrics = train_epoch(state, train_ds, train_rng)
        _, test_metrics = test_epoch(state, test_ds, test_rng)

        print(f"{epoch + 1}: Train {last(train_metrics)}, Test {last(test_metrics)}")
        writer.add_scalar(f"test/loss", test_metrics.loss.mean(), epoch)
        writer.add_scalar(f"test/accuracy", test_metrics.accuracy.mean(), epoch)
        for step, (loss, accuracy) in enumerate(zip(train_metrics.loss, train_metrics.accuracy)):
            writer.add_scalar(f"train/loss", loss, (epoch * n_batches) + step)
            writer.add_scalar(f"train/accuracy", accuracy, (epoch * n_batches) + step)
