import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import os
import tyro
from rich.pretty import pprint
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Callable

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
from nanax_utils.visualize import save_image
from nanax_utils.tree_utils import last, mean

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
    
    latents: int = 20
    learning_rate: float = 0.001
    batch_size: int = 128
    n_epochs: int = 30

    world_size: int = None
    distributed: bool = False
    local_rank: int = 0
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    global_learner_devices: List[int] = field(default_factory=lambda: [0])


@struct.dataclass
class Metrics:
    loss: float
    recon_loss: float
    kl_loss: float


class Encoder(nn.Module):
  """VAE Encoder."""

  latents: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(500)(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents)(x)
    logvar_x = nn.Dense(self.latents)(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
  """VAE Decoder."""

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(500)(z)
    z = nn.relu(z)
    z = nn.Dense(784)(z)
    return z


class VAE(nn.Module):
  """Full VAE model."""

  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = self.reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))

  def reparameterize(self, rng, mean, logvar):
      std = jnp.exp(0.5 * logvar)
      eps = jax.random.normal(rng, logvar.shape)
      return mean + eps * std

@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(
        labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits))
    )

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def compute_metrics(recon_x, x, mean, logvar):
    recon_loss = binary_cross_entropy_with_logits(recon_x, x)
    kl_loss = kl_divergence(mean, logvar)
    return Metrics(loss=recon_loss + kl_loss, recon_loss=recon_loss, kl_loss=kl_loss)

def loss_fn(params, batch):
    recon_x, mean, logvar = state.apply_fn(params, batch.image, batch.rng)
    x = batch.image.reshape((args.batch_size, -1))
    recon_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
    kl_loss = kl_divergence(mean, logvar).mean()
    loss = recon_loss + kl_loss
    return loss, (recon_loss, kl_loss)

grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

@jax.jit
def train_epoch(state, dataset, rng, latents):
    def train_step(state, batch):
        (loss, (recon_loss, kl_loss)), grads = grad_fn(state.params, batch)
        state = state.apply_gradients(grads=grads)
        return state, Metrics(loss=loss, recon_loss=recon_loss, kl_loss=kl_loss)
    batches = sample_batches(dataset, args.batch_size, rng)
    state, metrics = jax.lax.scan(train_step, state, batches)
    return state, metrics
  
@jax.jit
def evaluate(params, images, z, z_rng):
    def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate([
            images.reshape(-1, 28, 28, 1),
            recon_images.reshape(-1, 28, 28, 1),
        ])

        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images

    return nn.apply(eval_model, VAE(args.latents))(params)

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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    rng = jax.random.key(0)
    rng, key = jax.random.split(rng)

    (train_ds, test_ds), n_classes, n_batches = get_dataset(args.dataset, args.batch_size, key, augmentations=['binarize'])
    assert n_classes == args.n_classes, "Dataset does not match n_classes"
    
    network = VAE(args.latents)
    state = TrainState.create(
        apply_fn=network.apply,
        params=network.init(key, train_ds.image[0][None,], rng),
        tx=optax.adam(args.learning_rate),
    )

    rng, z_key = jax.random.split(rng)
    z = jax.random.normal(z_key, (64, args.latents))
    example_idx = jnp.array([jnp.where(test_ds.label == c)[0][0] for c in range(args.n_classes)])
    eval_examples = test_ds.image[example_idx]

    for epoch in range(args.n_epochs):
        rng, key = jax.random.split(rng)
        state, train_metrics = train_epoch(state, train_ds, key, args.latents)

        eval_metrics, comparison, sample = evaluate(state.params, eval_examples, z, key)
        save_image(comparison, f'{args.run_name}/reconstruction_{epoch}.png', nrow=10)
        save_image(sample, f'{args.run_name}/sample_{epoch}.png', nrow=10)
        
        print(f"{epoch + 1}: Train {last(train_metrics)}, Test {mean(eval_metrics)}")
        for name, metrics in zip(["train", "test"], [train_metrics, eval_metrics]):
            for step, (loss, recon_loss, kl_loss) in enumerate(zip(metrics.loss, metrics.recon_loss, metrics.kl_loss)):
                writer.add_scalar(f"{name}/loss", loss, (epoch * n_batches) + step)
                writer.add_scalar(f"{name}/recon_loss", recon_loss, (epoch * n_batches) + step)
                writer.add_scalar(f"{name}/kl_loss", kl_loss, (epoch * n_batches) + step)