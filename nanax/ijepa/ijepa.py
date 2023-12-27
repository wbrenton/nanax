import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import os
import copy
import tyro
from rich.pretty import pprint
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Callable
from functools import partial

import jax
import flax
import optax
import numpy as np
import jax.numpy as jnp
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
    run_name: tyro.conf.Suppress[str] = None
    
    # dataset
    dataset: str = "mnist"
    tfds_data_dir: str = "/admin/home-willb/tensorflow_datasets/"
    n_classes: int = 10
    tensorboard_dir: str = "tuning_logs/mnist/"
    
    # training
    learning_rate: float = 0.001
    batch_size: int = 128
    n_epochs: int = 30

    # masking
    crop_size: int = 28
    patch_size: int = 4
    ctxt_size: int = 6
    mask_size: int = 3

    # encoder
    embed_dim: int = 32
    depth: int = 1
    n_heads: int = 4
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    
    # predictor
    pred_embed_dim: int = 16
    pred_depth: int = 2
    pred_n_heads: int = 8
    allow_overlap: bool = False # this isn't implemented (allows the encoder and prediction masks to overlap, which would make the prediction task easier via copying)
    min_keep: int = 10

    world_size: int = None
    distributed: bool = False
    local_rank: int = 0
    learner_device_ids: List[int] = field(default_factory=lambda: [0])
    global_learner_devices: List[int] = field(default_factory=lambda: [0])


def apply_masks(x, masks):
    return jnp.take(x, masks, axis=0)

def dpr_fn(depth, drop_path_rate):
    return np.linspace(0, drop_path_rate, depth)

@flax.struct.dataclass
class Metrics:
    loss: float


class PatchEmbed(nn.Module):
    image_size: int
    patch_size: int
    embed_dim: int

    def setup(self):
        self.n_patches = (self.image_size // self.patch_size) ** 2
        patch_shape = (self.patch_size, self.patch_size)
        self.patch_proj = nn.Conv(self.embed_dim, kernel_size=patch_shape, strides=patch_shape)
    
    def __call__(self, x):
        return self.patch_proj(x).reshape(x.shape[0], -1, self.embed_dim)


class Attention(nn.Module):
    dim: int
    n_heads: int
    qkv_bias: bool
    qk_scale: float
    attn_drop: Callable
    proj_drop: Callable

    def setup(self):
        head_dim = self.dim // self.n_heads
        self.scale = self.qk_scale or head_dim ** -0.5
        self.qkv = nn.Dense(3 * self.dim, use_bias=self.qkv_bias)
        self.proj = nn.Dense(self.dim)

    def __call__(self, x, training=True):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, C // self.n_heads) # (B, N, 3, nh, hd)
        qkv = qkv.transpose(2, 0, 3, 1, 4) # (3, B, nh, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale # (B, nh, N, hd) @ (B, nh, hd, N) -> (B, nh, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, deterministic=not training)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C) # (B, nh, N, N) @ (B, nh, N, hd) -> (B, N, nh, hd) -> (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x, deterministic=not training)
        return x, attn


class MLP(nn.Module):
    hidden: int
    ratio: float
    drop: float
    act: Callable = nn.gelu

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Dense(int(self.hidden * self.ratio))(x)
        x = self.act(x)
        x = nn.Dropout(self.drop)(x, deterministic=not training)
        x = nn.Dense(self.hidden)(x)
        x = nn.Dropout(self.drop)(x, deterministic=not training)
        return x


class Block(nn.Module):
    dim: int
    n_heads: int
    mlp_ratio: float
    qkv_bias: bool
    qk_scale: float
    drop: float
    attn_drop: float
    drop_path: float

    def setup(self):
        self.norm1 = nn.LayerNorm(epsilon=1e-6)
        self.attn = Attention(self.dim, self.n_heads, self.qkv_bias, self.qk_scale, nn.Dropout(self.attn_drop), nn.Dropout(self.drop))
        self.dropout = nn.Dropout(self.drop) if self.drop_path > 0 else lambda x, deterministic: x
        self.norm2 = nn.LayerNorm(epsilon=1e-6)
        self.mlp = MLP(self.dim, self.mlp_ratio, drop=self.drop)

    def __call__(self, x, training=True):
        y, attn = self.attn(self.norm1(x), training=training)
        x = x + self.dropout(y, deterministic=not training)
        x = x + self.dropout(self.mlp(self.norm2(x), training=training), deterministic=not training)
        return x


class Encoder(nn.Module):
    image_size: int = 28
    patch_size: int = 4
    embed_dim: int = 32
    depth: int = 1
    n_heads: int = 3
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dpr: jax.Array = dpr_fn(depth, drop_rate)

    def setup(self):
        self.patch_embed = PatchEmbed(self.image_size, self.patch_size, self.embed_dim)
        self.n_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Embed(self.n_patches, self.embed_dim, name="pos_embed")
        self.blocks = nn.Sequential([
            Block(
                self.embed_dim, self.n_heads, self.mlp_ratio, self.qkv_bias,
                self.qk_scale, self.drop_rate, self.attn_drop_rate, drop_path=self.dpr[i],
            ) for i in range(self.depth)
        ])
        self.norm = nn.LayerNorm(epsilon=1e-6)

    def __call__(self, x, masks=None, training=True):
        x = self.patch_embed(x)
        x = self.add_pos_encoding(x)
        if masks is not None:
            x = jax.vmap(apply_masks)(x, masks)
        x = self.blocks(x, training)
        x = self.norm(x)
        return x

    def add_pos_encoding(self, x):
        idx = jnp.arange(self.n_patches)
        return x + self.pos_embed(idx)


class Predictor(nn.Module):
    n_patches: int = 28
    enc_embed_dim: int = 32
    embed_dim: int = 16
    depth: int = 6
    n_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dpr: jax.Array = dpr_fn(depth, drop_rate)

    def setup(self):
        self.pred_embed = nn.Dense(self.embed_dim)
        self.mask_token = nn.Embed(1, self.embed_dim, name="mask_token")
        self.pos_embed = nn.Embed(self.n_patches, self.embed_dim, name="pos_embed") # +1 for cls token
        self.blocks = nn.Sequential([
            Block(
                self.embed_dim, self.n_heads, self.mlp_ratio, self.qkv_bias,
                self.qk_scale, self.drop_rate, self.attn_drop_rate, drop_path=self.dpr[i],
            ) for i in range(self.depth)
        ])
        self.norm = nn.LayerNorm(epsilon=1e-6)
        self.pred_proj = nn.Dense(self.enc_embed_dim)

    def __call__(self, x, masks_x, masks, training=True):
        x = self.pred_embed(x)
        x = self.add_pos_encoding(x, masks_x)
        
        N_ctxt = x.shape[1]
        x = self.add_mask_token(x, masks)
        
        x = self.blocks(x, training)
        x = self.norm(x)
        
        x = x[:, N_ctxt:]
        return self.pred_proj(x)

    def add_mask_token(self, x, masks):
        idx = jnp.arange(self.n_patches)
        pos_embed = apply_masks(self.pos_embed(idx), masks)
        pred_token = self.mask_token(jnp.arange(1))
        pred_token = jnp.tile(pred_token, pos_embed.shape[0:2] + (1,))
        pred_token += pos_embed
        return jnp.concatenate([x, pred_token], axis=1)

    def add_pos_encoding(self, x, masks):
        idx = jnp.arange(self.n_patches)
        return x + apply_masks(self.pos_embed(idx), masks)


class ImageJEPA(nn.Module):
    args: Args
    
    def setup(self):
        self.encoder = Encoder(
            image_size=args.crop_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            n_heads=args.n_heads,
        )
        self.predictor = Predictor(
            n_patches=args.n_patches,
            enc_embed_dim=args.embed_dim,
            embed_dim=args.pred_embed_dim,
            depth=args.pred_depth,
            n_heads=args.pred_n_heads,
        )
        self.target_encoder = Encoder(
            image_size=args.crop_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            n_heads=args.n_heads,
        )
        
    def __call__(self, x, masks_x, masks):
        h = self.target_representation(x, masks)
        z = self.predicted_representation(x, masks_x, masks)
        return z, h

    def target_representation(self, x, masks):
        h = self.target_encoder(x)
        h = jax.vmap(apply_masks)(h, masks)
        return jax.lax.stop_gradient(h)

    def predicted_representation(self, x, masks_x, masks):
        z = self.encoder(x, masks_x)
        z = self.predictor(z, masks_x, masks)
        return z


def train(args: Args):
    args.run_name = f"{args.exp_name}_{args.dataset}_{args.seed}"
    args.n_patches = (args.crop_size // args.patch_size) ** 2
    args.patch_grid_size = args.n_patches ** 0.5
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

    def loss_fn(params, image, masks_x, masks):
        z, h = jepa.apply(params, image, masks_x, masks)
        loss = optax.huber_loss(z, h) # NOTE: when moving to distributed take pmean here, not over grads
        return loss

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

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, data_rng = jax.random.split(rng, 3)

    (train_ds, test_ds), n_classes, n_batches = get_dataset(args.dataset, args.batch_size, data_rng)
    assert n_classes == args.n_classes, "Dataset does not match n_classes"
    
    jepa = ImageJEPA(args)
    
    image = train_ds.image[0][None, :]
    pred_idxs = jnp.arange(args.mask_size ** 2, dtype=jnp.int32)[None, :]
    ctxt_idxs = jnp.arange(args.mask_size ** 2, args.patch_grid_size ** 2, dtype=jnp.int32)[None, :]

    state = TrainState.create(
        apply_fn=jepa.apply,
        params=jepa.init({"params": init_rng, "dropout": init_rng}, image, ctxt_idxs, pred_idxs),
        tx=optax.adam(args.learning_rate),
    )

    best = []
    importance_mask = np.arange(1, 6)
    last = lambda x: jax.tree_map(lambda y: y[-1].item(), x)
    for epoch in range(args.n_epochs):
        rng, train_rng, test_rng = jax.random.split(rng, 3)
        state, train_metrics = train_epoch(state, train_ds, train_rng)
        _, test_metrics = test_epoch(state, test_ds, test_rng)

        print(f"{epoch + 1}: Train {last(train_metrics)}, Test {last(test_metrics)}")
        for name, metrics in zip(["train", "test"], [train_metrics, test_metrics]):
            for step, (loss, accuracy) in enumerate(zip(metrics.loss, metrics.accuracy)):
                writer.add_scalar(f"{name}/loss", loss, (epoch * n_batches) + step)
                writer.add_scalar(f"{name}/accuracy", accuracy, (epoch * n_batches) + step)

        if epoch >= args.n_epochs - 5:
            best.append(last(test_metrics.accuracy))

    best = np.array(best)
    mask = importance_mask / importance_mask.sum()
    importance = (best * mask).sum()
    return importance

if __name__ == "__main__":
    args = tyro.cli(Args)
    importance = train(args)
    print(importance)