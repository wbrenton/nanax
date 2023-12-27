import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from jax.nn.initializers import Initializer

from typing import Optional

from utils import get_2d_sincos_pos_embed, apply_masks, repeat_interleave_batch


@struct.dataclass
class iJEPAConfig:
    img_shape: int = (28, 28, 1)
    patch_size: int = 4
    n_classes: int = 10
    n_patch: int = (28//4)**2
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    predictor_n_embd: int = 128
    n_pred: int = 4
    dropout: float = 0.1
    use_bias: bool = False
    use_cls_token: bool = False
    dtype: Optional[str] = jnp.float32


class Attention(nn.Module):
    embed_dim: int
    n_head: int
    use_bias: bool
    dropout: float
    dtype: Optional[str]

    @nn.compact
    def __call__(self, x, train):
        B, T, C = x.shape
        assert C % self.n_head == 0, "Embedding dimensionality must be evenly divible by the number of heads"
        head_dim = self.embed_dim // self.n_head
        c_attn = nn.Dense(3 * C, use_bias=self.use_bias, dtype=self.dtype)(x) # (B, T, 3*C)
        q, k, v = jnp.split(c_attn, 3, axis=-1) # (B, T, C)
        q, k, v = map(lambda arr: arr.reshape(B, T, self.n_head, head_dim).swapaxes(1, 2), [q, k, v]) # (B, nh, T, hd)
        attn = q @ k.swapaxes(-2, -1) / jnp.sqrt(k.shape[-1]) # (B, nh, T, hd) @ (B, nh, hd, T) -> (B, nh, T, T)
        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.dropout)(attn, deterministic=not train)
        y = attn @ v # (B, nh, T, T) @ (B, T, C) -> (B, nh, T, hd)
        y = y.swapaxes(1, 2).reshape(B, T, C) # (B, T, nh, hd) -> (B, T, C)
        c_proj = nn.Dense(C, use_bias=self.use_bias, dtype=self.dtype)(y) # (B, T, C)
        x = nn.Dropout(self.dropout)(c_proj, deterministic=not train)
        return x


class MLP(nn.Module):
    dropout: float
    use_bias: bool
    dtype: Optional[str]

    @nn.compact
    def __call__(self, x, train=True):
        _, _, C = x.shape
        x = nn.Dense(C * 4, use_bias=self.use_bias, dtype=self.dtype)(x)
        x = jax.nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        x = nn.Dense(C, use_bias=self.use_bias, dtype=self.dtype)(x)
        x = nn.Dropout(self.dropout)(x, deterministic=not train)
        return x


class Block(nn.Module):
    n_embd: int
    n_head: int
    dropout: float
    use_bias: bool
    dtype: Optional[str]

    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=1e-6, use_bias=self.use_bias, dtype=self.dtype)
        self.attn = Attention(self.n_embd, self.n_head, self.use_bias, self.dropout, dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=1e-6, use_bias=self.use_bias, dtype=self.dtype)
        self.mlp = MLP(self.dropout, self.use_bias, self.dtype)

    def __call__(self, x, train):
        x = x + self.attn(self.ln_1(x), train)
        x = x + self.mlp(self.ln_2(x), train)
        return x

class PatchEmbed(nn.Module):
    patch_size: int
    n_embd: int

    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        x = nn.Conv(
            self.n_embd,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size)
        )(x)
        return x.reshape(B, -1, self.n_embd)


def custom_2d_sincos_initializer(embed_dim, grid_size, cls_token) -> Initializer:

    def init(key, shape, dtype):
        pos_embd = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)
        assert pos_embd.shape == shape, "Expected shape {shape} but got {pos_embd.shape}"
        return pos_embd

    return init

class ViTPredictor(nn.Module):
    config: iJEPAConfig

    def setup(self):
        c = self.config
        self.pred_embed = nn.Dense(c.predictor_n_embd)
        self.mask_token = nn.Embed(1, c.predictor_n_embd)
        pos_embd_init = custom_2d_sincos_initializer(
                c.predictor_n_embd,
                c.img_shape[0]//c.patch_size,
                c.use_cls_token)
        self._pos_embd = nn.Embed(
            num_embeddings=c.n_patch,
            features=c.predictor_n_embd,
            dtype=c.dtype,
            embedding_init=pos_embd_init,
            name="pos_embed")
        self.drop = nn.Dropout(c.dropout)
        self.h = [
            Block(
                c.predictor_n_embd,
                c.n_head,
                c.dropout,
                c.use_bias,
                c.dtype
            ) for _ in range(c.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.pred_proj = nn.Dense(c.n_embd, use_bias=True, dtype=c.dtype)

    def __call__(self, x, context_mask, target_mask, train):
        x = self.pred_embed(x)
        x = x + self.pos_embd(context_mask)
        B, N_ctxt, D = x.shape
        pos_embd = self.pos_embd(target_mask)
        pred_tokens = self.mask_token(jnp.arange(1))
        pred_tokens = pred_tokens[None,].repeat(B, axis=0)
        pred_tokens = pred_tokens.repeat(self.config.n_pred, axis=1)
        pred_tokens += pos_embd
        x = jnp.concatenate([x, pred_tokens], axis=1)
        # x = self.drop(x, deterministic=not train)
        for block in self.h:
            x = block(x, train)
        x = self.ln_f(x)
        x = x[:, N_ctxt:]
        x = self.pred_proj(x)
        return x

    def pos_embd(self, index):
        return jax.vmap(self._pos_embd)(index)


class ViT(nn.Module):
    config: iJEPAConfig

    def setup(self):
        c = self.config
        self.patch_embd = PatchEmbed(c.patch_size, c.n_embd)
        pos_embd_init = custom_2d_sincos_initializer(
                c.n_embd,
                c.img_shape[0]//c.patch_size,
                c.use_cls_token)
        self.pos_embd = nn.Embed(
            num_embeddings=c.n_patch,
            features=c.n_embd,
            dtype=c.dtype,
            embedding_init=pos_embd_init,
            name="pos_embed")
        self.drop = nn.Dropout(c.dropout)
        self.h = [
            Block(
                c.n_embd,
                c.n_head,
                c.dropout,
                c.use_bias,
                c.dtype
            ) for _ in range(c.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.linear_head = nn.Dense(c.n_classes, use_bias=c.use_bias, dtype=c.dtype)

    def __call__(self, x, mask=None, train=None):
        x = self.patch_embd(x)
        B, T, D = x.shape
        x = x + self.pos_embd(jnp.arange(T))
        if mask is not None:
            x = apply_masks(x, mask)
        x = self.drop(x, deterministic=not train)
        for block in self.h:
            x = block(x, train)
        x = self.ln_f(x)
        return x


class iJEPA(nn.Module):
    """
    The goal of Image-JEPA is "given a context block, predict the
    representations of various target blocks in the same image."
    - Assran et al. 2023 Section 3: Methods
    
    Here instead of using blocks (randomly selected subsets of
    images) than can overlap we partition naively over the patches.
    """
    config: iJEPAConfig

    def setup(self) -> None:
        self.context_encoder = ViT(self.config)
        self.predictor = ViTPredictor(self.config)
        self.target_encoder = ViT(self.config)

    def __call__(self, imgs, rng, train=None):
        latent_target, target_mask, context_mask = self.target(imgs, rng, train=train)
        latent_pred = self.context(imgs, context_mask, target_mask)
        return latent_target, latent_pred

    def context(self, imgs, context_mask, target_mask, train=None):
        """ 
        Context encoder produces latent representation conditioned
        only on the context patches. Predictor takes context encoder
        output, concatenates the missing target patches positional
        encoding and predicts latent representation of the target
        patches. 
        """
        z = self.context_encoder(imgs, context_mask, train=train)
        z = self.predictor(z, context_mask, target_mask, train=train)
        return z

    def target(self, imgs, rng, train=None):
        """
        Target encoder is used to obtain the 'true' latent representation
        of unmasked images. The target patches representation are indexed
        and returned to be used in latent reconstruction loss.
        """
        h = self.target_encoder(imgs, train=train)
        B, T, _ = h.shape
        target_mask, context_mask = get_masks(B, T, self.config.n_pred, rng)
        targets = apply_masks(h, target_mask)
        return jax.lax.stop_gradient(targets), target_mask, context_mask

def get_masks(n_batch, n_patch, n_pred, rng):
    """
    Randomly drops n_pred patches from jnp.arange(n_patch)
    Returns indices of dropped (target) patches, and remaining (context) patches
    """
    @jax.vmap
    def batch_permute(rng, x):
        return jax.random.permutation(rng, x)

    rngs = jax.random.split(rng, n_batch)
    token_indicies = jnp.arange(n_patch)[None,]
    token_indicies = token_indicies.repeat(n_batch, axis=0)
    shuffled_indices = batch_permute(rngs, token_indicies)
    target_mask = shuffled_indices[:, :n_pred]
    context_mask = shuffled_indices[:, n_pred:]
    return target_mask, context_mask

def apply_masks(arr, mask_indicies):
    @jax.vmap
    def batch(patch_arr, patch_indices):
        @jax.vmap
        def indexer(t):
            return patch_arr[t]
        return indexer(patch_indices)
    
    return batch(arr, mask_indicies)