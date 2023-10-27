import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from typing import Optional


@struct.dataclass
class ViTConfig:
    img_shape: int = (28, 28, 1)
    patch_size: int = 4
    n_classes: int = 10
    n_patch: int = (28//4)**2
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    use_bias: bool = False
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
    config: ViTConfig

    def setup(self):
        c = self.config
        self.ln_1 = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.attn = Attention(c.n_embd, c.n_head, c.use_bias, c.dropout, dtype=c.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.mlp = MLP(c.dropout, c.use_bias, c.dtype)

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

class ViT(nn.Module):
    config: ViTConfig

    def setup(self):
        c = self.config
        self.patch_embd = PatchEmbed(c.patch_size, c.n_embd)
        self.cls_token = nn.Embed(
            num_embeddings=1,
            features=c.n_embd,
            dtype=c.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="cls_token")
        self.pos_embd = nn.Embed(
            num_embeddings=1+c.n_patch,
            features=c.n_embd,
            dtype=c.dtype,
            embedding_init=nn.initializers.normal(stddev=1.0),
            name="pos_embed")
        self.drop = nn.Dropout(c.dropout)
        self.h = [Block(c) for _ in range(c.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.linear_head = nn.Dense(c.n_classes, use_bias=c.use_bias, dtype=c.dtype)

    def __call__(self, x, train):
        x = self.patch_embd(x)
        b, t, _ = x.shape
        cls_token = self.cls_token(jnp.arange(1)[None])
        cls_token = cls_token.repeat(b, axis=0)
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.pos_embd(jnp.arange(t+1))
        x = self.drop(x, deterministic=not train)
        for block in self.h:
            x = block(x, train)
        x = self.ln_f(x)
        cls = x[:,0]
        logits = self.linear_head(cls)
        return logits