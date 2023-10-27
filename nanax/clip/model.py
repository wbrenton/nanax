import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from jax.nn.initializers import Initializer

from typing import Optional


@struct.dataclass
class CLIPConfig:
    img_shape: int = (28, 28, 1)
    patch_size: int = 4
    vocab_size: int = 16
    block_size: int = 6
    n_classes: int = 10
    n_patch: int = (28//4)**2
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    n_embd_proj: int = 128
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


class Transformer(nn.Module):
    config: CLIPConfig
    
    def setup(self):
        c = self.config
        self.blocks = [
            Block(
                c.n_embd, c.n_head, c.dropout, c.use_bias, c.dtype
            ) for _ in range(c.n_layer)]

    def __call__(self, x, train):
        for block in self.blocks:
            x = block(x, train)
        return x


class ViT(nn.Module):
    config: CLIPConfig

    def setup(self):
        c = self.config
        self.patch_embd = PatchEmbed(c.patch_size, c.n_embd)
        self.pos_embd = nn.Embed(
            num_embeddings=c.n_patch,
            features=c.n_embd,
            dtype=c.dtype,
            name="pos_embed")
        self.drop = nn.Dropout(c.dropout)
        self.transformer = Transformer(c)
        self.ln_f = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.linear_head = nn.Dense(c.n_classes, use_bias=c.use_bias, dtype=c.dtype)

    def __call__(self, x, train=None):
        x = self.patch_embd(x)
        B, T, D = x.shape
        x = x + self.pos_embd(jnp.arange(T))
        x = self.drop(x, deterministic=not train)
        x = self.transformer(x, train)
        x = jnp.mean(x, axis=1) # pooled output
        x = self.ln_f(x)
        return x


class GPT(nn.Module):
    config: CLIPConfig

    def setup(self):
        c = self.config
        self.drop = nn.Dropout(c.dropout)
        self.transformer = Transformer(c)
        self.wte = nn.Embed(c.vocab_size, c.n_embd, dtype=c.dtype, name="wte")
        self.wpe = nn.Embed(c.block_size, c.n_embd, dtype=c.dtype, name="wpe")
        self.ln_f = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)

    def __call__(self, idx, train):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot apply sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t)[None]
        tok_emb = self.wte(idx) # (b, t, n_embd)
        pos_emb = self.wpe(pos) # (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)
        x = self.transformer(x, train)
        x = self.ln_f(x) 
        x = x[jnp.arange(x.shape[0]), idx.argmax(-1)]
        return x


def custom_logit_scale_init() -> Initializer:

    def init(key, shape, dtype):
        embd = jnp.ones(shape, dtype=dtype) * jnp.log(1 / 0.07)
        assert embd.shape == shape, "Expected shape {shape} but got {pos_embd.shape}"
        return embd

    return init

class CLIP(nn.Module):
    """
    Contrastive Language-Image Pretraining

    Goal is to minimize the cosine similiarity of real image:text pairs
    and maximize the cosine similiarity of non-real image:text pairs

    """
    config: CLIPConfig

    def setup(self) -> None:
        c = self.config
        self.image_encoder = ViT(c)
        self.text_encoder = GPT(c)
        self.image_proj = nn.Dense(c.n_embd_proj)
        self.text_proj = nn.Dense(c.n_embd_proj)
        self.logit_scale = nn.Embed(1, 1, embedding_init=custom_logit_scale_init())

    def __call__(self, img, text, train=None):
        image_features = self.image_encoder(img, train)
        image_features = self.image_proj(image_features)
        text_features = self.text_encoder(text, train)
        text_features = self.text_proj(text)

        image_features = image_features / jnp.linalg.norm(image_features, axis=1, keepdims=True)
        text_features = text_features / jnp.linalg.norm(text_features, axis=1, keepdims=True)

        logit_scale = jnp.exp(self.logit_scale(jnp.arange(1, dtype=jnp.int32)))
        logits = (image_features @ text_features.T) * logit_scale
        return logits