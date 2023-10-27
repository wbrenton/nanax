import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from typing import Optional
from dataclasses import dataclass


class CausalSelfAttention(nn.Module):
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
        
        causal_mask = jnp.tril(jnp.ones((T, T))).reshape(1, 1, T, T)
        
        attn = q @ k.swapaxes(-2, -1) / jnp.sqrt(k.shape[-1]) # (B, nh, T, hd) @ (B, nh, hd, T) -> (B, nh, T, T)
        attn = jnp.where(causal_mask[:, :, :T, :T], attn, -jnp.inf)
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
    config: dataclass

    def setup(self):
        c = self.config
        self.ln_1 = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.attn = CausalSelfAttention(c.n_embd, c.n_head, c.use_bias, c.dropout, dtype=c.dtype)
        self.ln_2 = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)
        self.mlp = MLP(c.dropout, c.use_bias, c.dtype)

    def __call__(self, x, train):
        x = x + self.attn(self.ln_1(x), train)
        x = x + self.mlp(self.ln_2(x), train)
        return x


class GPT(nn.Module):
    config: dataclass

    def setup(self):
        c = self.config
        self.wte = nn.Embed(c.vocab_size, c.n_embd, dtype=c.dtype, name="wte")
        self.wpe = nn.Embed(c.block_size, c.n_embd, dtype=c.dtype, name="wpe")
        self.drop = nn.Dropout(c.dropout)
        self.h = [Block(c) for _ in range(c.n_layer)]
        self.ln_f = nn.LayerNorm(epsilon=1e-6, use_bias=c.use_bias, dtype=c.dtype)

    def __call__(self, idx, train=True):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot apply sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(0, t)[None]
        tok_emb = self.wte(idx) # (b, t, n_embd)
        pos_emb = self.wpe(pos) # (1, t, n_embd)
        x = self.drop(tok_emb + pos_emb, deterministic=not train)
        for block in self.h:
            x = block(x, train)
        x = self.ln_f(x) 
        logits = self.wte.attend(x) # (b, t, n_embed) @ (1, vocab_size, n_embd).T -> (b, t, vocab_size)
        return logits
    
    def generate(self, key, params, input_tokens, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        B, T = input_tokens.shape
        block_size = self.config.block_size
        padding = jnp.zeros((B, max_new_tokens), dtype=jnp.int32)
        tokens = jnp.concatenate([input_tokens, padding], axis=-1)
        indexes = jnp.arange(T, T + max_new_tokens)
        start_indexes = (indexes - block_size).clip(min=0)

        # tokens index -> tokens None
        def scan_f(tokens, item):
            (i, start_i) = item
            # l: x y
            # t: a b - -
            # i: 0 1 2 3
            step_key = jax.random.fold_in(key, i)
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            token_slice = jax.lax.dynamic_slice(tokens, (0, start_i), (B, block_size))
            logits = self.apply(params, token_slice, train=False)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, i - 1, :] / temperature
            # optionally crop the logits to only the top k options
            # sample from the distribution
            if top_k is not None:
                top_logits, top_tokens = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                token_idx = jax.random.categorical(step_key, top_logits, axis=-1)
                next_token = jnp.take_along_axis(top_tokens, token_idx[:, None], axis=-1).squeeze(-1)
            else:
                next_token = jax.random.categorical(step_key, logits, axis=-1)
            # append sampled index to the running sequence and continue
            tokens = tokens.at[:, i].set(next_token)

            return tokens, None
        
        tokens, _ = jax.lax.scan(scan_f, tokens, (indexes, start_indexes))

        return tokens
