import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState

from dataclasses import dataclass

class Generator(nn.Module):
    output_channels = (32, 1)
    
    @nn.compact
    def __call__(self, x):
        """Maps noise to latent images"""
        x = nn.Linear(7 * 7 * 64)(x)
        x = jnp.reshape(x, x.shape[:1] + (7, 7, 64))
        for output_channels in self.output_channels:
            x = nn.relu(x)
            x = nn.ConvTranspose(
                output_channels,
                kernel_size=(5, 5),
                strides=(2, 2),
                padding="SAME")(x)
        return jnp.tanh(x)
    
class Discriminator(nn.Module):
    output_channels = (8, 16, 32, 64, 128)
    strides = (2, 1, 2, 1, 2)
    
    @nn.compact
    def __call__(self, x):
        """Binary classification that the image is real or fake"""
        for output_channel, stride in zip(self.output_channels, self.strides):
            x = nn.Conv(
                output_channel,
                kernel_size=(5, 5),
                strides=(stride, stride),
                padding="SAME")(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Flatten()(x)
        logits = nn.Dense(2)(x)

@struct.dataclass
class GANState:
    g: TrainState
    d: TrainState
  
  
class GAN(nn.Module):
    num_latents: int
    generator: Generator
    discriminator: Discriminator
    
    def sample(self, g_params, num_samples, key):
        latents = jax.random.normal(key, shape=(num_samples, self.num_latents))
        return self.generator(g_params, latents)

    def generator_loss(self, g_params, d_params, num_samples, key):
        x = self.sample(g_params, num_samples, key)
        logits = self.discriminator(d_params, x)
        probs = nn.softmax(logits)[:, 1]
        return -jnp.log(probs).mean()
    
    def discriminator_loss(self, d_params, g_params, x, key):
        # sample
        syn_x = self.sample(g_params, x.shape[0], key)
        
        # discriminate
        concat_x = jnp.concatenate([x, syn_x])
        concat_x = self.discriminator(d_params, concat_x)
        real_x, syn_x = jnp.split(concat_x, 2, axis=0)
        
        # loss
        real_label = jnp.ones((real_x.shape[0],), dtype=jnp.int32)
        real_loss = optax.softmax_cross_entropy_with_integer_labels(real_x, real_label)
        syn_label = jnp.zeros((syn_x.shape[0],), dtype=jnp.int32)
        syn_loss = optax.softmax_cross_entropy_with_integer_labels(syn_x, syn_label)
        
        return jnp.mean(real_loss + syn_loss)