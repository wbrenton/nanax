from datasets import load_dataset

import jax
import optax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from model import GAN, GANState, Generator, Discriminator

def load_data(name: str, batch_size):
    ds = load_dataset(name)
    ds = ds.with_format("jax")
    ds = ds.shuffle()

    def create_iter(dataset):
        while True:
            for i, batch in enumerate(dataset.iter(batch_size=batch_size)):
                batch['count'] = i
                xb = batch['image']
                xb = jnp.expand_dims(xb, axis=-1) if len(xb) != 4 else xb
                yield xb

    train_iter = create_iter(ds['train'])

    return train_iter

batch_size = 64
num_latents = 20
num_steps = 20001
log_every = num_steps // 100

## TRAINING SCRIPT

# Let's see what hardware we're working with. The training takes a few
# minutes on a GPU, a bit longer on CPU.
print(f"Number of devices: {jax.device_count()}")
print("Device:", jax.devices()[0].device_kind)
print("")

# Make the dataset.
dataset = load_data("mnist", batch_size)
sample_batch = next(dataset)

key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)

# The model.
gan = GAN(
    num_latents=num_latents,
    generator = Generator(),
    discriminator = Discriminator(),
)

gan_state = GANState(
    g=TrainState.create(
        apply_fn=gan.generator.apply,
        params=gan.generator.init(init_key, jnp.ones((1, num_latents))),
        tx=optax.adam(learning_rate=1e-4),
    ),
    d=TrainState.create(
        apply_fn=gan.discriminator.apply,
        params=gan.discriminator.init(init_key, sample_batch),
        tx=optax.adam(learning_rate=1e-4),
    )
)

def train_step(gan_state, batch, key):
    key, g_key, d_key = jax.random.split(key, 3)
    g_loss, g_grads = jax.value_and_grad(gan.generator_loss)(
        gan_state.g.params,
        gan_state.d.params,
        batch.shape[0],
        g_key
    )
    d_loss, d_grads = jax.value_and_grad(gan.discriminator_loss)(
        gan_state.d.params,
        gan_state.g.params,
        batch,
        d_key
    )
    gan_state = gan_state.replace(
        g=gan_state.g.apply_gradients(grads=g_grads),
        d=gan_state.d.apply_gradients(grads=d_grads)
    )
    return gan_state, g_loss, d_loss, key

for step in range(num_steps):
    batch = next(dataset)
    gan_state, g_loss, d_loss, key = train_step(gan_state, batch, key)
    if step % 10 == 0:
        print(f"Step {step}: g_loss={g_loss:.4f} d_loss={d_loss:.4f}")