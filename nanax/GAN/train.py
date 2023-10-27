import tf
import tfds

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from model import GAN, GANState, Generator, Discriminator

# Download the data once.
mnist = tfds.load("mnist")

def make_dataset(batch_size, seed=1):
  def _preprocess(sample):
    # Convert to floats in [0, 1].
    image = tf.image.convert_image_dtype(sample["image"], tf.float32)
    # Scale the data to [-1, 1] to stabilize training.
    return 2.0 * image - 1.0

  ds = mnist["train"]
  ds = ds.map(map_func=_preprocess,
              num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.cache()
  ds = ds.shuffle(10 * batch_size, seed=seed).repeat().batch(batch_size)
  return iter(tfds.as_numpy(ds))

num_steps = 20001
log_every = num_steps // 100

## TRAINING SCRIPT

# Let's see what hardware we're working with. The training takes a few
# minutes on a GPU, a bit longer on CPU.
print(f"Number of devices: {jax.device_count()}")
print("Device:", jax.devices()[0].device_kind)
print("")

# Make the dataset.
dataset = make_dataset(batch_size=64)

key = jax.random.PRNGKey(0)
key, init_key = jax.random.split(key)

# The model.
gan = GAN(
    num_latents=20,
    generator = Generator(),
    discriminator = Discriminator(),
)

gan_state = GANState(
    g=TrainState.create(
        apply_fn=gan.generator.apply,
        params=gan.generator.init(init_key, jnp.ones((1, 20))),
        tx=optax.adam(learning_rate=1e-4),
    ),
    d=TrainState.create(
        apply_fn=gan.discriminator.apply,
        params=gan.discriminator.init(init_key, jnp.ones((1, 28, 28, 1))),
        tx=optax.adam(learning_rate=1e-4),
    )
)

def train_step(gan_state, batch, key):
    key, g_key, d_key = jax.random.split(key, 3)
    g_loss, g_grads = jax.value_and_grad(gan.generator_loss)(
        gan_state.g.params, gan_state.d.params, batch.shape[0], g_key
    )
    d_loss, d_grads = jax.value_and_grad(gan.discriminator_loss)(
        gan_state.d.params, gan_state.g.params, batch, d_key
    )
    gan_state = gan_state.replace(
        g=gan_state.g.apply_gradients(grads=g_grads),
        d=gan_state.d.apply_gradients(grads=d_grads)
    )
    return gan_state, g_loss, d_loss, key

for step in range(num_steps):
    batch = next(dataset)
    gan_state, g_loss, d_loss, key = train_step(gan_state, batch, key)
    if step % log_every == 0:
        print(f"Step {step}: g_loss={g_loss:.4f} d_loss={d_loss:.4f}")