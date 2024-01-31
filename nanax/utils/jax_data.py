import jax
import numpy as np
from flax.jax_utils import prefetch_to_device

# taken from https://github.com/google-research/vision_transformer/blob/10ffdebb01aa40714b175a7c3be700c872efb2f4/vit_jax/input_pipeline.py#L243
def prefetch(dataset, num_prefetch):
    """Prefetches data to device and converts to numpy array."""
    ds_iter = iter(dataset)
    ds_iter = map(lambda x: jax.tree_map(lambda t: np.asarray(memoryview(t)), x),
                ds_iter)
    if num_prefetch:
        ds_iter = prefetch_to_device(ds_iter, num_prefetch)
    return ds_iter