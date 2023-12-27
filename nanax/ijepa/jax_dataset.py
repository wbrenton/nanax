import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import os
import tyro
import jax
import jax.numpy as jnp
import jax.numpy as np
from functools import partial
from dataclasses import dataclass, field
from typing import List

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
    n_pred_masks = 4
    pred_mask_scale = jnp.array([0.15, 0.2])
    pred_mask_aspect_ratio = jnp.array([0.75, 1.5])
    n_enc_masks = 1
    enc_mask_scale = jnp.array([0.85, 1.0])
    enc_mask_aspect_ratio = jnp.array([1.0, 1.0])

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


def sample_masks(args, rng):

    def sample_block_size(scale, aspect_ratio, rng):
        _rand = jax.random.uniform(rng, (1,))
        # sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = np.rint(height * width * mask_scale)
        # sample block aspect ratio
        min_ar, max_ar = aspect_ratio
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        h = np.rint(np.sqrt(max_keep * aspect_ratio))
        w = np.rint(np.sqrt(max_keep / aspect_ratio))
        h = np.clip(h, a_max=height - 1)
        w = np.clip(w, a_max=width - 1)
        return np.concatenate([h, w], dtype=np.int32)

    def sample_block_predictor_mask(b_size, rng):
        h, w = b_size
        # sample top left corner of block
        top = jax.random.randint(rng, (), 0, height - h)
        left = jax.random.randint(rng, (), 0, width - w)

        # apply mask to block
        mask = np.zeros((height, width), dtype=np.int32)
        mask = mask.at[top:top+h, left:left+w].set(1)
        mask = np.nonzero(mask.flatten())[0] # indices of mask
        mask = mask.squeeze()

        # create mask complement
        mask_complement = np.ones((height, width), dtype=np.int32)
        mask_complement = mask_complement.at[top:top + h, left:left + w].set(0)
        return mask, mask_complement

    def sample_block_encoder_mask(b_size, acceptable_regions, rng):

        def constrain_mask(mask, acceptable_regions):
            for k in range(len(acceptable_regions)):
                mask = mask * acceptable_regions[k]
            return mask
        
        h, w = b_size
        # sample top left corner of block
        top = jax.random.randint(rng, (), 0, height - h)
        left = jax.random.randint(rng, (), 0, width - w)

        # apply mask to block
        mask = np.zeros((height, width), dtype=np.int32)
        mask = mask.at[top:top+h, left:left+w].set(1)
        mask = constrain_mask(mask, acceptable_regions)
        mask = np.nonzero(mask.flatten())[0] # indices of mask
        mask = mask.squeeze()

        return mask

    im_size = args.crop_size
    patch_size = args.patch_size
    batch_size = args.batch_size
    n_pred_masks = args.n_pred_masks
    n_enc_masks = args.n_enc_masks
    pred_mask_scale = args.pred_mask_scale
    pred_mask_aspect_ratio = args.pred_mask_aspect_ratio
    enc_mask_scale = args.enc_mask_scale
    enc_mask_aspect_ratio = args.enc_mask_aspect_ratio
    
    height, width = im_size // patch_size, im_size // patch_size
    
    min_keep_pred = height * width
    min_keep_enc = height * width
    
    # sample block size for predictor and encoder
    rng, enc_rng, pred_rng = jax.random.split(rng, 3)
    p_size = sample_block_size(pred_mask_scale, pred_mask_aspect_ratio, pred_rng)
    e_size = sample_block_size(enc_mask_scale, enc_mask_aspect_ratio, enc_rng)

    masks_pred = []
    masks_enc = []
    for _ in range(batch_size): # change to vmap
        # sample prediction masks
        masks_p, masks_C = [], []
        for _ in range(n_pred_masks): # change to scan
            rng, step_rng = jax.random.split(rng)
            mask_p, mask_C = sample_block_predictor_mask(p_size, step_rng)
            masks_p.append(mask_p)
            masks_C.append(mask_C)
            min_keep_pred = min(min_keep_pred, len(mask_p))
        masks_pred.append(masks_p)

        acceptable_regions = masks_C

        # sample encoder masks
        masks_e = []
        for _ in range(n_enc_masks):
            rng, step_rng = jax.random.split(rng)
            mask_e = sample_block_encoder_mask(e_size, acceptable_regions, step_rng)
            masks_e.append(mask_e)
            min_keep_enc = min(min_keep_enc, len(mask_e))
        masks_enc.append(masks_e)

    # truncate masks to min_keep (prevents masking to much of the image)
    masks_pred = [[m[:min_keep_pred] for m in m_list] for m_list in masks_pred]
    masks_enc = [[m[:min_keep_enc] for m in m_list] for m_list in masks_enc]

    return masks_pred, masks_enc

class Batch:
    image: jax.Array
    masks: jax.Array
    masks_x: jax.Array
    rng: jax.Array


def sample_batches_w_masks(args, dataset, batch_size, rng):
    batches = sample_batches(dataset, batch_size, rng)
    maskss, maskss_x = [], []
    for _ in range(len(batches.image)):
        masks, masks_x = sample_masks(args, rng)
        maskss.append(masks)
        maskss_x.append(masks_x)
    maskss = np.array(maskss)
    maskss_x = np.array(maskss_x)
    return Batch(
        image=batches.image,
        masks=maskss,
        masks_x=maskss_x,
        rng=batches.rng,
    )

if __name__ == """__main__""":
    from nanax.ijepa.ijepa import Args

    args = Args()
    rng = jax.random.PRNGKey(42)
    batch_size = 8
    
    (train_ds, test_ds), _, _ = get_dataset("mnist", batch_size, rng)
    sample_batches_w_masks(args, train_ds, batch_size, rng)