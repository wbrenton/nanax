import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import tensorflow as tf
import tensorflow_datasets as tfds
import tyro
from dataclasses import dataclass, field
import os
import jax.numpy as jnp
from typing import List
import tensorflow.experimental.numpy as np

import numpy as np
np.random.seed(0)  # Seed numpy's random number generator

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


def make_sample_masks(args):
    
    def sample_masks():

        def sample_block_size(scale, aspect_ratio):
            _rand = np.random.uniform(0, 1)
            # sample block scale
            min_s, max_s = scale
            mask_scale = min_s + _rand * (max_s - min_s)
            max_keep = np.rint(height * width * mask_scale)
            # sample block aspect ratio
            min_ar, max_ar = aspect_ratio
            aspect_ratio = min_ar + _rand * (max_ar - min_ar)
            h = np.rint(np.sqrt(max_keep * aspect_ratio))
            w = np.rint(np.sqrt(max_keep / aspect_ratio))
            h = np.clip(h, a_min=0, a_max=height - 1)
            w = np.clip(w, a_min=0, a_max=width - 1)
            return np.array([h, w], dtype=np.int32)

        def sample_block_mask(b_size, acceptable_regions=None):

            def constrain_mask(mask, acceptable_regions):
                for k in range(len(acceptable_regions)):
                    mask = mask * acceptable_regions[k]
                return mask
            
            # sample top left corner of block
            h, w = b_size
            top = np.random.randint(0, height - h)
            left = np.random.randint(0, width - w)

            # apply mask to block
            mask = np.zeros((height, width), dtype=np.int32)
            mask[top:top+h, left:left+w] = 1
            
            # constrain mask to acceptable regions
            if acceptable_regions is not None:
                mask = constrain_mask(mask, acceptable_regions)
            
            mask = np.nonzero(mask.flatten())[0]  # indices of mask
            mask = mask.squeeze()
            
            # create mask complement
            mask_complement = np.ones((height, width), dtype=np.int32)
            mask_complement[top:top + h, left:left + w] = 0

            return mask, mask_complement

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
        p_size = sample_block_size(pred_mask_scale, pred_mask_aspect_ratio)
        e_size = sample_block_size(enc_mask_scale, enc_mask_aspect_ratio)

        masks_pred = []
        masks_enc = []
        for _ in range(batch_size):
            # sample prediction masks
            masks_p, masks_C = [], []
            for _ in range(n_pred_masks):
                mask_p, mask_C = sample_block_mask(p_size)
                masks_p.append(mask_p)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask_p))
            masks_pred.append(masks_p)

            acceptable_regions = masks_C

            # sample encoder masks
            masks_e = []
            for _ in range(n_enc_masks):
                mask_e, _ = sample_block_mask(e_size, acceptable_regions)
                masks_e.append(mask_e)
                min_keep_enc = min(min_keep_enc, len(mask_e))
            masks_enc.append(masks_e)

        # truncate masks to min_keep
        # masks_pred = [[m[:min_keep_pred] for m in m_list] for m_list in masks_pred]
        # masks_enc = [[m[:min_keep_enc] for m in m_list] for m_list in masks_enc]
        masks_pred = tf.ragged.constant(masks_pred)
        masks_enc = tf.ragged.constant(masks_enc)

        return masks_pred, masks_enc

    return sample_masks

def make_preprocess(args):
    
    def preprocess(data):
        image, label = data['image'], data['label']
        sample_masks_fn = make_sample_masks(args)
        mask_pred, mask_enc = sample_masks_fn()

        return {'image': image, 'label': label, 'mask_enc': mask_enc, 'mask_pred': mask_pred}
    
    return preprocess

def prepare_dataset(args, split, batch_size=32):
    dataset = tfds.load('mnist', split=split, as_supervised=False)
    preprocess = make_preprocess(args)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_datasets(args):
    train_ds = prepare_dataset(args, 'train')
    test_ds = prepare_dataset(args, 'test')
    return train_ds, test_ds


if __name__ == "__main__":
    args = Args()
    # sample_masks = make_sample_masks(args)
    # sample_masks()
    
    test_ds = prepare_dataset(args, 'test')
    train_ds = prepare_dataset(args, 'train')

    batch = next(iter(train_ds))
    print(batch['image'].shape)