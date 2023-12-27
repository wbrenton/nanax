import sys
#######################################
### Replace with your path to nanax ###
sys.path.append("/admin/home-willb/nanax/")
#######################################

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import tyro
from dataclasses import dataclass, field
import os
import jax.numpy as jnp
from typing import List

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

import numpy as np

import tensorflow as tf

def make_sample_masks(args):
    
    def _sample_masks():
        
        tf.random.set_seed(0)  # Seed TensorFlow's random number generator

        def sample_block_size(scale, aspect_ratio):
            _rand = tf.random.uniform([1], 0, 1)[0]
            # sample block scale
            min_s, max_s = tf.cast(scale[0], tf.float32), tf.cast(scale[1], tf.float32)
            mask_scale = min_s + _rand * (max_s - min_s)
            max_keep = tf.math.round(args.crop_size * args.crop_size * mask_scale)

            # sample block aspect ratio
            min_ar, max_ar = tf.cast(aspect_ratio[0], tf.float32), tf.cast(aspect_ratio[1], tf.float32)
            aspect_ratio = min_ar + _rand * (max_ar - min_ar)
            h = tf.cast(tf.math.round(tf.sqrt(max_keep * aspect_ratio)), tf.int32)
            w = tf.cast(tf.math.round(tf.sqrt(max_keep / aspect_ratio)), tf.int32)

            # Ensure h and w are within the valid range
            h = tf.clip_by_value(h, 0, args.crop_size // args.patch_size - 1)
            w = tf.clip_by_value(w, 0, args.crop_size // args.patch_size - 1)
            return h, w


        def sample_block_predictor_mask(b_size):
            h, w = b_size
            # sample top left corner of block
            top = tf.random.uniform([1], 0, args.crop_size // args.patch_size - h, dtype=tf.int32)[0]
            left = tf.random.uniform([1], 0, args.crop_size // args.patch_size - w, dtype=tf.int32)[0]

            # apply mask to block
            mask = tf.Variable(tf.zeros((args.crop_size // args.patch_size, args.crop_size // args.patch_size), dtype=tf.int32))
            mask[top:top+h, left:left+w].assign(tf.ones((h, w), dtype=tf.int32))
            mask = tf.reshape(mask, [-1])
            mask_indices = tf.where(mask == 1)[:, 0]

            # create mask complement
            mask_complement = tf.Variable(tf.ones((args.crop_size // args.patch_size, args.crop_size // args.patch_size), dtype=tf.int32))
            mask_complement[top:top + h, left:left + w].assign(tf.zeros((h, w), dtype=tf.int32))
            return mask_indices, mask_complement

        def sample_block_encoder_mask(b_size, acceptable_regions):
            h, w = b_size
            # sample top left corner of block
            top = tf.random.uniform([1], 0, args.crop_size // args.patch_size - h, dtype=tf.int32)[0]
            left = tf.random.uniform([1], 0, args.crop_size // args.patch_size - w, dtype=tf.int32)[0]

            # apply mask to block
            mask = tf.Variable(tf.zeros((args.crop_size // args.patch_size, args.crop_size // args.patch_size), dtype=tf.int32))
            mask[top:top+h, left:left+w].assign(tf.ones((h, w), dtype=tf.int32))

            for acceptable_region in acceptable_regions:
                mask = mask * acceptable_region

            mask = tf.reshape(mask, [-1])
            mask_indices = tf.where(mask == 1)[:, 0]
            return mask_indices

        # Initialize parameters from args
        height, width = args.crop_size // args.patch_size, args.crop_size // args.patch_size
        min_keep_pred = height * width
        min_keep_enc = height * width
        # Sample block size for predictor and encoder
        p_size = sample_block_size(args.pred_mask_scale, args.pred_mask_aspect_ratio)
        e_size = sample_block_size(args.enc_mask_scale, args.enc_mask_aspect_ratio)

        masks_pred = []
        masks_enc = []
        for _ in tf.range(args.batch_size):
            # Sample prediction masks
            masks_p, masks_C = [], []
            for _ in tf.range(args.n_pred_masks):
                mask_p, mask_C = sample_block_predictor_mask(p_size)
                masks_p.append(mask_p)
                masks_C.append(mask_C)
                min_keep_pred = tf.minimum(min_keep_pred, tf.shape(mask_p)[0])
            masks_pred.append(masks_p)

            acceptable_regions = masks_C

            # Sample encoder masks
            masks_e = []
            for _ in tf.range(args.n_enc_masks):
                mask_e = sample_block_encoder_mask(e_size, acceptable_regions)
                masks_e.append(mask_e)
                min_keep_enc = tf.minimum(min_keep_enc, tf.shape(mask_e)[0])
            masks_enc.append(masks_e)

        # Truncate masks to min_keep
        masks_pred = [[m[:min_keep_pred] for m in m_list] for m_list in masks_pred]
        masks_enc = [[m[:min_keep_enc] for m in m_list] for m_list in masks_enc]

        return masks_pred, masks_enc

    return _sample_masks


def make_preprocess(args):
    
    sample_masks_fn = make_sample_masks(args)

    def preprocess(data):
        image, label = data['image'], data['label']
        mask_pred, mask_enc = sample_masks_fn()
        return {'image': image, 'label': label, 'mask_enc': mask_enc, 'mask_pred': mask_pred}
    
    return preprocess

def prepare_dataset(args, split, batch_size=32):
    dataset = tfds.load('mnist', split=split, as_supervised=False)
    preprocess = make_preprocess(args)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    args = Args()
    # sample_fn = make_sample_masks(args)
    # sample_fn()
    test_ds = prepare_dataset(args, 'test')
    train_ds = prepare_dataset(args, 'train')

    batch = next(iter(train_ds))
    print(batch['image'].shape)