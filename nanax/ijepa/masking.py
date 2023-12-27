import jax
import jax.numpy as jnp

# a block is a set of patches

@jax.jit
def sample_mask(rng):
    
    im_size: int = 28
    patch_size: int = 4
    n_pred_masks = 1 #4
    batch_size: int = 32
    pred_mask_scale = jnp.array([0.15, 0.2])
    pred_mask_aspect_ratio = jnp.array([0.75, 1.5])
    n_enc_masks = 1
    enc_mask_scale = jnp.array([0.85, 1.0])
    enc_mask_aspect_ratio = jnp.array([1.0, 1.0])
    
    height = im_size // patch_size
    width = im_size // patch_size
    
    def sample_block_size(scale, aspect_ratio, rng):
        _rand = jax.random.uniform(rng, (1,))
        # sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = jnp.rint(height * width * mask_scale)
        # sample block aspect ratio
        min_ar, max_ar = aspect_ratio
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        h = jnp.rint(jnp.sqrt(max_keep * aspect_ratio))
        w = jnp.rint(jnp.sqrt(max_keep / aspect_ratio))
        h = jnp.clip(h, a_max=height - 1)
        w = jnp.clip(w, a_max=width - 1)
        return jnp.concatenate([h, w], dtype=jnp.int32)

    def sample_block_predictor_mask(size, rng):
        
        h, w = size
        # sample top left corner of block
        top = jax.random.randint(rng, (), 0, height - h)
        left = jax.random.randint(rng, (), 0, width - w)
        # apply mask to block
        mask = jnp.zeros((height, width), dtype=jnp.int32)
        # from top to top + h, from left to left + w
        row_idxs = top + jnp.arange(h)
        col_idxs = left + jnp.arange(w)
        mask = mask.at[row_idxs[:, None], col_idxs].set(1)
        inverse_mask = 1 - mask
        mask = jnp.where(mask.flatten() > 0)[0] # indices of mask
        return mask, inverse_mask

    def sample_block_encoder_mask(size, acceptable_regions, rng):
        
        def constrain_mask(mask, acceptable_regions):
            for k in range(len(acceptable_regions)):
                mask = mask * acceptable_regions[k]
            return mask
        
        h, w = size
        # sample top left corner of block
        top = jax.random.randint(rng, (), 0, height - h)
        left = jax.random.randint(rng, (), 0, width - w)
        # apply mask to block
        mask = jnp.zeros((height, width), dtype=jnp.int32)
        # from top to top + h, from left to left + w
        row_idxs = top + jnp.arange(h)
        col_idxs = left + jnp.arange(w)
        mask = mask.at[row_idxs[:, None], col_idxs].set(1)
        mask = constrain_mask(mask, acceptable_regions)
        mask = jnp.where(mask.flatten() > 0)[0] # indices of mask
        return mask
    
    # get block size for predictor and encoder
    rng, e_rng, p_rng = jax.random.split(rng, 3)
    # e_size = sample_block_size(enc_mask_scale, enc_mask_aspect_ratio, e_rng)
    # p_size = sample_block_size(pred_mask_scale, pred_mask_aspect_ratio, p_rng)
    e_size = (3, 3)
    p_size = (6, 6)
    
    # sample predictor mask
    rng, p_rng = jax.random.split(rng, 2)
    mask_p, inv_mask_p = sample_block_predictor_mask(p_size, p_rng)
    mask_e = sample_block_encoder_mask(e_size, inv_mask_p[None, :], e_rng)

rng = jax.random.PRNGKey(0)
sample_mask(rng)
    