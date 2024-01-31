import flax.linen as nn

def parse_activation_fn(fn: str):
    if fn == 'relu':
        return nn.relu
    elif fn == 'gelu':
        return nn.gelu
    elif fn == 'elu':
        return nn.elu
    elif fn == 'leaky_relu':
        return nn.leaky_relu
    elif fn == 'silu' or 'swish':
        return nn.silu
    else:
        raise NotImplementedError