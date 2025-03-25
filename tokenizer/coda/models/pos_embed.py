from typing import Optional, List, Tuple

import torch

def ndgrid(*tensors) -> Tuple[torch.Tensor, ...]:
    """generate N-D grid in dimension order.

    The ndgrid function is like meshgrid except that the order of the first two input arguments are switched.

    That is, the statement
    [X1,X2,X3] = ndgrid(x1,x2,x3)

    produces the same result as

    [X2,X1,X3] = meshgrid(x2,x1,x3)

    This naming is based on MATLAB, the purpose is to avoid confusion due to torch's change to make
    torch.meshgrid behaviour move from matching ndgrid ('ij') indexing to numpy meshgrid defaults of ('xy').

    """
    try:
        return torch.meshgrid(*tensors, indexing='ij')
    except TypeError:
        # old PyTorch < 1.10 will follow this path as it does not have indexing arg,
        # the old behaviour of meshgrid was 'ij'
        return torch.meshgrid(*tensors)

def freq_bands(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=torch.int64, device=device).to(torch.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands


def build_sincos2d_pos_embed(
        feat_shape: List[int],
        dim: int = 64,
        temperature: float = 10000.,
        reverse_coord: bool = False,
        interleave_sin_cos: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
) -> torch.Tensor:
    """

    Args:
        feat_shape:
        dim:
        temperature:
        reverse_coord: stack grid order W, H instead of H, W
        interleave_sin_cos: sin, cos, sin, cos stack instead of sin, sin, cos, cos
        dtype:
        device:

    Returns:

    """
    assert dim % 4 == 0, 'Embed dimension must be divisible by 4 for sin-cos 2D position embedding'
    pos_dim = dim // 4
    bands = freq_bands(pos_dim, temperature=temperature, step=1, device=device)

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W
    grid = torch.stack(ndgrid([
        torch.arange(s, device=device, dtype=torch.int64).to(torch.float32)
        for s in feat_shape
    ])).flatten(1).transpose(0, 1)
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = 2 if interleave_sin_cos else 1  # stack sin, cos, sin, cos  instead of sin sin cos cos
    pos_emb = torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)
    return pos_emb.to(dtype=dtype)

