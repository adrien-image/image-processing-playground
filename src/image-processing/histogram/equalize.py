"""
Histogram equalisation in PyTorch — the same mapping as equalize.ts.

Count the gray levels (bincount), accumulate them into the CDF (cumsum), and map every
level v to 255·(cdf[v] − cdf_min) / (N − cdf_min), so the lowest occupied level becomes 0,
the highest 255 and the levels in between spread out by how many pixels they hold.
"""
import torch


def equalize(image: torch.Tensor):
    """Returns (equalised image, mapping (256,)) for a float image with values 0–255."""
    levels = image.round().clamp(0, 255).long()
    hist = torch.bincount(levels.flatten(), minlength=256).double()
    cdf = torch.cumsum(hist, 0)
    cdf_min = cdf[cdf > 0][0]
    denom = levels.numel() - cdf_min
    if denom == 0:                    # a single-level image has nothing to stretch
        mapping = torch.arange(256, dtype=torch.float64)
    else:
        mapping = ((cdf - cdf_min) / denom * 255).clamp(min=0)
    return mapping[levels], mapping


def _self_check():
    torch.manual_seed(0)
    dull = 100 + torch.rand(64, 64) * 40          # low contrast: levels 100–140
    out, mapping = equalize(dull)
    assert out.min() == 0 and out.max() == 255
    assert (mapping[1:] >= mapping[:-1]).all()     # monotonic: the order of gray levels is kept
    flat = torch.full((8, 8), 77.)
    assert torch.equal(equalize(flat)[0], flat.double())
    print("equalize.py: ok")


if __name__ == "__main__":
    _self_check()
