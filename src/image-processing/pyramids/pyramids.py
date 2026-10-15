"""
Resampling, Gaussian and Laplacian pyramids, and multiband blending in PyTorch —
the algorithms of resample.ts and pyramid.ts.

  resize        F.interpolate (nearest / bilinear / bicubic), optionally with antialias=True,
                which blurs before shrinking so fine detail can't alias into false patterns
  reduce        blur with the binomial filter [1 4 6 4 1]/16 (rows, then columns), keep every 2nd pixel
  expand        upsample back: even outputs (s[i−1] + 6s[i] + s[i+1]) / 8, odd outputs (s[i] + s[i+1]) / 2
  laplacian     Lᵢ = Gᵢ − expand(Gᵢ₊₁): one band of detail per level; the bands add back up to the image
  blend         combine two images band by band with a mask blurred to each band's scale
                (Burt & Adelson, 1983): wide, smooth seams for coarse bands, sharp ones for fine detail
Images are float tensors (C, H, W).
"""
import torch
import torch.nn.functional as F

BINOMIAL_5 = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float64) / 16


def resize(image: torch.Tensor, height: int, width: int, method: str = "bilinear", antialias: bool = False) -> torch.Tensor:
    kwargs = {} if method == "nearest" else {"align_corners": False, "antialias": antialias}
    return F.interpolate(image[None].double(), size=(height, width), mode=method, **kwargs)[0]


def reduce(image: torch.Tensor) -> torch.Tensor:
    """One level down: binomial blur (replicate borders), then keep pixels 0, 2, 4, …"""
    x = F.pad(image[None].double(), (2, 2, 2, 2), mode="replicate")
    c = image.shape[0]
    x = F.conv2d(x, BINOMIAL_5.view(1, 1, 1, 5).expand(c, 1, 1, 5), groups=c)
    x = F.conv2d(x, BINOMIAL_5.view(1, 1, 5, 1).expand(c, 1, 5, 1), groups=c)
    return x[0, :, ::2, ::2]


def _expand_last_axis(x: torch.Tensor, out_len: int) -> torch.Tensor:
    """Upsample the last axis to out_len (≤ twice its length), replicating the borders."""
    s = F.pad(x, (1, 1), mode="replicate")                 # s[j + 1] = x[j]
    i = torch.arange(out_len) // 2 + 1                     # position of x[o // 2] in s
    even = (s[..., i - 1] + 6 * s[..., i] + s[..., i + 1]) / 8
    odd = (s[..., i] + s[..., i + 1]) / 2
    return torch.where(torch.arange(out_len) % 2 == 0, even, odd)


def expand(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """One level up, to (height, width) — normally the size of the finer level."""
    rows = _expand_last_axis(image.double(), width)                                 # (C, h, width)
    return _expand_last_axis(rows.transpose(1, 2), height).transpose(1, 2)          # (C, height, width)


def gaussian_pyramid(image: torch.Tensor, levels: int) -> list[torch.Tensor]:
    pyramid = [image.double()]
    for _ in range(levels - 1):
        pyramid.append(reduce(pyramid[-1]))
    return pyramid


def laplacian_pyramid(image: torch.Tensor, levels: int) -> list[torch.Tensor]:
    g = gaussian_pyramid(image, levels)
    bands = [g[i] - expand(g[i + 1], *g[i].shape[1:]) for i in range(levels - 1)]
    return bands + [g[-1]]


def reconstruct(bands: list[torch.Tensor]) -> torch.Tensor:
    image = bands[-1]
    for band in reversed(bands[:-1]):
        image = band + expand(image, *band.shape[1:])
    return image


def blend(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, levels: int = 5) -> torch.Tensor:
    """Multiband blend: mask = 1 takes `a`, 0 takes `b`. mask is (1, H, W) in [0, 1]."""
    la, lb = laplacian_pyramid(a, levels), laplacian_pyramid(b, levels)
    gm = gaussian_pyramid(mask, levels)
    return reconstruct([m * x + (1 - m) * y for x, y, m in zip(la, lb, gm)])


def _self_check():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 45, 61, generator=g, dtype=torch.float64) * 255   # odd sizes on purpose
    bands = laplacian_pyramid(img, 4)
    assert [tuple(b.shape[1:]) for b in bands] == [(45, 61), (23, 31), (12, 16), (6, 8)]
    assert torch.allclose(reconstruct(bands), img, atol=1e-9)            # exact reconstruction
    flat = torch.full((1, 16, 16), 7., dtype=torch.float64)
    assert torch.allclose(expand(reduce(flat), 16, 16), flat)             # constants stay constant
    # Blending with an all-ones mask gives back the first image.
    assert torch.allclose(blend(img, torch.zeros_like(img), torch.ones(1, 45, 61, dtype=torch.float64)), img, atol=1e-9)
    assert resize(img, 20, 30, "bicubic", antialias=True).shape == (3, 20, 30)
    print("pyramids.py: ok")


if __name__ == "__main__":
    _self_check()
