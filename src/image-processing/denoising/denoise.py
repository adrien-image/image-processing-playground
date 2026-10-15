"""
Noise and denoising in PyTorch — the same filters as denoise.ts.

The median filter gathers every (2r+1)² neighbourhood with `unfold` (one column per
pixel) and takes the median of each column; the Gaussian filter is a convolution.
Both use replicate padding. Images are float tensors (H, W), values 0–255.
"""
import math
import torch
import torch.nn.functional as F


def add_salt_pepper(image: torch.Tensor, density: float, generator=None) -> torch.Tensor:
    r = torch.rand(image.shape, generator=generator)
    out = image.clone()
    out[r < density / 2] = 0
    out[(r >= density / 2) & (r < density)] = 255
    return out


def add_gaussian_noise(image: torch.Tensor, sigma: float, generator=None) -> torch.Tensor:
    return (image + sigma * torch.randn(image.shape, generator=generator)).clamp(0, 255)


def median_filter(image: torch.Tensor, radius: int) -> torch.Tensor:
    k = 2 * radius + 1
    padded = F.pad(image[None, None], (radius,) * 4, mode="replicate")
    patches = F.unfold(padded, k)                       # (1, k², H·W)
    # The window size k² is odd, so the median is the middle value (no averaging).
    return patches.median(dim=1).values.reshape(image.shape)


def gaussian_sigma_for_radius(radius: int) -> float:
    """σ that OpenCV picks for a k×k Gaussian (k = 2r+1)."""
    return 0.3 * (radius - 1) + 0.8


def gaussian_filter(image: torch.Tensor, radius: int) -> torch.Tensor:
    sigma = gaussian_sigma_for_radius(radius)
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    padded = F.pad(image[None, None], (radius,) * 4, mode="replicate")
    return F.conv2d(padded, kernel[None, None])[0, 0]


def psnr(reference: torch.Tensor, test: torch.Tensor) -> float:
    """Peak signal-to-noise ratio in dB (higher = closer to the reference)."""
    mse = ((reference - test) ** 2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def _self_check():
    g = torch.Generator().manual_seed(0)
    clean = torch.zeros(64, 64)
    clean[16:48, 16:48] = 200
    sp = add_salt_pepper(clean, 0.1, g)
    # The median removes salt & pepper almost entirely and beats the Gaussian on it…
    assert psnr(clean, median_filter(sp, 1)) > psnr(clean, gaussian_filter(sp, 1)) + 5
    # …while on Gaussian noise the Gaussian filter is at least as good.
    gn = add_gaussian_noise(clean, 25, g)
    assert psnr(clean, gaussian_filter(gn, 1)) > psnr(clean, gn)
    print("denoise.py: ok")


if __name__ == "__main__":
    _self_check()
