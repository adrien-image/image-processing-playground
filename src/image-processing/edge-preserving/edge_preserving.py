"""
Edge-preserving smoothing in PyTorch: the bilateral filter and non-local means of
edgePreserving.ts.

Bilateral: `unfold` gathers every pixel's (2r+1)² neighbourhood as a column, the
weights are a spatial Gaussian times a Gaussian of the value difference to the centre.
Non-local means loops over the offsets of the search window instead of over pixels:
for one offset, the patch distance of every pixel at once is a box filter (avg_pool2d)
of the squared difference between the image and its shifted copy. Images are float
tensors (H, W), values 0–255, replicate padding.
"""
import math
import torch
import torch.nn.functional as F


def bilateral_filter(image, radius, sigma_space, sigma_range):
    k = 2 * radius + 1
    h, w = image.shape
    padded = F.pad(image[None, None], (radius,) * 4, mode="replicate")
    windows = F.unfold(padded, k)[0]                              # (k², H·W)
    d = torch.arange(-radius, radius + 1, dtype=image.dtype)
    spatial = torch.exp(-(d[:, None] ** 2 + d[None, :] ** 2) / (2 * sigma_space ** 2)).reshape(-1, 1)
    centre = image.reshape(1, -1)
    weights = spatial * torch.exp(-(windows - centre) ** 2 / (2 * sigma_range ** 2))
    return ((weights * windows).sum(0) / weights.sum(0)).reshape(h, w)


def non_local_means(image, patch_radius=1, search_radius=6, sigma=20.0, h=16.0):
    r, s = patch_radius, search_radius
    H, W = image.shape
    pad = r + s
    padded = F.pad(image[None, None], (pad,) * 4, mode="replicate")[0, 0]
    centre = padded[s:s + H + 2 * r, s:s + W + 2 * r]             # the image with an r-pixel margin
    num = torch.zeros(H, W, dtype=image.dtype)
    den = torch.zeros(H, W, dtype=image.dtype)
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            shifted = padded[s + dy:s + dy + H + 2 * r, s + dx:s + dx + W + 2 * r]
            d2 = F.avg_pool2d(((centre - shifted) ** 2)[None, None], 2 * r + 1, stride=1)[0, 0]
            weight = torch.exp(-(d2 - 2 * sigma ** 2).clamp(min=0) / h ** 2)
            num += weight * shifted[r:r + H, r:r + W]
            den += weight
    return num / den


def gaussian_filter(image, radius):
    sigma = 0.3 * (radius - 1) + 0.8
    x = torch.arange(-radius, radius + 1, dtype=image.dtype)
    kernel = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return F.conv2d(F.pad(image[None, None], (radius,) * 4, mode="replicate"), kernel[None, None])[0, 0]


def psnr(reference, test):
    mse = ((reference - test) ** 2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def _self_check():
    n = 48
    y, x = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    clean = torch.where(x < n // 2, 60.0, 190.0).double() + 30.0 * ((x > 30) & ((y // 4) % 2 == 1))
    g = torch.Generator().manual_seed(3)
    sigma = 20.0
    noisy = (clean + sigma * torch.randn(n, n, generator=g, dtype=torch.float64)).clamp(0, 255)

    # The bilateral filter keeps the edge where a Gaussian blurs it…
    assert abs(bilateral_filter(clean, 3, 2, 20)[20, n // 2 - 1] - 60) < 1
    assert abs(gaussian_filter(clean, 3)[20, n // 2 - 1] - 60) > 20
    # …and becomes a Gaussian when σr is huge.
    assert torch.allclose(bilateral_filter(noisy, 3, 1.4, 1e6), gaussian_filter(noisy, 3), atol=1e-6)
    # Non-local means removes the most noise and keeps flat images flat.
    nlm = psnr(clean, non_local_means(noisy, 1, 5, sigma, 0.8 * sigma))
    assert nlm > psnr(clean, gaussian_filter(noisy, 2)) + 1
    assert nlm > psnr(clean, bilateral_filter(noisy, 3, 2, 2 * sigma))
    flat = torch.full((n, n), 100.0, dtype=torch.float64)
    assert torch.allclose(non_local_means(flat, 1, 5, sigma, 16), flat)
    print(f"edge_preserving.py: ok (PSNR: noisy {psnr(clean, noisy):.1f} dB, NLM {nlm:.1f} dB)")


if __name__ == "__main__":
    _self_check()
