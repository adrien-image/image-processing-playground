"""
Scale-invariant keypoints in PyTorch: the same DoG detector as keypoints.ts.

The Gaussian levels are separable convolutions; the DoG layers of an octave are stacked
into one (layers, H, W) volume, so "larger than all 26 neighbours" is a single
3-D max-pool compared with the volume itself. The edge test and the parabola refinement
are tensor expressions over the candidate points. Images are float tensors (H, W), 0–255.
"""
import math
import torch
import torch.nn.functional as F

INPUT_SIGMA = 0.5


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, math.ceil(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=image.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    img = F.pad(image[None, None], (radius, radius, 0, 0), mode="replicate")
    img = F.conv2d(img, k.view(1, 1, 1, -1))
    img = F.pad(img, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(img, k.view(1, 1, -1, 1))[0, 0]


def build_scale_space(gray: torch.Tensor, octaves=4, s=3, sigma0=1.6):
    """List of (gaussians (s+3, H, W), dogs (s+2, H, W), sigmas in input pixels, step)."""
    k = 2 ** (1 / s)
    base = gaussian_blur(gray / 255, math.sqrt(sigma0 ** 2 - INPUT_SIGMA ** 2))
    result = []
    for o in range(octaves):
        if min(base.shape) < 8:
            break
        local = [sigma0 * k ** i for i in range(s + 3)]
        levels = [base]
        for i in range(1, s + 3):
            levels.append(gaussian_blur(levels[-1], math.sqrt(local[i] ** 2 - local[i - 1] ** 2)))
        gaussians = torch.stack(levels)
        step = 2 ** o
        result.append((gaussians, gaussians[1:] - gaussians[:-1], [v * step for v in local], step))
        base = gaussians[s, ::2, ::2]   # σ = 2·σ0: the next octave's first level
    return result


def parabola_peak(minus, centre, plus):
    d = plus + minus - 2 * centre
    return torch.where(d == 0, torch.zeros_like(d), (-0.5 * (plus - minus) / d)).clamp(-0.5, 0.5)


def detect_keypoints(space, s=3, contrast=0.02, edge_ratio=10.0):
    """Rows of (x, y, sigma, response), strongest first."""
    found = []
    for gaussians, dogs, sigmas, step in space:
        vol = dogs[None, None]
        is_max = F.max_pool3d(vol, 3, stride=1, padding=1)[0, 0] == dogs
        is_min = -F.max_pool3d(-vol, 3, stride=1, padding=1)[0, 0] == dogs
        mask = (is_max | is_min) & (dogs.abs() >= contrast)
        mask[[0, -1]] = False                       # need a layer above and below
        mask[:, [0, -1]] = False
        mask[:, :, [0, -1]] = False
        i, y, x = mask.nonzero(as_tuple=True)
        d = lambda di, dy, dx: dogs[i + di, y + dy, x + dx]
        v = d(0, 0, 0)
        dxx, dyy = d(0, 0, 1) + d(0, 0, -1) - 2 * v, d(0, 1, 0) + d(0, -1, 0) - 2 * v
        dxy = (d(0, 1, 1) - d(0, 1, -1) - d(0, -1, 1) + d(0, -1, -1)) / 4
        det = dxx * dyy - dxy ** 2
        keep = (det > 0) & ((dxx + dyy) ** 2 < det * (edge_ratio + 1) ** 2 / edge_ratio)
        ox = parabola_peak(d(0, 0, -1), v, d(0, 0, 1))
        oy = parabola_peak(d(0, -1, 0), v, d(0, 1, 0))
        os = parabola_peak(d(-1, 0, 0), v, d(1, 0, 0))
        sigma = sigmas[0] * 2 ** ((i + os) / s)
        rows = torch.stack([(x + ox) * step, (y + oy) * step, sigma, v], dim=1)[keep]
        found.append(rows)
    rows = torch.cat(found) if found else torch.zeros(0, 4)
    return rows[rows[:, 3].abs().argsort(descending=True)]


def _blob(size, t, cx, cy):
    y, x = torch.meshgrid(torch.arange(size, dtype=torch.float64), torch.arange(size, dtype=torch.float64), indexing="ij")
    return 20 + 200 * torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * t ** 2))


def _self_check():
    small = detect_keypoints(build_scale_space(_blob(128, 4, 60, 70)))[0]
    large = detect_keypoints(build_scale_space(_blob(128, 8, 60, 70)))[0]
    # Found at the blob's centre, light blob → negative DoG, σ close to the blob's size…
    assert torch.hypot(small[0] - 60, small[1] - 70) < 1.5 and small[3] < 0
    assert 0.7 < small[2] / 4 < 1.4
    # …and twice the scale for a blob twice as large.
    assert 1.7 < large[2] / small[2] < 2.3
    # A straight edge and a flat image give no keypoints.
    edge = torch.where(torch.arange(64)[None, :] < 32, 30.0, 220.0).expand(64, 64).double()
    assert len(detect_keypoints(build_scale_space(edge))) == 0
    assert len(detect_keypoints(build_scale_space(torch.full((64, 64), 90.0, dtype=torch.float64)))) == 0
    print(f"keypoints.py: ok (σ {small[2]:.2f} for a blob of size 4, {large[2]:.2f} for size 8)")


if __name__ == "__main__":
    _self_check()
