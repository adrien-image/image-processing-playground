"""
SLIC superpixels in PyTorch — slic.ts with tensors.

Pixels are 5-D points (L, a, b, x, y). Instead of looping over centres and their 2S windows,
each pixel computes its distance to the 9 centres of the grid cells around it (its own cell
and the 8 neighbours: exactly the centres whose 2S window can contain it), all pixels at once,
and takes the argmin. Updating the centres is a scatter-mean (index_add_).
"""
import torch


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """(3, H, W) sRGB in [0, 1] → (3, H, W) CIELAB (D65)."""
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], dtype=rgb.dtype)
    xyz = torch.einsum("ij,jhw->ihw", M, lin) / torch.tensor([0.95047, 1.0, 1.08883], dtype=rgb.dtype)[:, None, None]
    f = torch.where(xyz > 216 / 24389, xyz.clamp(min=1e-12) ** (1 / 3), (24389 / 27 * xyz + 16) / 116)
    return torch.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def slic(rgb: torch.Tensor, k: int = 200, compactness: float = 10.0, iterations: int = 10) -> torch.Tensor:
    """(3, H, W) image in [0, 1] → (H, W) superpixel labels (before the connectivity clean-up)."""
    _, h, w = rgb.shape
    lab = rgb_to_lab(rgb)
    step = (h * w / k) ** 0.5
    gy, gx = int(round(h / step)), int(round(w / step))       # grid of gy × gx centres
    ys, xs = torch.meshgrid(torch.arange(h, dtype=rgb.dtype), torch.arange(w, dtype=rgb.dtype), indexing="ij")
    features = torch.cat([lab, xs[None], ys[None]]).reshape(5, -1).T            # (N, 5)

    cy = (torch.arange(gy, dtype=rgb.dtype) + 0.5) * h / gy
    cx = (torch.arange(gx, dtype=rgb.dtype) + 0.5) * w / gx
    cyy, cxx = torch.meshgrid(cy, cx, indexing="ij")
    idx = (cyy.long() * w + cxx.long()).flatten()
    centres = features[idx].clone()                                              # (K, 5)

    # The 9 candidate centres of each pixel: its grid cell and the neighbouring cells.
    cell_y = (ys * gy / h).long().clamp(max=gy - 1).flatten()
    cell_x = (xs * gx / w).long().clamp(max=gx - 1).flatten()
    offsets = torch.tensor([(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    ny = (cell_y[:, None] + offsets[:, 0]).clamp(0, gy - 1)
    nx = (cell_x[:, None] + offsets[:, 1]).clamp(0, gx - 1)
    candidates = ny * gx + nx                                                    # (N, 9)

    scale = torch.tensor([1, 1, 1, compactness / step, compactness / step], dtype=rgb.dtype)
    for _ in range(iterations):
        d = ((features[:, None, :] - centres[candidates]) * scale).pow(2).sum(-1)  # (N, 9)
        labels = candidates.gather(1, d.argmin(1, keepdim=True))[:, 0]
        sums = torch.zeros_like(centres).index_add_(0, labels, features)
        counts = torch.zeros(len(centres), dtype=rgb.dtype).index_add_(0, labels, torch.ones_like(labels, dtype=rgb.dtype))
        centres = torch.where(counts[:, None] > 0, sums / counts.clamp(min=1)[:, None], centres)
    return labels.reshape(h, w)


def boundaries(labels: torch.Tensor) -> torch.Tensor:
    edge = torch.zeros_like(labels, dtype=torch.bool)
    edge[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    edge[:-1, :] |= labels[1:, :] != labels[:-1, :]
    return edge


def boundary_recall(truth: torch.Tensor, predicted: torch.Tensor, tolerance: int = 1) -> float:
    """Fraction of true edge pixels with a predicted boundary within `tolerance` px."""
    near = torch.nn.functional.max_pool2d(predicted[None, None].float(), 2 * tolerance + 1, 1, tolerance)[0, 0] > 0
    return near[truth].float().mean().item()


def _self_check():
    h, w = 48, 64
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    region = torch.where((xs - 40) ** 2 + (ys - 24) ** 2 < 100, 2, torch.where(ys < 20 + 6 * torch.sin(xs / 5), 1, 0))
    colours = torch.tensor([[200, 60, 50], [40, 90, 200], [240, 220, 60]], dtype=torch.float64) / 255
    img = colours[region].permute(2, 0, 1)
    assert abs(rgb_to_lab(torch.ones(3, 1, 1, dtype=torch.float64))[0].item() - 100) < 0.1
    red = rgb_to_lab(torch.tensor([1.0, 0, 0], dtype=torch.float64).reshape(3, 1, 1)).flatten()
    assert torch.allclose(red, torch.tensor([53.24, 80.09, 67.20], dtype=torch.float64), atol=0.1), red
    truth = boundaries(region)
    loose = boundary_recall(truth, boundaries(slic(img, 12, 5)))
    stiff = boundary_recall(truth, boundaries(slic(img, 12, 200)))
    assert loose > 0.9 and stiff < loose - 0.15, (loose, stiff)
    print(f"slic.py: ok (boundary recall: m=5 {loose:.0%}, m=200 {stiff:.0%})")


if __name__ == "__main__":
    _self_check()
