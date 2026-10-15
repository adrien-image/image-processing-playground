"""
Pyramidal Lucas–Kanade optical flow in PyTorch — opticalFlow.ts, vectorised over points.

Every tracked point gets a (2r+1)² window. All windows of all points are sampled in one
grid_sample call (frame 2 at each point's current flow estimate), so one iteration is a
handful of tensor ops: gradients, the structure-tensor sums ΣIx², ΣIxIy, ΣIy², ΣIxIt,
ΣIyIt per point, and the 2×2 solve written out in closed form. Coarse-to-fine pyramid
levels handle motions larger than a pixel or two. Frames are float tensors (H, W).
"""
import torch
import torch.nn.functional as F

BINOMIAL = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float64) / 16


def downsample(img: torch.Tensor) -> torch.Tensor:
    """Binomial blur, then keep every second pixel."""
    k = (BINOMIAL[:, None] * BINOMIAL[None, :])[None, None]
    x = F.pad(img[None, None], (2, 2, 2, 2), mode="replicate")
    return F.conv2d(x, k)[0, 0, ::2, ::2]


def sample(img: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Bilinear samples at pixel coordinates of any shape (borders clamped)."""
    h, w = img.shape
    grid = torch.stack([2 * xs.clamp(0, w - 1) / (w - 1) - 1, 2 * ys.clamp(0, h - 1) / (h - 1) - 1], -1)
    return F.grid_sample(img[None, None], grid.reshape(1, -1, 1, 2), align_corners=True).reshape(xs.shape)


def lucas_kanade(f1, f2, points, radius=5, u=None, v=None, iterations=5):
    """Flow (u, v) and the structure tensor's smaller eigenvalue (per window pixel) at each point (N, 2)."""
    n = len(points)
    u = torch.zeros(n, dtype=f1.dtype) if u is None else u
    v = torch.zeros(n, dtype=f1.dtype) if v is None else v
    r = torch.arange(-radius, radius + 1, dtype=f1.dtype)
    oy, ox = torch.meshgrid(r, r, indexing="ij")
    px = points[:, 0, None] + ox.flatten()                     # (N, K) window pixel coordinates
    py = points[:, 1, None] + oy.flatten()
    for _ in range(iterations):
        qx, qy = px + u[:, None], py + v[:, None]              # where each window lands in frame 2
        ix = (sample(f1, px + 1, py) - sample(f1, px - 1, py) + sample(f2, qx + 1, qy) - sample(f2, qx - 1, qy)) / 4
        iy = (sample(f1, px, py + 1) - sample(f1, px, py - 1) + sample(f2, qx, qy + 1) - sample(f2, qx, qy - 1)) / 4
        it = sample(f2, qx, qy) - sample(f1, px, py)
        sxx, sxy, syy = (ix * ix).sum(1), (ix * iy).sum(1), (iy * iy).sum(1)
        sxt, syt = (ix * it).sum(1), (iy * it).sum(1)
        det = sxx * syy - sxy ** 2
        ok = det.abs() > 1e-9
        safe = torch.where(ok, det, torch.ones_like(det))
        u = u + torch.where(ok, (-syy * sxt + sxy * syt) / safe, torch.zeros_like(det))
        v = v + torch.where(ok, (sxy * sxt - sxx * syt) / safe, torch.zeros_like(det))
    tr = sxx + syy
    min_eigen = (tr / 2 - torch.sqrt((tr ** 2 / 4 - det).clamp(min=0))) / (2 * radius + 1) ** 2
    return u, v, min_eigen


def pyramidal_lucas_kanade(f1, f2, points, levels=3, radius=5):
    """Estimate on the coarsest level first; double the flow and refine on each finer level."""
    p1, p2 = [f1.double()], [f2.double()]
    for _ in range(levels - 1):
        p1.append(downsample(p1[-1]))
        p2.append(downsample(p2[-1]))
    u = v = None
    for level in reversed(range(levels)):
        s = 2 ** level
        if u is not None:
            u, v = 2 * u, 2 * v
        u, v, min_eigen = lucas_kanade(p1[level], p2[level], points.double() / s, radius, u, v)
    return u, v, min_eigen


def grid_points(h: int, w: int, step: int, margin: int) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(margin, h - margin, step), torch.arange(margin, w - margin, step), indexing="ij")
    return torch.stack([xs.flatten(), ys.flatten()], 1).double()


def texture(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The same multi-scale synthetic texture as opticalFlow.ts."""
    return (128 + 30 * torch.sin(x * 0.045 + 1) * torch.cos(y * 0.05 - 0.5) + 22 * torch.cos((x - 2 * y) * 0.03)
            + 25 * torch.sin(x * 0.23 + torch.sin(y * 0.07) * 2) + 20 * torch.cos(y * 0.19 - x * 0.05)
            + 18 * torch.sin((x + y) * 0.11) * torch.cos((x - y) * 0.13))


def _self_check():
    h, w = 80, 96
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    points = grid_points(h, w, 12, 16)
    errors = []
    for shift, levels, tol in [(1.2, 1, 0.05), (10.0, 3, 0.1)]:
        f1, f2 = texture(xs, ys), texture(xs - shift, ys - shift / 2)   # content moves by (shift, shift/2)
        u, v, _ = pyramidal_lucas_kanade(f1, f2, points, levels)
        err = torch.hypot(u - shift, v - shift / 2).mean().item()
        assert err < tol, (shift, levels, err)
        errors.append(err)
    u1, v1, _ = pyramidal_lucas_kanade(texture(xs, ys), texture(xs - 10, ys - 5), points, 1)
    assert torch.hypot(u1 - 10, v1 - 5).mean() > 1                   # without a pyramid, 10 px is too far
    # Aperture problem: on vertical stripes the structure tensor is singular.
    stripes = lambda x: 128 + 90 * torch.sin(x * 0.35)               # noqa: E731
    _, _, min_eigen = lucas_kanade(stripes(xs), stripes(xs - 1), points)
    assert min_eigen.abs().max() < 1e-6
    print(f"optical_flow.py: ok (mean error {errors[0]:.3f} px for a 1.2 px shift, {errors[1]:.3f} px for 10 px with 3 levels)")


if __name__ == "__main__":
    _self_check()
