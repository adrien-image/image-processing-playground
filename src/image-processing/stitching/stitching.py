"""
Panorama stitching in PyTorch — homography.ts with tensors.

DLT: each match (p → q) gives two rows of A; H is the right singular vector of A with the
smallest singular value (torch.linalg.svd), after Hartley normalisation of both point sets.
RANSAC: fit H to many random 4-match samples at once (a batched SVD), count how many matches
each agrees with, keep the best and refit on its inliers.
Warping: F.grid_sample with a grid of B-coordinates computed from H⁻¹ (inverse mapping), and
feathering weights = distance to each image's border.
"""
import torch
import torch.nn.functional as F


def apply_h(H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """H (…, 3, 3) applied to points (…, N, 2)."""
    homog = torch.cat([pts, torch.ones_like(pts[..., :1])], -1) @ H.transpose(-1, -2)
    return homog[..., :2] / homog[..., 2:]


def _normalisation(pts: torch.Tensor) -> torch.Tensor:
    """(…, 3, 3) similarity moving the centroid to 0 and the mean distance to √2."""
    mean = pts.mean(-2)
    scale = 2 ** 0.5 / (pts - mean[..., None, :]).norm(dim=-1).mean(-1).clamp(min=1e-12)
    T = torch.zeros(*pts.shape[:-2], 3, 3, dtype=pts.dtype)
    T[..., 0, 0] = T[..., 1, 1] = scale
    T[..., :2, 2] = -scale[..., None] * mean
    T[..., 2, 2] = 1
    return T


def fit_homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Least-squares H with dst ≈ H·src; works on a batch: src, dst (…, N, 2) with N ≥ 4."""
    Ts, Td = _normalisation(src), _normalisation(dst)
    p, q = apply_h(Ts, src), apply_h(Td, dst)
    x, y, u, v = p[..., 0], p[..., 1], q[..., 0], q[..., 1]
    zero, one = torch.zeros_like(x), torch.ones_like(x)
    rows_x = torch.stack([-x, -y, -one, zero, zero, zero, u * x, u * y, u], -1)
    rows_y = torch.stack([zero, zero, zero, -x, -y, -one, v * x, v * y, v], -1)
    A = torch.cat([rows_x, rows_y], -2)
    if A.shape[-2] < 9:                                  # 4 matches = 8 rows: pad so the SVD gives all 9 vectors
        A = torch.cat([A, torch.zeros(*A.shape[:-2], 9 - A.shape[-2], 9, dtype=A.dtype)], -2)
    h = torch.linalg.svd(A).Vh[..., -1, :].reshape(*A.shape[:-2], 3, 3)
    H = torch.linalg.inv(Td) @ h @ Ts
    return H / H[..., 2:, 2:]


def ransac_homography(src, dst, iterations=500, threshold=3.0, generator=None):
    n = src.shape[0]
    samples = torch.stack([torch.randperm(n, generator=generator)[:4] for _ in range(iterations)])
    Hs = fit_homography(src[samples], dst[samples])                       # (iterations, 3, 3)
    Hs = torch.where(torch.isfinite(Hs).all(-1).all(-1)[:, None, None], Hs, torch.eye(3, dtype=Hs.dtype))
    errors = (apply_h(Hs, src.expand(iterations, n, 2)) - dst).norm(dim=-1)  # (iterations, N)
    inliers = errors.nan_to_num(float("inf")) <= threshold
    best = inliers[inliers.sum(1).argmax()]
    H = fit_homography(src[best], dst[best])
    final = (apply_h(H, src) - dst).norm(dim=-1) <= threshold
    return H, final


def stitch(a: torch.Tensor, b: torch.Tensor, H: torch.Tensor, blend="feather"):
    """a, b: (3, h, w) float images; H maps b's pixels into a's frame. Returns the panorama."""
    _, ha, wa = a.shape
    _, hb, wb = b.shape
    corners = apply_h(H, torch.tensor([[0., 0], [wb - 1, 0], [0, hb - 1], [wb - 1, hb - 1]], dtype=H.dtype))
    x0, y0 = min(0.0, corners[:, 0].min().item()), min(0.0, corners[:, 1].min().item())
    x1, y1 = max(wa - 1.0, corners[:, 0].max().item()), max(ha - 1.0, corners[:, 1].max().item())
    x0, y0 = int(torch.floor(torch.tensor(x0))), int(torch.floor(torch.tensor(y0)))
    width, height = int(x1) - x0 + 2, int(y1) - y0 + 2
    ys, xs = torch.meshgrid(torch.arange(height, dtype=H.dtype) + y0, torch.arange(width, dtype=H.dtype) + x0, indexing="ij")
    canvas = torch.stack([xs, ys], -1)                   # panorama pixel → coordinates in A's frame

    def warp(img, coords):
        """Bilinear sample img at pixel coords (h, w, 2); also return a weight that is 0 outside."""
        _, h, w = img.shape
        grid = torch.stack([coords[..., 0] / (w - 1) * 2 - 1, coords[..., 1] / (h - 1) * 2 - 1], -1)
        colour = F.grid_sample(img[None].to(grid.dtype), grid[None], align_corners=True)[0]
        border = torch.minimum(torch.minimum(coords[..., 0], coords[..., 1]), torch.minimum(w - 1 - coords[..., 0], h - 1 - coords[..., 1]))
        inside = border >= 0
        weight = (border.clamp(min=0) + 1) if blend == "feather" else torch.ones_like(border)
        return colour, weight * inside

    ca, wa_ = warp(a, canvas)
    cb, wb_ = warp(b, apply_h(torch.linalg.inv(H), canvas))
    if blend == "overwrite":
        wa_ = wa_ * (wb_ == 0)
    total = wa_ + wb_
    return torch.where(total > 0, (ca * wa_ + cb * wb_) / total.clamp(min=1e-12), torch.zeros_like(ca))


def camera_homography(angle_deg, shift, focal, cx, cy):
    """T · K · R_y(angle) · K⁻¹: the view of a camera turned about its vertical axis, then shifted."""
    t = torch.deg2rad(torch.tensor(float(angle_deg), dtype=torch.float64))
    K = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=torch.float64)
    R = torch.tensor([[torch.cos(t), 0, torch.sin(t)], [0, 1, 0], [-torch.sin(t), 0, torch.cos(t)]], dtype=torch.float64)
    T = torch.tensor([[1, 0, shift], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    H = T @ K @ R @ torch.linalg.inv(K)
    return H / H[2, 2]


def _self_check():
    g = torch.Generator().manual_seed(0)
    H_true = camera_homography(12, 90, 300, 100, 80)
    src = torch.rand(60, 2, generator=g, dtype=torch.float64) * torch.tensor([200., 160.], dtype=torch.float64)
    dst = apply_h(H_true, src)
    assert torch.allclose(fit_homography(src[:4], dst[:4]), H_true, atol=1e-6)
    assert torch.allclose(fit_homography(src, dst), H_true, atol=1e-6)

    noisy = dst + (torch.rand(dst.shape, generator=g, dtype=torch.float64) - 0.5)
    wrong = torch.arange(60) % 10 < 3                      # 30 % outliers
    noisy[wrong] = torch.rand(int(wrong.sum()), 2, generator=g, dtype=torch.float64) * 300
    least_squares = fit_homography(src, noisy)
    H, inliers = ransac_homography(src, noisy, threshold=2.0, generator=g)
    probe = torch.tensor([[0., 0.], [200., 160.]], dtype=torch.float64)
    err_ransac = (apply_h(H, probe) - apply_h(H_true, probe)).norm(dim=-1).max().item()
    err_ls = (apply_h(least_squares, probe) - apply_h(H_true, probe)).norm(dim=-1).max().item()
    assert err_ransac < 1.5 and err_ls > 10 * err_ransac, (err_ransac, err_ls)
    assert inliers[~wrong].float().mean() > 0.9 and not inliers[wrong].any()

    img = torch.full((3, 10, 20), 0.8, dtype=torch.float64)
    shift = torch.tensor([[1., 0, 15], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    pano = stitch(img, img, shift)
    assert pano.shape[-1] >= 35 and torch.allclose(pano[:, 5, 17], torch.tensor(0.8, dtype=torch.float64))
    print(f"stitching.py: ok (corner error: RANSAC {err_ransac:.2f} px, plain least squares {err_ls:.0f} px)")


if __name__ == "__main__":
    _self_check()
