"""
Feature matching in PyTorch — harris.ts, descriptors.ts and transform.ts in one file.

  1. Harris corners      structure tensor from Sobel gradients, summed over 5×5;
                         R = det(M) − k·trace(M)²; keep 3×3 local maxima (max-pooling)
  2. descriptors         a 9×9 patch sampled in each corner's dominant-gradient frame
                         (all corners at once with grid_sample), normalised to mean 0, std 1
  3. matching            all-pairs distances (cdist), Lowe's ratio test + mutual best match
  4. RANSAC similarity   score hundreds of 2-point hypotheses in one batched tensor op,
                         then refit by least squares on the inliers

Images are float tensors (H, W), values 0–255.
"""
import math
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
PATCH_RADIUS = 4
ORIENTATION_RADIUS = 6
RATIO = 0.8


def _filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r = kernel.shape[-1] // 2
    x = F.pad(image[None, None], (r, r, r, r), mode="replicate")
    return F.conv2d(x, kernel[None, None])[0, 0]


def harris(image: torch.Tensor, max_corners=200, k=0.05, quality=0.01) -> torch.Tensor:
    """Corners as an (N, 3) tensor of (x, y, score), strongest first."""
    gx, gy = _filter(image, SOBEL_X), _filter(image, SOBEL_X.T.contiguous())
    window = torch.ones(5, 5)
    ixx, iyy, ixy = _filter(gx * gx, window), _filter(gy * gy, window), _filter(gx * gy, window)
    response = ixx * iyy - ixy ** 2 - k * (ixx + iyy) ** 2
    response[:2], response[-2:], response[:, :2], response[:, -2:] = 0, 0, 0, 0  # full 5×5 windows only
    # A pixel is a corner if it equals the maximum of its 3×3 neighbourhood.
    local_max = F.max_pool2d(response[None, None], 3, stride=1, padding=1)[0, 0]
    keep = (response == local_max) & (response > 0) & (response >= quality * response.max())
    ys, xs = torch.nonzero(keep, as_tuple=True)
    scores = response[ys, xs]
    order = scores.argsort(descending=True)[:max_corners]
    return torch.stack([xs[order].double(), ys[order].double(), scores[order].double()], 1)


def _sample(image: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Bilinear samples at pixel coordinates (clamped to the image), any shape."""
    h, w = image.shape
    grid = torch.stack([2 * xs.clamp(0, w - 1) / (w - 1) - 1, 2 * ys.clamp(0, h - 1) / (h - 1) - 1], -1)
    out = F.grid_sample(image[None, None].double(), grid.reshape(1, -1, 1, 2).double(), align_corners=True)
    return out.reshape(xs.shape)


def descriptors(image: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
    """(N, 81) rotation-normalised patch descriptors."""
    x, y = corners[:, 0, None, None], corners[:, 1, None, None]
    r = torch.arange(-ORIENTATION_RADIUS, ORIENTATION_RADIUS + 1, dtype=torch.float64)
    dy, dx = torch.meshgrid(r, r, indexing="ij")
    gx = _sample(image, x + dx + 1, y + dy) - _sample(image, x + dx - 1, y + dy)
    gy = _sample(image, x + dx, y + dy + 1) - _sample(image, x + dx, y + dy - 1)
    weight = torch.exp(-(dx ** 2 + dy ** 2) / (2 * (ORIENTATION_RADIUS / 2) ** 2))
    angle = torch.atan2((gy * weight).sum((1, 2)), (gx * weight).sum((1, 2)))[:, None, None]

    p = torch.arange(-PATCH_RADIUS, PATCH_RADIUS + 1, dtype=torch.float64)
    v, u = torch.meshgrid(p, p, indexing="ij")
    c, s = torch.cos(angle), torch.sin(angle)
    patch = _sample(image, x + u * c - v * s, y + u * s + v * c).flatten(1)
    patch = patch - patch.mean(1, keepdim=True)
    std = patch.pow(2).mean(1, keepdim=True).sqrt()
    return torch.where(std > 1e-6, patch / std.clamp(min=1e-6), patch)


def match(desc_a: torch.Tensor, desc_b: torch.Tensor) -> torch.Tensor:
    """(M, 2) index pairs passing the ratio test and the mutual-best check."""
    d = torch.cdist(desc_a, desc_b)                       # Euclidean distances
    best2 = d.topk(2, dim=1, largest=False)
    ratio_ok = best2.values[:, 0] < RATIO * best2.values[:, 1]
    j = best2.indices[:, 0]
    mutual = d.argmin(dim=0)[j] == torch.arange(len(desc_a))
    i = torch.nonzero(ratio_ok & mutual).flatten()
    return torch.stack([i, j[i]], 1)


def fit_similarity(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Least-squares (a, b, tx, ty) with x' = a·x − b·y + tx, y' = b·x + a·y + ty. Works on batches (..., N, 2)."""
    ms, md = src.mean(-2, keepdim=True), dst.mean(-2, keepdim=True)
    p, q = src - ms, dst - md
    den = (p ** 2).sum((-1, -2))
    a = (q[..., 0] * p[..., 0] + q[..., 1] * p[..., 1]).sum(-1) / den
    b = (q[..., 1] * p[..., 0] - q[..., 0] * p[..., 1]).sum(-1) / den
    tx = md[..., 0, 0] - (a * ms[..., 0, 0] - b * ms[..., 0, 1])
    ty = md[..., 0, 1] - (b * ms[..., 0, 0] + a * ms[..., 0, 1])
    return torch.stack([a, b, tx, ty], -1)


def apply_similarity(t: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    a, b, tx, ty = t[..., 0, None], t[..., 1, None], t[..., 2, None], t[..., 3, None]
    x, y = pts[..., 0], pts[..., 1]
    return torch.stack([a * x - b * y + tx, b * x + a * y + ty], -1)


def ransac_similarity(src: torch.Tensor, dst: torch.Tensor, iterations=500, threshold=3.0, seed=42):
    """Best similarity and the inlier mask; all hypotheses are scored in one batched op."""
    g = torch.Generator().manual_seed(seed)
    n = len(src)
    i = torch.randint(n, (iterations,), generator=g)
    j = (i + 1 + torch.randint(n - 1, (iterations,), generator=g)) % n      # j ≠ i
    pairs = torch.stack([i, j], 1)
    hypotheses = fit_similarity(src[pairs], dst[pairs])                       # (iterations, 4)
    err = (apply_similarity(hypotheses, src) - dst).norm(dim=-1)
    inliers = err <= threshold                                                # (iterations, n)
    best = inliers[int(inliers.sum(1).argmax())]
    refined = fit_similarity(src[best], dst[best])
    final = (apply_similarity(refined, src) - dst).norm(dim=-1) <= threshold
    a, b, tx, ty = refined.tolist()
    return {"angle": math.atan2(b, a), "scale": math.hypot(a, b), "tx": tx, "ty": ty}, final


def _warp_similarity(image: torch.Tensor, angle: float, scale: float, tx: float, ty: float) -> torch.Tensor:
    """Image B(p') = A(T⁻¹ p') for p' = s·R(angle)·p + t (pixel coordinates, origin top-left)."""
    h, w = image.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    c, s = math.cos(angle), math.sin(angle)
    u, v = (xs - tx) / scale, (ys - ty) / scale
    src_x, src_y = c * u + s * v, -s * u + c * v
    inside = (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    return torch.where(inside, _sample(image, src_x, src_y), torch.zeros(()).double()).float()


def _self_check():
    g = torch.Generator().manual_seed(1)
    img = torch.full((120, 140), 20.)
    for _ in range(14):  # rectangles of different brightness: plenty of corners
        x0, y0 = torch.randint(15, 100, (1,), generator=g).item(), torch.randint(15, 80, (1,), generator=g).item()
        w, h = torch.randint(8, 25, (2,), generator=g).tolist()
        img[y0:y0 + h, x0:x0 + w] = torch.randint(80, 255, (1,), generator=g).float()
    true = dict(angle=0.15, scale=0.95, tx=12.0, ty=-4.0)
    img_b = _warp_similarity(img, **true)

    ca, cb = harris(img), harris(img_b)
    assert len(ca) > 20 and len(cb) > 20
    pairs = match(descriptors(img, ca), descriptors(img_b, cb))
    t, inliers = ransac_similarity(ca[pairs[:, 0], :2], cb[pairs[:, 1], :2])
    assert inliers.sum() >= 8, inliers.sum()
    assert abs(t["angle"] - true["angle"]) < 0.03 and abs(t["scale"] - true["scale"]) < 0.03, t
    assert abs(t["tx"] - true["tx"]) < 2 and abs(t["ty"] - true["ty"]) < 2, t
    print(f"matching.py: ok ({len(pairs)} matches, {int(inliers.sum())} inliers)")


if __name__ == "__main__":
    _self_check()
