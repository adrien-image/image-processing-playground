"""
Stereo block matching in PyTorch — stereo.ts as a cost volume.

For every disparity d at once: shift the right image d pixels right, take |L − R_shifted|,
average it over the window (avg_pool2d), stack the D + 1 cost maps into a (D+1, H, W)
volume and take the argmin along d. The left–right check matches the other way too and
keeps pixels whose two disparities agree within 1 px. Depth = f · B / d.
"""
import torch
import torch.nn.functional as F


def cost_volume(a: torch.Tensor, b: torch.Tensor, max_d: int, radius: int, a_is_left=True) -> torch.Tensor:
    """(D+1, H, W) mean absolute difference over the window; out-of-image matches cost 255."""
    h, w = a.shape
    costs = []
    for d in range(max_d + 1):
        shifted = torch.full_like(b, float("nan"))
        if a_is_left:
            shifted[:, d:] = b[:, :w - d]        # left pixel x is compared with right pixel x − d
        else:
            shifted[:, :w - d] = b[:, d:]        # right pixel x is compared with left pixel x + d
        diff = torch.nan_to_num((a - shifted).abs(), nan=255.0)
        k = 2 * radius + 1
        costs.append(F.avg_pool2d(F.pad(diff[None, None], (radius,) * 4, mode="replicate"), k, stride=1)[0, 0])
    return torch.stack(costs)


def block_match(left, right, max_d, radius):
    return cost_volume(left, right, max_d, radius).argmin(0)


def left_right_check(d_left: torch.Tensor, d_right: torch.Tensor) -> torch.Tensor:
    """Disparity where both directions agree within 1 px, −1 elsewhere."""
    h, w = d_left.shape
    xs = torch.arange(w).expand(h, w)
    xr = xs - d_left
    valid = xr >= 0
    back = d_right.gather(1, xr.clamp(min=0))
    return torch.where(valid & ((back - d_left).abs() <= 1), d_left, torch.full_like(d_left, -1))


def depth_from_disparity(disparity: torch.Tensor, focal_px: float, baseline: float) -> torch.Tensor:
    return torch.where(disparity > 0, focal_px * baseline / disparity.clamp(min=1e-6), torch.full_like(disparity, float("inf")))


def render_pair(h: int, w: int, noise: float = 0.0, generator=None):
    """Synthetic scene: textured wall (d=4), box (10), disc (16), pole (22). Returns left, right, true disparity, occluded."""
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")

    def tex(x, y, seed):
        s = seed * 12.9898
        return 128 + 45 * torch.sin(x * 0.9 + s) * torch.cos(y * 0.7 - s) + 35 * torch.sin(x * 0.31 + y * 0.53 + s * 1.7) + 25 * torch.cos(x * 1.7 - y * 1.3 + s * 0.3)

    layers = [  # (disparity, seed, membership of left-image coordinates), far to near
        (4, 1, lambda x, y: torch.ones_like(x, dtype=torch.bool)),
        (10, 2, lambda x, y: (x > w * 0.12) & (x < w * 0.45) & (y > h * 0.2) & (y < h * 0.8)),
        (16, 3, lambda x, y: (x - w * 0.68) ** 2 + (y - h * 0.45) ** 2 < (h * 0.22) ** 2),
        (22, 4, lambda x, y: (x > w * 0.86) & (x < w * 0.9)),
    ]
    left, right, disparity = torch.zeros(h, w, dtype=torch.float64), torch.zeros(h, w, dtype=torch.float64), torch.zeros(h, w)
    for d, seed, inside in layers:                       # painter's order: nearer layers overwrite
        m = inside(xs, ys)
        left = torch.where(m, tex(xs, ys, seed), left)
        disparity = torch.where(m, torch.tensor(float(d)), disparity)
        mr = inside(xs + d, ys)                          # right camera: the layer appears d px further left
        right = torch.where(mr, tex(xs + d, ys, seed), right)
    occluded = (xs - disparity) < 0
    for d, _, inside in layers:
        occluded |= (d > disparity) & inside(xs - disparity + d, ys)
    if noise > 0:
        left = left + noise * torch.randn(h, w, generator=generator, dtype=torch.float64)
        right = right + noise * torch.randn(h, w, generator=generator, dtype=torch.float64)
    return left, right, disparity.long(), occluded


def _self_check():
    g = torch.Generator().manual_seed(2)
    h, w, max_d = 80, 120, 26
    left, right, truth, occluded = render_pair(h, w, noise=3, generator=g)
    region = torch.zeros(h, w, dtype=torch.bool)
    region[:, max_d:] = True
    visible = region & ~occluded

    def bad(d):
        ok = visible & (d >= 0)
        return ((d - truth).abs() > 1)[ok].float().mean().item()

    tiny, medium = bad(block_match(left, right, max_d, 0)), bad(block_match(left, right, max_d, 3))
    assert medium < 0.1 and tiny > 2 * medium, (tiny, medium)
    d_left = block_match(left, right, max_d, 3)
    d_right = cost_volume(right, left, max_d, 3, a_is_left=False).argmin(0)
    checked = left_right_check(d_left, d_right)
    caught = (checked[region & occluded] < 0).float().mean().item()
    assert caught > 0.6, caught
    assert torch.isinf(depth_from_disparity(torch.tensor([0.]), 500, 0.1))[0]
    print(f"stereo.py: ok (bad pixels: 1×1 window {tiny:.0%}, 7×7 window {medium:.0%}; occlusions caught {caught:.0%})")


if __name__ == "__main__":
    _self_check()
