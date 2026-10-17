"""
Active contours (snakes) in PyTorch: the same semi-implicit scheme as snake.ts.

    (I + τA) · x_{t+1} = x_t + τ · (w·∇|∇G_σ⋆I|² + balloon · normal)

The circulant matrix A is a sum of rolled identity matrices, inverted once with
torch.linalg.inv. The edge force is sampled at the snake's points with F.grid_sample,
and the normals and resampling are vectorised with torch.roll and torch.searchsorted.
Points are an (n, 2) tensor of (x, y) pixel positions.
"""
import math
import torch
import torch.nn.functional as F


def force_field(gray, sigma):
    r = math.ceil(3 * sigma)
    x = torch.arange(-r, r + 1, dtype=gray.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k /= k.sum()
    img = F.pad(gray[None, None], (r, r, 0, 0), mode="replicate")
    img = F.conv2d(F.pad(F.conv2d(img, k.view(1, 1, 1, -1)), (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))[0, 0]

    def grad(p):
        p = F.pad(p[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
        return (p[1:-1, 2:] - p[1:-1, :-2]) / 2, (p[2:, 1:-1] - p[:-2, 1:-1]) / 2

    gx, gy = grad(img)
    edges = gx ** 2 + gy ** 2
    edges = edges / edges.max().clamp(min=1e-300)
    fx, fy = grad(edges)
    scale = torch.hypot(fx, fy).max().clamp(min=1e-300)
    return edges, torch.stack([fx, fy]) / scale                       # (2, H, W)


def snake_matrix(n, alpha, beta, tau, dtype=torch.float64):
    eye = torch.eye(n, dtype=dtype)
    roll = lambda k: torch.roll(eye, k, dims=1)
    A = beta * (roll(2) + roll(-2)) + (-alpha - 4 * beta) * (roll(1) + roll(-1)) + (2 * alpha + 6 * beta) * eye
    return torch.linalg.inv(eye + tau * A)


def sample(force, points):
    _, h, w = force.shape
    grid = torch.stack([points[:, 0] / (w - 1) * 2 - 1, points[:, 1] / (h - 1) * 2 - 1], 1).view(1, 1, -1, 2)
    return F.grid_sample(force[None], grid, align_corners=True, padding_mode="border")[0, :, 0].T   # (n, 2)


def snake_step(points, inverse, force, tau, edge_weight, balloon):
    tangent = torch.roll(points, -1, 0) - torch.roll(points, 1, 0)
    x, y = points[:, 0], points[:, 1]
    orientation = torch.sign((x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum()) or 1.0
    normal = orientation * torch.stack([tangent[:, 1], -tangent[:, 0]], 1) / tangent.norm(dim=1, keepdim=True).clamp(min=1e-12)
    moved = inverse @ (points + tau * (edge_weight * sample(force, points) + balloon * normal))
    _, h, w = force.shape
    return torch.stack([moved[:, 0].clamp(0, w - 1), moved[:, 1].clamp(0, h - 1)], 1)


def resample(points):
    n = len(points)
    closed = torch.cat([points, points[:1]])
    seg = (closed[1:] - closed[:-1]).norm(dim=1)
    cum = torch.cat([seg.new_zeros(1), seg.cumsum(0)])
    targets = torch.arange(n, dtype=points.dtype) / n * cum[-1]
    i = (torch.searchsorted(cum, targets, right=True) - 1).clamp(0, n - 1)
    t = ((targets - cum[i]) / seg[i].clamp(min=1e-12))[:, None]
    return closed[i] + t * (closed[i + 1] - closed[i])


def circle(cx, cy, r, n):
    a = torch.arange(n, dtype=torch.float64) * 2 * math.pi / n
    return torch.stack([cx + r * torch.cos(a), cy + r * torch.sin(a)], 1)


def run(gray, points, steps, alpha=0.05, beta=0.05, tau=1.0, edge_weight=3.0, balloon=0.0, sigma=1.5):
    _, force = force_field(gray, sigma)
    inverse = snake_matrix(len(points), alpha, beta, tau)
    for i in range(steps):
        points = snake_step(points, inverse, force, tau, edge_weight, balloon)
        if i % 10 == 9:
            points = resample(points)
    return points


def _self_check():
    s = 64
    y, x = torch.meshgrid(torch.arange(s, dtype=torch.float64), torch.arange(s, dtype=torch.float64), indexing="ij")
    disc = torch.where(torch.hypot(x - 32, y - 30) < 14, 220.0, 30.0).double()
    radius = lambda p: torch.hypot(p[:, 0] - 32, p[:, 1] - 30).mean().item()
    # (I + τA)⁻¹ really is the inverse.
    n, alpha, beta, tau = 12, 0.3, 0.2, 2.0
    inv = snake_matrix(n, alpha, beta, tau)
    eye = torch.eye(n, dtype=torch.float64)
    A = beta * (torch.roll(eye, 2, 1) + torch.roll(eye, -2, 1)) + (-alpha - 4 * beta) * (torch.roll(eye, 1, 1) + torch.roll(eye, -1, 1)) + (2 * alpha + 6 * beta) * eye
    assert torch.allclose(inv @ (eye + tau * A), eye, atol=1e-10)
    # Deflating from outside and inflating from inside both stop on the disc's edge (radius 14).
    outside = run(disc, circle(32, 30, 26, 48), 300, balloon=-0.4)
    inside = run(disc, circle(30, 31, 4, 32), 300, balloon=0.4)
    assert abs(radius(outside) - 14) < 1.5 and abs(radius(inside) - 14) < 1.5
    print(f"snake.py: ok (radius {radius(outside):.2f} from outside, {radius(inside):.2f} from inside)")


if __name__ == "__main__":
    _self_check()
