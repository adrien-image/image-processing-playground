"""
Hough transforms in PyTorch — the voting of hough.ts, vectorised.

Lines: every edge pixel votes for all 180 lines (ρ, θ) through it. With tensors that is one
(edge pixels × 180) matrix of ρ values and a single bincount into the accumulator. Peaks
are cells that equal the maximum of their neighbourhood (max-pooling), with θ wrapping
round: (θ = 180°, ρ) is the same line as (0°, −ρ).

Circles: an edge pixel on a circle has its gradient along the radius, so for each radius
it votes only for the two centres at distance r along ±gradient. Votes are smoothed over
neighbouring cells, scored as a fraction of the circumference, and peaks are found in
(r, y, x) with a 3-D max-pool.

(The TypeScript version additionally traces each line to find its end points.)
"""
import math
import torch
import torch.nn.functional as F

NUM_THETA = 180


def hough_lines(edges: torch.Tensor, threshold: int, max_lines: int = 20):
    """Returns the accumulator (θ, ρ) and the peaks as (ρ, θ in radians, votes), strongest first."""
    h, w = edges.shape
    max_rho = math.ceil(math.hypot(w, h))
    num_rho = 2 * max_rho + 1
    theta = torch.arange(NUM_THETA, dtype=torch.float64) * math.pi / NUM_THETA
    ys, xs = torch.nonzero(edges >= 128, as_tuple=True)
    rho = torch.round(xs[:, None] * torch.cos(theta) + ys[:, None] * torch.sin(theta)).long() + max_rho
    cells = torch.arange(NUM_THETA) * num_rho + rho                         # (pixels, θ)
    acc = torch.bincount(cells.flatten(), minlength=NUM_THETA * num_rho).reshape(NUM_THETA, num_rho)

    # Wrap θ: two rows from the end go before row 0 and two from the start after the last
    # row, each with ρ reversed, so a 5×9 max-pool sees across the 0°/180° seam.
    wrapped = torch.cat([acc[-2:].flip(1), acc, acc[:2].flip(1)])
    local_max = F.max_pool2d(wrapped[None, None].double(), (5, 9), stride=1, padding=(0, 4))[0, 0]
    peaks = (acc == local_max) & (acc >= threshold)
    t, r = torch.nonzero(peaks, as_tuple=True)
    votes = acc[t, r]
    order = votes.argsort(descending=True)[:max_lines]
    return acc, [(int(r[i]) - max_rho, float(theta[t[i]]), int(votes[i])) for i in order]


def hough_circles(edges: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor, min_r: int, max_r: int, threshold: float, max_circles=10):
    """Circles as (x, y, r, score) where score ≈ fraction of the circumference with edge support."""
    h, w = edges.shape
    ys, xs = torch.nonzero((edges >= 128) & (torch.hypot(gx, gy) > 0), as_tuple=True)
    mag = torch.hypot(gx[ys, xs], gy[ys, xs])
    ux, uy = gx[ys, xs] / mag, gy[ys, xs] / mag
    radii = torch.arange(min_r, max_r + 1, dtype=torch.float64)
    acc = torch.zeros(len(radii), h, w, dtype=torch.float64)
    for sign in (1, -1):
        cx = torch.round(xs[None] + sign * radii[:, None] * ux[None]).long()
        cy = torch.round(ys[None] + sign * radii[:, None] * uy[None]).long()
        ri = torch.arange(len(radii))[:, None].expand_as(cx)
        ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        acc.index_put_((ri[ok], cy[ok], cx[ok]), torch.ones(int(ok.sum()), dtype=torch.float64), accumulate=True)

    # Rounding scatters a circle's votes over neighbouring cells: count those too, at half weight.
    k = torch.tensor([[0.25, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.25]], dtype=torch.float64)
    smoothed = F.conv2d(acc[:, None], k[None, None], padding=1)[:, 0]
    score = smoothed / (2 * math.pi * radii[:, None, None])
    local_max = F.max_pool3d(score[None, None], 7, stride=1, padding=3)[0, 0]
    r_i, y, x = torch.nonzero((score == local_max) & (score >= threshold), as_tuple=True)
    candidates = sorted(zip(x.tolist(), y.tolist(), (r_i + min_r).tolist(), score[r_i, y, x].tolist()), key=lambda c: -c[3])
    circles = []
    for c in candidates:  # a stroked circle has an inner and an outer edge: keep one
        if any(math.hypot(k[0] - c[0], k[1] - c[1]) <= 4 and abs(k[2] - c[2]) <= 6 for k in circles):
            continue
        circles.append(c)
        if len(circles) >= max_circles:
            break
    return circles


def _self_check():
    edges = torch.zeros(80, 100)
    edges[30, 10:90] = 255                       # horizontal line y = 30: θ = 90°, ρ = 30
    for t in range(10, 70):
        edges[t, t + 5] = 255                    # diagonal y = x − 5
    _, lines = hough_lines(edges, threshold=40)
    (rho, theta, votes) = lines[0]
    assert (rho, round(math.degrees(theta))) == (30, 90) and votes >= 80, lines[0]
    assert any(round(math.degrees(th)) == 135 for _, th, _ in lines[:3]), lines[:3]

    # A circle of radius 15 at (50, 40) with outward-pointing gradients.
    ys, xs = torch.meshgrid(torch.arange(80.), torch.arange(100.), indexing="ij")
    d = torch.hypot(xs - 50, ys - 40)
    ring = ((d - 15).abs() < 0.5).float() * 255
    circles = hough_circles(ring, (xs - 50) / d.clamp(min=1), (ys - 40) / d.clamp(min=1), 8, 25, 0.5)
    x, y, r, score = circles[0]
    assert (x, y, r) == (50, 40, 15) and score > 0.8, circles[0]
    print("hough.py: ok")


if __name__ == "__main__":
    _self_check()
