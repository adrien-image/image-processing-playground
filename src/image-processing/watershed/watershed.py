"""
Watershed in PyTorch — watershed.ts: distance transform, markers, flooding.

The exact Euclidean distance transform is separable: for each column, the squared distance
to the nearest background pixel in that column; then for each row, minimise
    g(x', y)² + (x − x')²    over x'
which is a (W, W) broadcast per row, done for all rows at once. Markers are plateau maxima
(max_pool2d), grouped into connected labels by label propagation. The flooding itself is a
priority queue, which is inherently sequential, so it runs in plain Python with heapq.
"""
import heapq

import torch
import torch.nn.functional as F


def distance_transform(mask: torch.Tensor) -> torch.Tensor:
    """(H, W) bool → (H, W) Euclidean distance to the nearest False pixel."""
    h, w = mask.shape
    big = float(h * h + w * w)
    ys = torch.arange(h, dtype=torch.float64)
    # Columns: squared distance to the nearest background pixel in the same column.
    col = torch.where(~mask[:, None, :], (ys[:, None, None] - ys[None, :, None]) ** 2, torch.tensor(big, dtype=torch.float64))
    g = col.min(0).values                                       # (H, W): g[y, x]
    xs = torch.arange(w, dtype=torch.float64)
    sq = (g[:, None, :] + (xs[:, None] - xs[None, :]) ** 2).min(-1).values
    return sq.clamp(max=big).sqrt() * mask


def find_markers(dist: torch.Tensor, radius: int, min_distance: float):
    """Pixels that are the maximum within `radius` (square window here) and ≥ min_distance; one label per plateau."""
    local_max = F.max_pool2d(dist[None, None], 2 * radius + 1, 1, radius)[0, 0]
    peak = (dist >= local_max - 1e-9) & (dist >= min_distance)
    h, w = dist.shape
    labels = torch.where(peak, torch.arange(1, h * w + 1, dtype=torch.float64).reshape(h, w), torch.zeros(()))
    while True:                                                 # every plateau takes its largest index
        grown = F.max_pool2d(labels[None, None], 3, 1, 1)[0, 0] * peak
        if torch.equal(grown, labels):
            break
        labels = grown
    _, compact = torch.unique(labels, return_inverse=True)      # renumber 0, 1..count
    return compact, int(compact.max())


def watershed(surface: torch.Tensor, markers: torch.Tensor, mask: torch.Tensor):
    """Meyer's flooding from the markers over `surface` (lower floods first), within `mask`. Returns labels, dam lines."""
    h, w = surface.shape
    labels = markers.clone().tolist()
    s = surface.tolist()
    m = mask.tolist()
    lines = [[False] * w for _ in range(h)]
    queued = [[False] * w for _ in range(h)]
    heap, order = [], 0

    def neighbours(y, x):
        for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= yy < h and 0 <= xx < w:
                yield yy, xx

    def push(y, x, level):
        nonlocal order
        queued[y][x] = True
        heapq.heappush(heap, (level, order, y, x))
        order += 1

    for y in range(h):
        for x in range(w):
            if labels[y][x]:
                for yy, xx in neighbours(y, x):
                    if not labels[yy][xx] and m[yy][xx] and not queued[yy][xx]:
                        push(yy, xx, s[yy][xx])
    while heap:
        level, _, y, x = heapq.heappop(heap)
        found = {labels[yy][xx] for yy, xx in neighbours(y, x) if labels[yy][xx] > 0}
        if len(found) > 1:
            lines[y][x] = True                                   # two floods meet: a dam
            continue
        labels[y][x] = found.pop()
        for yy, xx in neighbours(y, x):
            if not labels[yy][xx] and m[yy][xx] and not queued[yy][xx]:
                push(yy, xx, max(s[yy][xx], level))
    return torch.tensor(labels), torch.tensor(lines)


def _self_check():
    h, w = 50, 80
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    coins = ((xs - 25) ** 2 + (ys - 25) ** 2 <= 196) | ((xs - 50) ** 2 + (ys - 25) ** 2 <= 196)

    single = torch.ones(h, w, dtype=torch.bool)
    single[10, 10] = False
    d = distance_transform(single)
    assert torch.allclose(d, ((xs - 10) ** 2 + (ys - 10) ** 2).sqrt()), "distance transform is not exact"

    dist = distance_transform(coins)
    markers, count = find_markers(dist, 5, 3)
    assert count == 2, count
    labels, lines = watershed(-dist, markers, coins)
    assert labels[25, 20] != labels[25, 55] and labels[25, 20] > 0 and labels[25, 55] > 0
    assert lines[20:31, 37:39].any(1).all(), "no dam along the neck"
    assert ((labels > 0) | lines)[coins].all() and not (labels > 0)[~coins].any()
    print(f"watershed.py: ok (2 touching coins → {count} markers → split along x ≈ 37.5)")


if __name__ == "__main__":
    _self_check()
