"""
K-means on gray levels in PyTorch — the same algorithm as kmeans.ts.

Each iteration assigns every pixel to its nearest centroid (one broadcasted distance
matrix, pixels × k) and moves each centroid to the mean of its pixels, until no label
changes. Centroids start evenly spread over the value range, so the result is repeatable.
"""
import torch


def kmeans(pixels: torch.Tensor, k: int, max_iter: int = 50):
    """Returns (centroids (k,), labels (N,), iterations)."""
    x = pixels.flatten().double()
    lo, hi = x.min(), x.max()
    centroids = lo + (hi - lo) * (torch.arange(k, dtype=torch.float64) + 0.5) / k
    labels = torch.full_like(x, -1, dtype=torch.long)
    for iteration in range(1, max_iter + 1):
        new_labels = (x[:, None] - centroids[None, :]).abs().argmin(dim=1)
        sums = torch.zeros(k, dtype=torch.float64).index_add_(0, new_labels, x)
        counts = torch.bincount(new_labels, minlength=k)
        moved = torch.where(counts > 0, sums / counts.clamp(min=1), centroids)  # empty clusters stay put
        if torch.equal(new_labels, labels) and torch.equal(moved, centroids):
            return centroids, labels, iteration
        labels, centroids = new_labels, moved
    return centroids, labels, max_iter


def labels_to_image(labels: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Replace every pixel by its cluster's centroid value."""
    return centroids[labels]


def _self_check():
    torch.manual_seed(0)
    pixels = torch.cat([torch.full((300,), 40.), torch.full((300,), 128.), torch.full((300,), 220.)]) + torch.randn(900) * 4
    centroids, labels, iterations = kmeans(pixels, 3)
    assert torch.allclose(centroids, torch.tensor([40., 128., 220.], dtype=torch.float64), atol=1.5), centroids
    assert (labels[:300] == 0).all() and (labels[600:] == 2).all()
    assert iterations < 50
    print("kmeans.py: ok")


if __name__ == "__main__":
    _self_check()
