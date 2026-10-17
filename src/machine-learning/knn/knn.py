"""
k-nearest neighbours in PyTorch: the same classifier as knn.ts, written with whole-matrix ops.

    distances   D = cdist(queries, train)               (Euclidean, n_queries × n_train)
    neighbours  the k smallest distances in each row    (topk with largest=False)
    vote        uniform: 1 per neighbour,  distance: 1 / d per neighbour
    predict     the class with the most votes; vote fractions estimate p(class | x)

The self-check verifies that k = 1 reproduces the training labels, that kNN generalises
on noisy blobs, uniform against distance-weighted voting on a small hand-made example,
and the curse of dimensionality: for uniform random points the nearest and the farthest
point from a query end up almost equally far as the dimension grows.
"""
import torch


def knn_votes(train_x, train_y, queries, k, classes, weighting="uniform"):
    """Vote fractions (n_queries × classes) of the k nearest training points of each query."""
    dist, index = torch.cdist(queries, train_x).topk(k, dim=1, largest=False)
    weight = torch.ones_like(dist) if weighting == "uniform" else 1 / (dist + 1e-9)
    votes = torch.zeros(len(queries), classes, dtype=dist.dtype)
    votes.scatter_add_(1, train_y[index], weight)                 # add each neighbour's weight to its class
    return votes / votes.sum(1, keepdim=True)


def knn_predict(train_x, train_y, queries, k, classes, weighting="uniform"):
    return knn_votes(train_x, train_y, queries, k, classes, weighting).argmax(1)


def distance_ratio(d, n, queries, generator):
    """Mean of nearest / farthest distance from uniform queries to n uniform points in [0, 1]^d."""
    points = torch.rand(n, d, generator=generator, dtype=torch.float64)
    q = torch.rand(queries, d, generator=generator, dtype=torch.float64)
    dist = torch.cdist(q, points)
    return (dist.min(1).values / dist.max(1).values).mean().item()


def _blobs(n, generator, noise=0.6, classes=3):
    y = torch.arange(n) % classes
    angle = 2 * torch.pi * y.double() / classes
    centres = torch.stack([angle.cos(), angle.sin()], 1)
    return centres + noise * torch.randn(n, 2, generator=generator, dtype=torch.float64), y


def _self_check():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(int(torch.randint(1 << 30, (1,))))

    # k = 1 memorises: every training point is its own nearest neighbour.
    x, y = _blobs(300, g)
    assert (knn_predict(x, y, x, 1, 3) == y).all()
    # ...and generalises: fresh points from the same blobs.
    xt, yt = _blobs(1000, g)
    acc = {k: (knn_predict(x, y, xt, k, 3) == yt).double().mean().item() for k in (1, 15)}
    assert acc[15] > 0.8 and acc[15] > acc[1] + 0.01, acc

    # Uniform vs distance-weighted: two class-1 points close to the query, three class-0 farther away.
    tx = torch.tensor([[0.1, 0], [-0.1, 0], [1, 0], [0, 1], [-1, 0.2]], dtype=torch.float64)
    ty = torch.tensor([1, 1, 0, 0, 0])
    q = torch.zeros(1, 2, dtype=torch.float64)
    assert knn_predict(tx, ty, q, 5, 2, "uniform").item() == 0       # 3 votes against 2
    assert knn_predict(tx, ty, q, 5, 2, "distance").item() == 1      # 2 × 1/0.1 beats 3 × about 1
    assert torch.allclose(knn_votes(tx, ty, q, 5, 2), torch.tensor([[0.6, 0.4]], dtype=torch.float64))

    # The curse of dimensionality: nearest / farthest climbs towards 1.
    dims = [1, 2, 10, 100, 1000]
    ratios = [distance_ratio(d, 500, 20, g) for d in dims]
    assert all(a < b for a, b in zip(ratios, ratios[1:])), ratios
    assert ratios[0] < 0.02 and ratios[-1] > 0.85, ratios
    print(f"knn.py: ok (blobs test accuracy k=1 {acc[1]:.3f}, k=15 {acc[15]:.3f}; nearest/farthest "
          + ", ".join(f"d={d}: {r:.3f}" for d, r in zip(dims, ratios)) + ")")


if __name__ == "__main__":
    _self_check()
