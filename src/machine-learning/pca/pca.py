"""
PCA in PyTorch: the principal components are the top eigenvectors of the covariance

    C = (1/n) Σᵢ (xᵢ − μ)(xᵢ − μ)ᵀ,     reconstruction  x ≈ μ + Σₖ ((x − μ)·vₖ) vₖ

found as in pca.ts by subspace iteration (repeat V ← orthonormalise(C·V), then
Rayleigh–Ritz on the small matrix VᵀCV), and checked against torch.linalg.eigh, which
diagonalises the whole matrix. Rows of `components` are the unit eigenvectors.
"""
import torch


def covariance(X):
    mean = X.mean(0)
    Xc = X - mean
    return mean, Xc.T @ Xc / len(X)


def gram_schmidt(V):
    """Orthonormalise the columns of V (modified Gram–Schmidt, twice)."""
    V = V.clone()
    for _ in range(2):
        for a in range(V.shape[1]):
            for b in range(a):
                V[:, a] -= (V[:, a] @ V[:, b]) * V[:, b]
            V[:, a] /= V[:, a].norm()
    return V


def top_eigen(C, k, iterations=8, oversample=10, generator=None):
    """The k largest eigenvalues (decreasing) and eigenvectors (rows) of a symmetric PSD matrix."""
    d = C.shape[0]
    p = min(d, k + oversample)
    V = torch.linalg.qr(torch.randn(d, p, generator=generator, dtype=C.dtype)).Q
    for _ in range(iterations):
        V = gram_schmidt(C @ V)          # torch.linalg.qr(C @ V).Q does the same job, faster
    values, S = torch.linalg.eigh(V.T @ C @ V)      # Rayleigh–Ritz on the p × p problem
    order = values.argsort(descending=True)[:k]
    vectors = (V @ S[:, order]).T
    # Fix the arbitrary sign: largest entry positive, as in pca.ts.
    big = vectors.gather(1, vectors.abs().argmax(1, keepdim=True))
    return values[order].clamp(min=0), vectors * big.sign()


def fit_pca(X, k, **kwargs):
    mean, C = covariance(X)
    variances, components = top_eigen(C, k, **kwargs)
    return {"mean": mean, "components": components, "variances": variances, "total": C.trace()}


def reconstruct(pca, x, k):
    V = pca["components"][:k]
    return pca["mean"] + ((x - pca["mean"]) @ V.T) @ V


def explained_variance(pca):
    ratio = pca["variances"] / pca["total"]
    return ratio, ratio.cumsum(0)


def _self_check():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(torch.initial_seed() + 1)   # follows the global seed
    d = 30
    # A covariance with a known spectrum: C = Q diag(λ) Qᵀ.
    Q = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64)).Q
    lam = 10 * 0.8 ** torch.arange(d, dtype=torch.float64)
    C = Q @ torch.diag(lam) @ Q.T
    values, vectors = top_eigen(C, 6, iterations=30, generator=g)
    assert torch.allclose(values, lam[:6], rtol=1e-8), (values, lam[:6])
    assert torch.allclose((vectors @ Q[:, :6]).abs(), torch.eye(6, dtype=torch.float64), atol=1e-6)

    # Samples drawn with that covariance: PCA finds the same directions, and agrees with eigh.
    X = torch.randn(4000, d, generator=g, dtype=torch.float64) * lam.sqrt() @ Q.T + 3.0
    pca = fit_pca(X, 5, generator=g)
    _, C_hat = covariance(X)
    ref_values, ref_vectors = torch.linalg.eigh(C_hat)
    ref_values, ref_vectors = ref_values.flip(0)[:5], ref_vectors.flip(1)[:, :5].T
    assert torch.allclose(pca["variances"], ref_values, rtol=1e-6)
    assert torch.allclose((pca["components"] @ ref_vectors.T).abs(), torch.eye(5, dtype=torch.float64), atol=1e-4)
    assert (pca["components"] @ Q[:, :3]).diagonal()[:3].abs().min() > 0.95   # near the true axes (sampling noise)
    V = pca["components"]
    assert torch.allclose(V @ V.T, torch.eye(5, dtype=torch.float64), atol=1e-10)   # orthonormal

    # All d components rebuild the data exactly; the variances add up to the total.
    full = fit_pca(X, d, oversample=0, iterations=60, generator=g)
    assert torch.allclose(reconstruct(full, X, d), X, atol=1e-8)
    ratio, cumulative = explained_variance(full)
    assert abs(cumulative[-1].item() - 1) < 1e-9
    error = lambda k: ((reconstruct(full, X, k) - X) ** 2).sum(1).mean().item()
    # The mean squared error of a k-component reconstruction is the variance left out.
    assert abs(error(5) - full["variances"][5:].sum().item()) < 1e-8 * full["total"].item()
    print(f"pca.py: ok (top eigenvalues {values[0]:.3f}, {values[1]:.3f}, {values[2]:.3f}; "
          f"5 of {d} components explain {cumulative[4].item():.1%} of the variance)")


if __name__ == "__main__":
    _self_check()
