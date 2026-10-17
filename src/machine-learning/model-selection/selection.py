"""
Model selection in PyTorch: polynomial least squares in the Legendre basis, k-fold
cross-validation and the bias–variance decomposition, as in selection.ts.

    fit       minimise (1/n) Σ (y − Φw)² + λ Σ_{j≥1} w_j²   →   (ΦᵀΦ/n + λD) w = Φᵀy/n
    CV        k fits, each trained without one fold and scored on it; average the k errors
    E[(f̂(x) − y)²] = (E f̂(x) − f(x))² + Var f̂(x) + σ²        (bias² + variance + noise)

The Legendre polynomials P₀ … P_d are orthogonal on [−1, 1], so the normal equations
stay well conditioned up to degree 15, where the monomials 1, x, …, x¹⁵ would not.
"""
import torch

MAX_DEGREE = 15
JITTER = 1e-10
DT = torch.float64


def true_curve(x):
    return 0.6 * torch.sin(2.6 * x) + 0.25 * x


def make_regression(n, noise):
    x = torch.sort(torch.rand(n, dtype=DT) * 2 - 1).values
    return x, true_curve(x) + noise * torch.randn(n, dtype=DT)


def legendre(x, degree):
    """(n, degree + 1) design matrix of P₀(x) … P_degree(x), by Bonnet's recurrence."""
    cols = [torch.ones_like(x), x]
    for j in range(1, degree):
        cols.append(((2 * j + 1) * x * cols[j] - j * cols[j - 1]) / (j + 1))
    return torch.stack(cols[: degree + 1], dim=1)


def fit(x, y, degree, lam=0.0):
    phi = legendre(x, degree)
    penalty = torch.full((degree + 1,), lam, dtype=DT)
    penalty[0] = 0
    A = phi.T @ phi / len(x) + torch.diag(penalty + JITTER)
    return torch.linalg.solve(A, phi.T @ y / len(x))


def predict(w, x):
    return legendre(x, len(w) - 1) @ w


def mse(w, x, y):
    return ((predict(w, x) - y) ** 2).mean().item()


def k_folds(n, k):
    """fold[i] = the fold of point i: shuffle, then deal the points out in turn."""
    fold = torch.empty(n, dtype=torch.long)
    fold[torch.randperm(n)] = torch.arange(n) % k
    return fold


def cross_validate(x, y, degree, fold, k, lam=0.0):
    errors = [mse(fit(x[fold != f], y[fold != f], degree, lam), x[fold == f], y[fold == f]) for f in range(k)]
    return sum(errors) / k


def bias_variance(degree, n, noise, lam=0.0, datasets=100, test_points=400):
    # Expectations over x ~ U[−1, 1] as averages over equal strips (stratified sampling).
    grid = -1 + 2 * (torch.arange(200, dtype=DT) + 0.5) / 200
    preds, measured = [], 0.0
    for _ in range(datasets):
        w = fit(*make_regression(n, noise), degree, lam)
        preds.append(predict(w, grid))
        xt = -1 + 2 * (torch.arange(test_points, dtype=DT) + torch.rand(test_points, dtype=DT)) / test_points
        yt = true_curve(xt) + noise * torch.randn(test_points, dtype=DT)
        measured += mse(w, xt, yt) / datasets
    preds = torch.stack(preds)                                   # (datasets, grid)
    mean = preds.mean(0)
    bias2 = ((mean - true_curve(grid)) ** 2).mean().item()
    variance = preds.var(0, unbiased=False).mean().item()
    return bias2, variance, noise ** 2, measured


def _self_check():
    torch.manual_seed(0)
    # Training error never rises with the degree (the models are nested).
    x, y = make_regression(60, 0.3)
    train = [mse(fit(x, y, d), x, y) for d in range(MAX_DEGREE + 1)]
    assert all(b <= a * (1 + 1e-6) + 1e-12 for a, b in zip(train, train[1:])), train

    # Folds partition the points into near-equal parts.
    fold = k_folds(31, 5)
    sizes = torch.bincount(fold, minlength=5)
    assert sizes.sum() == 31 and sizes.max() - sizes.min() <= 1

    # Cross-validation picks a moderate degree: the median over five datasets lies in 3…9, never 0 or 15.
    chosen = []
    for _ in range(5):
        x, y = make_regression(80, 0.25)
        fold = k_folds(80, 5)
        cv = [cross_validate(x, y, d, fold, 5) for d in range(MAX_DEGREE + 1)]
        chosen.append(min(range(len(cv)), key=cv.__getitem__))
    median = sorted(chosen)[2]
    assert 3 <= median <= 9 and 0 not in chosen and MAX_DEGREE not in chosen, chosen

    # bias² + variance + noise matches the Monte-Carlo expected test error.
    gaps, parts = [], {}
    for degree in (1, 3, 6):
        b2, var, noise2, measured = parts[degree] = bias_variance(degree, 40, 0.3)
        gaps.append(abs(b2 + var + noise2 - measured) / measured)
        assert gaps[-1] < 0.06, (degree, b2, var, noise2, measured)
    # More degree: less bias, more variance.
    assert parts[1][0] > 5 * parts[3][0] and parts[3][1] > parts[1][1] and parts[6][1] > parts[3][1], parts
    print(f"selection.py: ok (CV degrees {chosen}; decomposition within {100 * max(gaps):.1f}% of the measured error)")


if __name__ == "__main__":
    _self_check()
