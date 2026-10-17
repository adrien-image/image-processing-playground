"""
Gradient boosting for 1-D regression with the squared loss in PyTorch, the algorithm of
boosting.ts:

    F₀(x) = mean(y)
    r_i   = y_i − F_{m−1}(x_i)       the residuals = −∂/∂F ½(y − F)², the negative gradient
    h_m   = a small regression tree fitted to (x_i, r_i)
    F_m   = F_{m−1} + η · h_m       η: the learning rate (shrinkage)

The regression tree is CART with the squared error: at each node, sort the points by x,
and running sums of r and r² give the error of every possible split at once.
"""
import math

import torch


def fit_tree(x, r, depth, min_leaf=1):
    """A regression tree on 1-D inputs as nested tuples: ("leaf", value) or ("split", t, left, right)."""
    n = len(r)
    if depth == 0 or n < 2 * min_leaf or r.var(unbiased=False) < 1e-14:
        return ("leaf", r.mean().item())
    xs, order = torch.sort(x, stable=True)
    rs = r[order]
    cs, cs2 = rs.cumsum(0)[:-1], (rs * rs).cumsum(0)[:-1]
    nl = torch.arange(1, n, dtype=r.dtype)
    # Sum of squared errors of both children: Σr² − (Σr)²/n on each side.
    sse = cs2 - cs * cs / nl + ((rs * rs).sum() - cs2) - (rs.sum() - cs) ** 2 / (n - nl)
    valid = (xs[1:] > xs[:-1]) & (nl >= min_leaf) & (n - nl >= min_leaf)
    if not valid.any():
        return ("leaf", r.mean().item())
    i = int(torch.where(valid, sse, torch.full_like(sse, math.inf)).argmin())
    t = (xs[i] + xs[i + 1]).item() / 2
    left = x <= t
    return ("split", t, fit_tree(x[left], r[left], depth - 1, min_leaf), fit_tree(x[~left], r[~left], depth - 1, min_leaf))


def predict_tree(tree, x):
    if tree[0] == "leaf":
        return torch.full_like(x, tree[1])
    _, t, left, right = tree
    return torch.where(x <= t, predict_tree(left, x), predict_tree(right, x))


def gradient_boost(x, y, x_test, y_test, rounds, depth, rate, min_leaf=1):
    """Boost from F₀ = mean(y); returns F₀, the trees and the train / test MSE after every round."""
    f0 = y.mean().item()
    F, F_test = torch.full_like(y, f0), torch.full_like(y_test, f0)
    trees, train_mse, test_mse = [], [((y - F) ** 2).mean().item()], [((y_test - F_test) ** 2).mean().item()]
    for _ in range(rounds):
        tree = fit_tree(x, y - F, depth, min_leaf)  # the residuals are the negative gradient
        F = F + rate * predict_tree(tree, x)
        F_test = F_test + rate * predict_tree(tree, x_test)
        trees.append(tree)
        train_mse.append(((y - F) ** 2).mean().item())
        test_mse.append(((y_test - F_test) ** 2).mean().item())
    return f0, trees, train_mse, test_mse


def make_regression(n, noise):
    """The lesson's data (shared/datasets.ts): a sine with a slow trend plus Gaussian noise."""
    x = torch.rand(n, dtype=torch.float64) * 2 - 1
    return x, torch.sin(2.6 * x) * 0.6 + 0.25 * x + noise * torch.randn(n, dtype=torch.float64)


def _self_check():
    torch.manual_seed(0)
    x, y = make_regression(80, 0.3)
    x_test, y_test = make_regression(400, 0.3)
    # F₀ is the mean, the best constant for the squared loss.
    f0, _, train_mse, _ = gradient_boost(x, y, x_test, y_test, 0, 2, 0.1)
    assert abs(f0 - y.mean().item()) < 1e-12 and abs(train_mse[0] - y.var(unbiased=False).item()) < 1e-12
    # With η = 1 and deep trees, the training set is interpolated within a few rounds.
    _, _, deep_train, _ = gradient_boost(x, y, x_test, y_test, 3, 12, 1.0)
    assert deep_train[-1] < 1e-20, deep_train
    # With a small learning rate, no round ever increases the training error.
    _, _, train_mse, test_slow = gradient_boost(x, y, x_test, y_test, 300, 2, 0.05)
    assert all(b <= a + 1e-12 for a, b in zip(train_mse, train_mse[1:])) and train_mse[-1] < 0.5 * train_mse[0]
    # High variance (depth 3, η = 0.3): the test error dips, then rises as the trees fit the noise.
    _, _, train_fast, test_fast = gradient_boost(x, y, x_test, y_test, 300, 3, 0.3)
    best = min(test_fast)
    at = test_fast.index(best)
    assert train_fast[-1] < 1e-6 and at < 60 and test_fast[-1] > 1.2 * best, (at, best, test_fast[-1])
    # The small learning rate ends far better than the fast one.
    assert test_slow[-1] < test_fast[-1]
    print(f"boosting.py: ok (depth 3, η 0.3: test MSE {test_fast[0]:.3f} → {best:.3f} at round {at} → {test_fast[-1]:.3f}; "
          f"depth 2, η 0.05: {test_slow[-1]:.3f} after 300)")


if __name__ == "__main__":
    _self_check()
