"""
Linear and logistic regression in PyTorch: the same models as linear.ts.

    least squares        θ = argmin (1/n)|Aθ − y|²,   AᵀA θ = Aᵀy   (the normal equations)
    gradient descent     θ ← θ − η·(2/n)·Aᵀ(Aθ − y),   stable only for η < 2 / λmax((2/n)AᵀA)
    logistic regression  p = σ(Aθ),   L = −mean(y log p + (1 − y) log(1 − p)),   ∇L = (1/n)·Aᵀ(p − y)

A is the input matrix with a column of ones appended, θ = (w, b). The self-check solves
the normal equations by hand and compares with torch.linalg.lstsq, checks the hand-derived
logistic gradient against autograd, and shows why least squares is a poor classifier:
far-away points that are already correctly classified drag its boundary, not the logistic one.
"""
import math
import torch


def with_ones(X):
    """A = [X | 1], so that A @ θ = X @ w + b."""
    return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1)


def least_squares(X, y):
    """The exact fit θ = (w, b) from the normal equations AᵀA θ = Aᵀy."""
    A = with_ones(X)
    return torch.linalg.solve(A.T @ A, A.T @ y)


def mse(X, y, theta):
    return ((with_ones(X) @ theta - y) ** 2).mean()


def gradient_descent(X, y, lr, steps, start):
    """Full-batch gradient descent on the MSE; returns the final θ."""
    A, theta = with_ones(X), start.clone()
    for _ in range(steps):
        theta = theta - lr * 2 / len(y) * A.T @ (A @ theta - y)
    return theta


def max_stable_lr(X):
    """2 / λmax of the MSE's (constant) Hessian (2/n)·AᵀA."""
    A = with_ones(X)
    return 2 / torch.linalg.eigvalsh(2 / len(A) * A.T @ A).max().item()


def log_loss(X, y, theta, l2=0.0):
    p = torch.sigmoid(with_ones(X) @ theta)
    w = theta[:-1]
    return torch.nn.functional.binary_cross_entropy(p, y) + l2 / 2 * (w ** 2).sum()


def logistic_gradient(X, y, theta, l2=0.0):
    """The hand-derived gradient (1/n)·Aᵀ(σ(Aθ) − y), plus λw on the weights (not on b)."""
    A = with_ones(X)
    g = A.T @ (torch.sigmoid(A @ theta) - y) / len(y)
    return g + l2 * torch.cat([theta[:-1], torch.zeros(1, dtype=theta.dtype)])


def fit_logistic(X, y, lr=1.0, steps=1000, l2=1e-3):
    """Logistic regression by full-batch gradient descent, as fitLogistic in linear.ts."""
    theta = torch.zeros(X.shape[1] + 1, dtype=X.dtype)
    for _ in range(steps):
        theta = theta - lr * logistic_gradient(X, y, theta, l2)
    return theta


def two_blobs(n, separation, spread, stretch=2.0):
    """Two elongated Gaussian blobs along the direction (cos 0.5, sin 0.5), labels 0 and 1."""
    axis = torch.tensor([math.cos(0.5), math.sin(0.5)], dtype=torch.float64)
    across = torch.stack([-axis[1], axis[0]])
    labels = (torch.arange(n) % 2).double()
    along = (labels - 0.5) * separation + spread * torch.randn(n, dtype=torch.float64)
    side = stretch * spread * torch.randn(n, dtype=torch.float64)
    return along[:, None] * axis + side[:, None] * across, labels


def accuracy(scores, y, threshold):
    return ((scores >= threshold).double() == y).double().mean().item()


def angle(theta):
    return math.degrees(math.atan2(theta[1].item(), theta[0].item()))


def _self_check():
    torch.manual_seed(0)

    # Regression: a noisy line on x ∈ [0, 2].
    x = 2 * torch.rand(30, 1, dtype=torch.float64)
    y = 0.6 * x[:, 0] + 0.4 + 0.2 * torch.randn(30, dtype=torch.float64)
    exact = least_squares(x, y)
    lstsq = torch.linalg.lstsq(with_ones(x), y[:, None]).solution[:, 0]
    assert torch.allclose(exact, lstsq, atol=1e-10), (exact, lstsq)

    # Gradient descent reaches the closed form below 2/λmax, and diverges above it.
    limit = max_stable_lr(x)
    start = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    for lr in (0.3 * limit, 0.9 * limit):
        assert torch.allclose(gradient_descent(x, y, lr, 5000, start), exact, atol=1e-8), lr
    assert not mse(x, y, gradient_descent(x, y, 1.1 * limit, 2000, start)) < 1e6

    # One outlier pulls the squared-error fit: its residual counts squared.
    outlier = least_squares(torch.cat([x, torch.tensor([[2.4]], dtype=torch.float64)]), torch.cat([y, torch.tensor([-0.9], dtype=torch.float64)]))
    assert outlier[0] < exact[0] - 0.15, (exact, outlier)

    # Logistic regression: the hand-derived gradient is autograd's.
    X, labels = two_blobs(100, 1.2, 0.25)
    theta = torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64, requires_grad=True)
    log_loss(X, labels, theta, l2=1e-3).backward()
    assert torch.allclose(theta.grad, logistic_gradient(X, labels, theta.detach(), l2=1e-3), atol=1e-12)

    # Ten far points of class 1, already on the right side: least squares rotates, logistic does not.
    far = torch.tensor([1.4, 2.8], dtype=torch.float64) + 0.15 * (2 * torch.rand(10, 2, dtype=torch.float64) - 1)
    X_far, labels_far = torch.cat([X, far]), torch.cat([labels, torch.ones(10, dtype=torch.float64)])
    logit, logit_far = fit_logistic(X, labels), fit_logistic(X_far, labels_far)
    ls, ls_far = least_squares(X, labels), least_squares(X_far, labels_far)
    turn_logit, turn_ls = abs(angle(logit_far) - angle(logit)), abs(angle(ls_far) - angle(ls))
    assert turn_logit < 2 and turn_ls > 10, (turn_logit, turn_ls)
    A = with_ones(X)
    # Judged on the original points, the logistic accuracy stays put; least squares' can only get worse.
    acc_logit, acc_logit_far = accuracy(torch.sigmoid(A @ logit), labels, 0.5), accuracy(torch.sigmoid(A @ logit_far), labels, 0.5)
    acc_ls, acc_ls_far = accuracy(A @ ls, labels, 0.5), accuracy(A @ ls_far, labels, 0.5)
    assert abs(acc_logit_far - acc_logit) <= 0.01 and acc_ls_far <= acc_ls, (acc_logit, acc_logit_far, acc_ls, acc_ls_far)

    print(f"linear.py: ok (slope {exact[0]:.2f} → {outlier[0]:.2f} with one outlier, η limit {limit:.2f}; "
          f"far points turn least squares {turn_ls:.0f}° and logistic {turn_logit:.1f}°, "
          f"least-squares accuracy {acc_ls:.2f} → {acc_ls_far:.2f})")


if __name__ == "__main__":
    _self_check()
