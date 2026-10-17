"""
A soft-margin support vector machine trained with SMO, in PyTorch: the same algorithm as svm.ts.

    primal   minimise  ½‖w‖² + C Σᵢ max(0, 1 − yᵢ f(xᵢ)),       f(x) = w·x + b
    dual     maximise  Σ αᵢ − ½ Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ, xⱼ),     0 ≤ αᵢ ≤ C,  Σ αᵢ yᵢ = 0
             f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b

SMO updates two α at a time (the pair that violates the KKT conditions most) in closed form.
The self-check verifies the geometry on two points, the KKT conditions, that the dual's
w = Σ αᵢ yᵢ xᵢ is the w a plain gradient descent on the primal finds, and that an RBF kernel
separates circles that a line cannot.
"""
import math
import torch


def kernel_matrix(a, b, kernel="linear", gamma=1.0, degree=3):
    if kernel == "rbf":
        return torch.exp(-gamma * torch.cdist(a, b) ** 2)
    dot = a @ b.T
    return dot if kernel == "linear" else (dot + 1) ** degree


def train_svm(x, y, C, tol=1e-3, max_iterations=50_000, **kernel):
    """x: n × d (float64), y: n labels ±1. Returns (alpha, b)."""
    n = len(y)
    K = kernel_matrix(x, x, **kernel)
    alpha = torch.zeros(n, dtype=x.dtype)
    u = torch.zeros(n, dtype=x.dtype)                  # Σⱼ αⱼ yⱼ K(xⱼ, xₜ): f without b
    for _ in range(max_iterations):
        g = y - u
        up = torch.where(y > 0, alpha < C, alpha > 0)  # αₜ can move to raise yₜ f(xₜ)
        low = torch.where(y > 0, alpha > 0, alpha < C)  # αₜ can move to lower it
        g_up = torch.where(up, g, -math.inf)
        i = int(g_up.argmax())
        b_min, b_max = g_up[i].item(), torch.where(low, g, math.inf).min().item()
        if b_min - b_max < tol:                        # KKT holds up to tol: done
            break
        # Partner: the violating "low" point whose pair step gains the most (second-order choice).
        eta = (K[i, i] + K.diagonal() - 2 * K[i]).clamp_min(1e-12)
        gain = torch.where(low & (g < b_min), (b_min - g) ** 2 / eta, -math.inf)
        j = int(gain.argmax())
        # The closed-form step for α_j along yᵢαᵢ + yⱼαⱼ = const, clipped to the box.
        ai, aj, yi, yj = alpha[i].item(), alpha[j].item(), y[i].item(), y[j].item()
        if yi != yj:
            lo, hi = max(0.0, aj - ai), min(C, C + aj - ai)
        else:
            lo, hi = max(0.0, ai + aj - C), min(C, ai + aj)
        ei, ej = u[i].item() - yi, u[j].item() - yj
        aj_new = min(hi, max(lo, aj + yj * (ei - ej) / eta[j].item()))
        ai_new = ai - yi * yj * (aj_new - aj)
        ai_new = 0.0 if ai_new < 1e-12 * C else C if ai_new > C * (1 - 1e-12) else ai_new
        alpha[i], alpha[j] = ai_new, aj_new
        u += (ai_new - ai) * yi * K[i] + (aj_new - aj) * yj * K[j]
    # b from the free support vectors (exactly on the margin), else the middle of [b_min, b_max].
    free = (alpha > 1e-8 * C) & (alpha < C * (1 - 1e-8))
    b = (y - u)[free].mean().item() if free.any() else (b_min + b_max) / 2
    return alpha, b


def decision(x_train, y, alpha, b, z, **kernel):
    return kernel_matrix(z, x_train, **kernel) @ (alpha * y) + b


def _blobs(n, generator, spread=0.25):
    y = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0).double()
    x = 0.5 * torch.stack([y, y], 1) + spread * torch.randn(n, 2, generator=generator, dtype=torch.float64)
    return x, y


def _circles(n, generator):
    y = torch.where(torch.arange(n) % 2 == 0, 1.0, -1.0).double()
    t = 2 * math.pi * torch.rand(n, generator=generator, dtype=torch.float64)
    r = torch.where(y > 0, 0.35, 0.85).double()
    x = torch.stack([r * t.cos(), r * t.sin()], 1) + 0.05 * torch.randn(n, 2, generator=generator, dtype=torch.float64)
    return x, y


def _self_check():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(int(torch.randint(1 << 30, (1,))))

    # Two points: the boundary is their perpendicular bisector and the margin their distance.
    x2 = torch.tensor([[0.3, -0.2], [-0.5, 0.4]], dtype=torch.float64)
    y2 = torch.tensor([1.0, -1.0], dtype=torch.float64)
    alpha, b = train_svm(x2, y2, C=100.0)
    w = (alpha * y2) @ x2
    assert abs(decision(x2, y2, alpha, b, x2.mean(0, keepdim=True)).item()) < 1e-6
    assert abs(2 / w.norm().item() - (x2[0] - x2[1]).norm().item()) < 1e-3

    # Noisy blobs: KKT conditions at the SMO solution.
    C, tol = 1.0, 2e-3
    x, y = _blobs(100, g)
    alpha, b = train_svm(x, y, C)
    margin = y * decision(x, y, alpha, b, x)
    bounded = alpha >= C * (1 - 1e-8)
    free = (alpha > 0) & ~bounded
    assert ((alpha >= 0) & (alpha <= C)).all()
    assert abs((alpha * y).sum().item()) < 1e-9
    assert (margin[alpha == 0] >= 1 - tol).all()             # outside the margin: α = 0
    assert ((margin[free] - 1).abs() < tol).all()             # on the margin: 0 < α < C
    assert (margin[bounded] <= 1 + tol).all()                 # inside it: α = C
    n_sv = int((alpha > 0).sum())

    # The primal, minimised directly with autograd, finds the same w as Σ αᵢ yᵢ xᵢ.
    w_dual = (alpha * y) @ x
    wb = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([wb], lr=0.05)
    steps = 3000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
    for _ in range(steps):
        loss = 0.5 * wb[:2].pow(2).sum() + C * torch.relu(1 - y * (x @ wb[:2] + wb[2])).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    w_error = ((wb[:2].detach() - w_dual).norm() / w_dual.norm()).item()
    assert w_error < 0.02, w_error

    # Circles: an RBF kernel separates them, a line cannot.
    x, y = _circles(160, g)
    acc = lambda **k: (torch.sign(decision(x, y, *train_svm(x, y, 10.0, **k), x, **k)) == y).double().mean().item()
    acc_rbf, acc_linear = acc(kernel="rbf", gamma=5.0), acc(kernel="linear")
    assert acc_rbf >= 0.98 and acc_linear < 0.75, (acc_rbf, acc_linear)
    print(f"svm.py: ok ({n_sv} support vectors of 100, primal w within {100 * w_error:.2f}% of Σαyx; "
          f"circles: RBF {acc_rbf:.0%} vs linear {acc_linear:.0%})")


if __name__ == "__main__":
    _self_check()
