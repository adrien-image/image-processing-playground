"""
Optimizers on 2-D loss surfaces in PyTorch: the update rules of optimizers.ts written
out by hand, checked against torch.optim step for step.

    SGD       p ← p − η·g
    Momentum  b ← μ·b + g,                p ← p − η·b
    Nesterov  b ← μ·b + g,                p ← p − η·(g + μ·b)
    RMSprop   v ← α·v + (1 − α)·g²,       p ← p − η·g / (√v + ε)
    Adam      m ← β₁·m + (1 − β₁)·g,  v ← β₂·v + (1 − β₂)·g²,  p ← p − η·m̂ / (√v̂ + ε)

The gradient comes from autograd on the surface's formula.
"""
import torch

SURFACES = {
    "ravine": (lambda p: 0.1 * p[0] ** 2 + 2 * p[1] ** 2, (-4.5, 1.6)),
    "rosenbrock": (lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2, (-1.5, 2.5)),
    "saddle": (lambda p: p[0] ** 4 / 4 - p[0] ** 2 / 2 + p[1] ** 2 / 2 + 0.25, (0.001, 1.8)),
}


def manual_run(name, f, start, lr, steps, momentum=0.9, alpha=0.99, betas=(0.9, 0.999), eps=1e-8):
    p = torch.tensor(start, dtype=torch.float64)
    b, m, v = torch.zeros(2, dtype=p.dtype), torch.zeros(2, dtype=p.dtype), torch.zeros(2, dtype=p.dtype)
    path = [p.clone()]
    for t in range(1, steps + 1):
        q = p.clone().requires_grad_(True)
        f(q).backward()
        g = q.grad
        if name == "sgd":
            p = p - lr * g
        elif name in ("momentum", "nesterov"):
            b = g.clone() if t == 1 else momentum * b + g
            p = p - lr * (g + momentum * b if name == "nesterov" else b)
        elif name == "rmsprop":
            v = alpha * v + (1 - alpha) * g ** 2
            p = p - lr * g / (v.sqrt() + eps)
        elif name == "adam":
            m = betas[0] * m + (1 - betas[0]) * g
            v = betas[1] * v + (1 - betas[1]) * g ** 2
            m_hat, v_hat = m / (1 - betas[0] ** t), v / (1 - betas[1] ** t)
            p = p - lr * m_hat / (v_hat.sqrt() + eps)
        path.append(p.clone())
    return torch.stack(path)


def torch_run(name, f, start, lr, steps):
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    make = {
        "sgd": lambda: torch.optim.SGD([p], lr=lr),
        "momentum": lambda: torch.optim.SGD([p], lr=lr, momentum=0.9),
        "nesterov": lambda: torch.optim.SGD([p], lr=lr, momentum=0.9, nesterov=True),
        "rmsprop": lambda: torch.optim.RMSprop([p], lr=lr, alpha=0.99, eps=1e-8),
        "adam": lambda: torch.optim.Adam([p], lr=lr, betas=(0.9, 0.999), eps=1e-8),
    }
    opt = make[name]()
    path = [p.detach().clone()]
    for _ in range(steps):
        opt.zero_grad()
        f(p).backward()
        opt.step()
        path.append(p.detach().clone())
    return torch.stack(path)


def _self_check():
    f, start = SURFACES["rosenbrock"]
    rates = {"sgd": 0.0015, "momentum": 0.0003, "nesterov": 0.0003, "rmsprop": 0.005, "adam": 0.05}
    # The hand-written rules are exactly torch.optim's.
    for name, lr in rates.items():
        assert torch.allclose(manual_run(name, f, start, lr, 200), torch_run(name, f, start, lr, 200), atol=1e-10), name
    # Momentum crosses the ravine far faster than SGD at the same learning rate…
    ravine, rstart = SURFACES["ravine"]
    loss = lambda path: ravine(path[-1]).item()
    assert loss(manual_run("momentum", ravine, rstart, 0.05, 100)) < loss(manual_run("sgd", ravine, rstart, 0.05, 100)) / 10
    # …and SGD diverges beyond η = 2 / curvature (= 0.5 here) where Adam does not.
    assert loss(manual_run("sgd", ravine, rstart, 0.55, 300)) > 1e6
    assert loss(manual_run("adam", ravine, rstart, 0.55, 300)) < 1
    adam_end = manual_run("adam", f, start, 0.05, 3000)[-1]
    print(f"optimizers.py: ok (Adam ends at ({adam_end[0]:.3f}, {adam_end[1]:.3f}) on Rosenbrock)")


if __name__ == "__main__":
    _self_check()
