"""
Batch normalisation in PyTorch: the formula of batchnorm.ts, checked against
nn.BatchNorm1d, and the paper's experiment: a sigmoid MLP trained with and without it.

    x̂ = (x − μ_B) / √(σ²_B + ε),   y = γ·x̂ + β

In training mode μ_B and σ²_B come from the batch (biased variance), and running averages
are updated for evaluation (PyTorch: running ← (1 − m)·running + m·batch, m = 0.1, with the
unbiased variance; TensorFlow.js keeps 0.99·running + 0.01·batch). The synthetic data
stands in for MNIST so the self-check needs no download.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(0)
    var = x.var(0, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def mlp(batch_norm_layers: bool, d_in=32, hidden=(64, 64, 64, 64), classes=10, seed=0):
    torch.manual_seed(seed)
    layers, width = [], d_in
    for h in hidden:
        layers.append(nn.Linear(width, h, bias=not batch_norm_layers))
        if batch_norm_layers:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.Sigmoid())
        width = h
    layers.append(nn.Linear(width, classes))
    return nn.Sequential(*layers)


def synthetic(n, d=32, classes=10, seed=1):
    """Gaussian clusters around random class centres."""
    g = torch.Generator().manual_seed(seed)
    centres = 2 * torch.randn(classes, d, generator=g)
    labels = torch.randint(classes, (n,), generator=g)
    return centres[labels] + torch.randn(n, d, generator=g), labels


def train(model, x, y, lr=0.5, steps=300, batch=64, seed=2):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        idx = torch.randint(len(x), (batch,), generator=g)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def _self_check():
    g = torch.Generator().manual_seed(0)
    x = 5 * torch.randn(32, 6, generator=g, dtype=torch.float64) + 3
    gamma, beta = torch.rand(6, generator=g, dtype=torch.float64), torch.randn(6, generator=g, dtype=torch.float64)
    # The formula is exactly nn.BatchNorm1d in training mode…
    layer = nn.BatchNorm1d(6).double()
    with torch.no_grad():
        layer.weight.copy_(gamma)
        layer.bias.copy_(beta)
    assert torch.allclose(layer(x), batch_norm(x, gamma, beta), atol=1e-10)
    # …which normalises each unit before γ and β.
    plain = batch_norm(x, torch.ones(6, dtype=torch.float64), torch.zeros(6, dtype=torch.float64))
    assert torch.allclose(plain.mean(0), torch.zeros(6, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(plain.var(0, unbiased=False), torch.ones(6, dtype=torch.float64), atol=1e-4)
    # In eval mode it uses the running statistics: m = 0.1, unbiased variance.
    assert torch.allclose(layer.running_mean, 0.1 * x.mean(0), atol=1e-10)
    assert torch.allclose(layer.running_var, 0.9 + 0.1 * x.var(0, unbiased=True), atol=1e-10)

    # A deep sigmoid network learns much faster with batch normalisation.
    data, labels = synthetic(4000)
    without = train(mlp(False), data, labels)
    with_bn = train(mlp(True), data, labels)
    late = lambda losses: sum(losses[-50:]) / 50
    assert late(with_bn) < 0.7 * late(without), (late(with_bn), late(without))
    print(f"batchnorm.py: ok (loss after 300 steps: {late(without):.3f} without, {late(with_bn):.3f} with batch norm)")


if __name__ == "__main__":
    _self_check()
