"""
A GAN on 2-D points in PyTorch: the same networks and losses as gan.ts.

    D minimises  −log D(x) − log(1 − D(G(z)))
    G minimises  −log D(G(z))          (non-saturating)   or   log(1 − D(G(z)))   (minimax)

Both are written with F.binary_cross_entropy_with_logits on D's logit. The self-check
shows with autograd why the non-saturating loss is used: where D confidently rejects a
generated point, the minimax loss has almost no gradient.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_out, activation):
    layers, width = [], d_in
    for _ in range(3):
        layers += [nn.Linear(width, 64), activation()]
        width = 64
    return nn.Sequential(*layers, nn.Linear(64, d_out))


def mixture(n, generator, spread=0.06):
    k = torch.randint(8, (n,), generator=generator).double() * math.pi / 4
    centres = 1.2 * torch.stack([torch.cos(k), torch.sin(k)], 1)
    return (centres + spread * torch.randn(n, 2, generator=generator, dtype=torch.float64)).float()


def train_step(G, D, opt_g, opt_d, real, generator, loss="non-saturating", d_steps=1):
    n = len(real)
    for _ in range(d_steps):
        fake = G(torch.randn(n, 2, generator=generator)).detach()
        real_logits, fake_logits = D(real), D(fake)
        d_loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                  + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
    logits = D(G(torch.randn(n, 2, generator=generator)))
    if loss == "non-saturating":
        g_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))       # −log D(G(z))
    else:
        g_loss = -F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))     # log(1 − D(G(z)))
    opt_g.zero_grad()
    g_loss.backward()
    opt_g.step()
    return d_loss.item(), g_loss.item()


def _self_check():
    # Where D rejects a generated point (logit −5, D ≈ 0.007): the minimax loss is flat there.
    logit = torch.tensor([-5.0], requires_grad=True)
    minimax = -F.binary_cross_entropy_with_logits(logit, torch.zeros(1))
    (g_minimax,) = torch.autograd.grad(minimax, logit)
    logit2 = torch.tensor([-5.0], requires_grad=True)
    (g_nonsat,) = torch.autograd.grad(F.binary_cross_entropy_with_logits(logit2, torch.ones(1)), logit2)
    assert abs(g_minimax) < 0.01 and abs(g_nonsat) > 0.99

    # The generator moves to a target blob at (1, −1).
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(3)
    G, D = mlp(2, 2, nn.ReLU), mlp(2, 1, lambda: nn.LeakyReLU(0.2))
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-3, betas=(0.5, 0.999))
    target = torch.tensor([1.0, -1.0])
    distance = lambda: (G(torch.randn(500, 2, generator=g)).mean(0) - target).norm().item()
    start = distance()
    for _ in range(300):
        train_step(G, D, opt_g, opt_d, target + 0.05 * torch.randn(128, 2, generator=g), g)
    end = distance()
    assert end < start / 2, (start, end)
    print(f"gan.py: ok (minimax gradient {g_minimax.item():.4f} vs non-saturating {g_nonsat.item():.4f}; "
          f"distance to the target {start:.2f} → {end:.2f})")


if __name__ == "__main__":
    _self_check()
