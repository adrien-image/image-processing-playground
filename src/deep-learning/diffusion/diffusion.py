"""
Denoising diffusion (DDPM) in PyTorch — diffusion.ts and denoiser.ts.

Forward process (fixed): x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε,  ε ~ N(0, I),  ᾱ_t = Π (1 − β_s).
Reverse process (learned): an MLP ε_θ(x_t, t) is trained to predict ε (plain MSE); sampling
starts from noise and applies
    x_{t−1} = (x_t − β_t / √(1 − ᾱ_t) · ε_θ(x_t, t)) / √α_t + √β_t · z
T times. The data here are 2-D points (a spiral), so each sample is a point in the plane.
"""
import math
import torch
import torch.nn as nn


def linear_schedule(steps: int, start=1e-4, end=0.02) -> torch.Tensor:
    return torch.linspace(start, end, steps, dtype=torch.float64)


def cosine_schedule(steps: int, s=0.008) -> torch.Tensor:
    """Nichol & Dhariwal (2021): ᾱ follows a squared cosine, so information is destroyed more evenly."""
    t = torch.arange(steps + 1, dtype=torch.float64)
    f = torch.cos(((t / steps + s) / (1 + s)) * math.pi / 2) ** 2
    return (1 - f[1:] / f[:-1]).clamp(max=0.999)


def alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    return torch.cumprod(1 - betas, 0)


def q_sample(x0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    ab = alpha_bars(betas)[t].view(-1, *([1] * (x0.dim() - 1)))
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def time_embedding(t: torch.Tensor, steps: int, dims=16) -> torch.Tensor:
    """Sinusoidal embedding of the step number, so the network knows how noisy its input is."""
    half = dims // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, dtype=torch.float64) / half)
    x = (t.double() / steps * 1000)[:, None] * freqs[None]
    return torch.cat([torch.sin(x), torch.cos(x)], 1)


class Denoiser(nn.Module):
    def __init__(self, steps: int, dims=16, width=128):
        super().__init__()
        self.steps, self.dims = steps, dims
        self.net = nn.Sequential(nn.Linear(2 + dims, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2)).double()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, time_embedding(t, self.steps, self.dims)], 1))


def reverse_step(xt, eps, t: int, betas, generator=None):
    ab = alpha_bars(betas)
    mean = (xt - betas[t] / (1 - ab[t]).sqrt() * eps) / (1 - betas[t]).sqrt()
    if t == 0:
        return mean
    return mean + betas[t].sqrt() * torch.randn(xt.shape, generator=generator, dtype=xt.dtype)


@torch.no_grad()
def sample(model: Denoiser, betas, count: int, generator=None) -> torch.Tensor:
    x = torch.randn(count, 2, generator=generator, dtype=torch.float64)
    for t in reversed(range(len(betas))):
        x = reverse_step(x, model(x, torch.full((count,), t)), t, betas, generator)
    return x


def spiral(count: int, generator=None) -> torch.Tensor:
    t = torch.rand(count, generator=generator, dtype=torch.float64).sqrt() * 3 * math.pi
    pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], 1) / 6.5
    return pts + 0.03 * torch.randn(count, 2, generator=generator, dtype=torch.float64)


def train(model: Denoiser, data: torch.Tensor, betas, steps=3000, batch=256, lr=2e-3, generator=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x0 = data[torch.randint(len(data), (batch,), generator=generator)]
        t = torch.randint(len(betas), (batch,), generator=generator)
        noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
        loss = ((model(q_sample(x0, t, betas, noise), t) - noise) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def _self_check():
    g = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    for betas in (linear_schedule(1000), cosine_schedule(1000)):
        ab = alpha_bars(betas)
        assert ab[0] > 0.99 and ab[-1] < 1e-3 and (ab[1:] < ab[:-1]).all()
    # With the exact noise predictor for a one-point dataset, the reverse process lands on that point.
    betas = linear_schedule(200)
    target = torch.tensor([[0.7, -1.2]], dtype=torch.float64)
    x = torch.randn(1, 2, generator=g, dtype=torch.float64)
    ab = alpha_bars(betas)
    for t in reversed(range(200)):
        x = reverse_step(x, (x - ab[t].sqrt() * target) / (1 - ab[t]).sqrt(), t, betas, g)
    assert torch.allclose(x, target, atol=0.05), x
    # A trained denoiser turns noise into points near the spiral.
    betas = cosine_schedule(100)
    data = spiral(2000, g)
    model = Denoiser(100)
    losses = train(model, data, betas, steps=1500, generator=g)
    points = sample(model, betas, 500, g)
    nearest = torch.cdist(points, data).min(1).values
    assert sum(losses[-100:]) / 100 < 0.6 and nearest.median() < 0.15, (sum(losses[-100:]) / 100, nearest.median())
    print(f"diffusion.py: ok (median distance of samples to the data: {nearest.median():.3f})")


if __name__ == "__main__":
    _self_check()
