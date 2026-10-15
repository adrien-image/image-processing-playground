"""
Autoencoder and variational autoencoder (VAE) with a 2-D latent space — vae.ts in PyTorch.

    encoder  x (784) → 256 (ReLU) → μ (2), log σ² (2)
    decoder  z (2)   → 256 (ReLU) → x̂ logits (784)

Plain autoencoder: z = μ and the loss is the reconstruction error.
VAE: z = μ + σ·ε (reparameterisation trick: the randomness is in ε, so gradients flow to
μ and σ), and the loss adds β·KL(N(μ, σ²) ‖ N(0, 1)), which keeps the codes near the
origin with no gaps, so any point of the plane decodes to a plausible digit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, latent=2, hidden=256):
        super().__init__()
        self.enc = nn.Linear(784, hidden)
        self.mu = nn.Linear(hidden, latent)
        self.log_var = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 784))

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc(x.flatten(1)))
        return self.mu(h), self.log_var(h).clamp(-10, 10)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Pixel values in [0, 1]."""
        return torch.sigmoid(self.dec(z))

    def loss(self, x: torch.Tensor, variational=True, beta=1.0):
        """Mean over the batch of BCE summed over pixels (+ β·KL for a VAE). Returns (total, recon, kl)."""
        mu, log_var = self.encode(x)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu) if variational else mu
        recon = F.binary_cross_entropy_with_logits(self.dec(z), x.flatten(1), reduction="none").sum(1).mean()
        kl = (-0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(1)).mean() if variational else torch.zeros(())
        return recon + beta * kl, recon, kl


def train(model: Autoencoder, images: torch.Tensor, epochs=15, batch_size=50, lr=2e-3, variational=True, beta=1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(images))
        for start in range(0, len(images), batch_size):
            total, _, _ = model.loss(images[order[start:start + batch_size]], variational, beta)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()


@torch.no_grad()
def decode_grid(model: Autoencoder, n=15, extent=3.5) -> torch.Tensor:
    """Decode an n×n grid of latent points over [−extent, extent]² into one (28n, 28n) image."""
    coords = torch.linspace(-extent, extent, n)
    zy, zx = torch.meshgrid(coords.flip(0), coords, indexing="ij")
    digits = model.decode(torch.stack([zx, zy], -1).reshape(-1, 2)).reshape(n, n, 28, 28)
    return digits.permute(0, 2, 1, 3).reshape(28 * n, 28 * n)


def _self_check():
    torch.manual_seed(0)
    # KL is zero for N(0, 1) and matches the closed form otherwise.
    mu, log_var = torch.tensor([[0., 0.], [1., 2.]]), torch.tensor([[0., 0.], [torch.log(torch.tensor(4.)), 0.]])
    kl = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(1)
    assert torch.allclose(kl, torch.tensor([0., 0.5 * (4 - torch.log(torch.tensor(4.))) + 2]))
    # Two kinds of "digit": left-half or right-half images. A VAE should separate them in 2-D.
    x = torch.zeros(200, 1, 28, 28)
    x[:100, :, :, :14] = 0.9
    x[100:, :, :, 14:] = 0.9
    model = Autoencoder()
    first = model.loss(x)[0].item()
    train(model, x, epochs=20)
    assert model.loss(x)[0].item() < first / 2
    with torch.no_grad():
        mu, _ = model.encode(x)
    gap = (mu[:100].mean(0) - mu[100:].mean(0)).norm()
    assert gap > 1.0, gap
    assert decode_grid(model, 5).shape == (140, 140)
    print("vae.py: ok")


if __name__ == "__main__":
    _self_check()
