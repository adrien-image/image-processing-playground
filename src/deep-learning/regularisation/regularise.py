"""
Overfitting and regularisation in PyTorch — regularise.ts.

  dropout        nn.Dropout after each hidden layer (active in model.train(), off in model.eval())
  L2             adds λ·Σw² to the loss for the weight matrices (not the biases), like the Keras regulariser
  augmentation   a fresh random shift (±2 px) and rotation (±12°) of every image each epoch
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(dropout: float = 0.0) -> nn.Module:
    layers = [nn.Flatten()]
    for fan_in in (784, 256):
        layers += [nn.Linear(fan_in, 256), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(256, 10))
    return nn.Sequential(*layers)


def augment(images: torch.Tensor, shift=2.0, rotate=12.0, generator=None) -> torch.Tensor:
    """Random shift and rotation per image (bilinear, zeros outside). images: (N, 1, 28, 28)."""
    n = len(images)
    angle = (torch.rand(n, generator=generator) * 2 - 1) * math.radians(rotate)
    t = (torch.rand(n, 2, generator=generator) * 2 - 1) * shift * 2 / 28    # pixels → normalised units
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.stack([torch.stack([cos, sin, -t[:, 0]], 1), torch.stack([-sin, cos, -t[:, 1]], 1)], 1)
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False)


@torch.no_grad()
def measure(model, images, labels):
    model.eval()
    logits = model(images)
    return F.cross_entropy(logits, labels).item(), (logits.argmax(1) == labels).float().mean().item()


def train_run(model, train_x, train_y, val_x, val_y, epochs=30, l2=0.0, use_augmentation=False, seed=0):
    """Returns per-epoch (train loss, train acc, val loss, val acc), measured on clean images."""
    g = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for _ in range(epochs):
        model.train()
        x = augment(train_x, generator=g) if use_augmentation else train_x
        order = torch.randperm(len(x), generator=g)
        for start in range(0, len(x), 64):
            idx = order[start:start + 64]
            loss = F.cross_entropy(model(x[idx]), train_y[idx])
            if l2 > 0:
                loss = loss + l2 * sum((m.weight ** 2).sum() for m in model if isinstance(m, nn.Linear))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        history.append((*measure(model, train_x, train_y), *measure(model, val_x, val_y)))
    return history


def _self_check():
    torch.manual_seed(0)
    # Random "images" with random labels: there is nothing to learn, only to memorise.
    train_x, train_y = torch.rand(100, 1, 28, 28), torch.randint(0, 10, (100,))
    val_x, val_y = torch.rand(100, 1, 28, 28), torch.randint(0, 10, (100,))
    history = train_run(build_mlp(), train_x, train_y, val_x, val_y, epochs=40)
    train_acc, val_acc = history[-1][1], history[-1][3]
    assert train_acc == 1.0 and val_acc < 0.35          # a perfect memoriser that generalises at chance
    assert sum(p.numel() for p in build_mlp(0.5).parameters()) == 784 * 256 + 256 + 256 * 256 + 256 + 256 * 10 + 10
    img = torch.zeros(1, 1, 28, 28)
    img[0, 0, 14, 14] = 1
    assert torch.allclose(augment(img, 0, 0), img, atol=1e-6)
    print(f"regularise.py: ok (train {train_acc:.2f} vs validation {val_acc:.2f})")


if __name__ == "__main__":
    _self_check()
