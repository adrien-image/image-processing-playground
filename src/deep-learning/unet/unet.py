"""
A tiny U-Net for 32×32 binary segmentation — unet.ts in PyTorch, with the synthetic
dataset of shapes.ts (uneven lighting, noise and clutter lines that defeat Otsu).

    32×32×8 ─────────────────────────── skip ───────────┐
      │ pool                                            ▼
    16×16×16 ──────────── skip ──────────┐     up → cat → 32×32×8 → 1×1 conv → mask logits
      │ pool                             ▼
    8×8×32 (bottleneck) ────── up → cat → 16×16×16
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, skips: bool = True):
        super().__init__()
        self.skips = skips
        self.enc1, self.enc2, self.bottom = block(1, 8), block(8, 16), block(16, 32)
        self.dec2 = block(32 + (16 if skips else 0), 16)
        self.dec1 = block(16 + (8 if skips else 0), 8)
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 32, 32) → mask logits (N, 1, 32, 32); sigmoid gives the object probability."""
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottom(F.max_pool2d(e2, 2))
        up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")  # noqa: E731
        d2 = self.dec2(torch.cat([up(b), e2], 1) if self.skips else up(b))
        d1 = self.dec1(torch.cat([up(d2), e1], 1) if self.skips else up(d2))
        return self.head(d1)


def make_sample(size: int, g: torch.Generator, lighting=0.45, noise=0.06, clutter=3):
    """One (image, mask) pair: 1–3 discs or boxes on a lit, noisy background with thin bright lines."""
    u = lambda: torch.rand((), generator=g).item()  # noqa: E731
    ys, xs = torch.meshgrid(torch.arange(size) + 0.5, torch.arange(size) + 0.5, indexing="ij")
    angle = u() * 2 * math.pi
    image = 0.1 + 0.15 * u() + lighting * ((xs / size - 0.5) * math.cos(angle) + (ys / size - 0.5) * math.sin(angle) + 0.5)
    mask = torch.zeros(size, size, dtype=torch.bool)
    for _ in range(1 + int(u() * 3)):
        r = size * (0.1 + u() * 0.12)
        cx, cy = r + u() * (size - 2 * r), r + u() * (size - 2 * r)
        inside = ((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r) if u() < 0.5 else (((xs - cx).abs() <= 0.9 * r) & ((ys - cy).abs() <= 0.7 * r))
        image = image + (0.25 + 0.2 * u()) * (inside & ~mask)
        mask |= inside
    for _ in range(clutter):
        x0, y0, a, length = u() * size, u() * size, u() * math.pi, size * (0.4 + 0.5 * u())
        s = torch.arange(0, length, 0.5)
        px, py = (x0 + torch.cos(torch.tensor(a)) * s).long(), (y0 + torch.sin(torch.tensor(a)) * s).long()
        ok = (px >= 0) & (py >= 0) & (px < size) & (py < size)
        line = torch.zeros(size, size, dtype=torch.bool)
        line[py[ok], px[ok]] = True
        image = image + 0.35 * (line & ~mask)
    image = (image + noise * torch.randn(size, size, generator=g)).clamp(0, 1)
    return image, mask.float()


def make_dataset(count: int, size: int = 32, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    pairs = [make_sample(size, g) for _ in range(count)]
    return torch.stack([p[0] for p in pairs])[:, None], torch.stack([p[1] for p in pairs])[:, None]


def mask_iou(prob: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """IoU per image of (prob ≥ 0.5) against the true mask."""
    p, t = prob >= 0.5, truth >= 0.5
    inter = (p & t).flatten(1).sum(1).float()
    union = (p | t).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def train(model: UNet, images, masks, epochs=10, batch_size=32, lr=3e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(images))
        for start in range(0, len(images), batch_size):
            idx = order[start:start + batch_size]
            loss = F.binary_cross_entropy_with_logits(model(images[idx]), masks[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def otsu_masks(images: torch.Tensor) -> torch.Tensor:
    """Otsu's global threshold per image (the Segmentation lesson's method), as a baseline."""
    out = []
    for img in images[:, 0]:
        hist = torch.bincount((img * 255).round().long().flatten(), minlength=256).double()
        levels = torch.arange(256, dtype=torch.float64)
        wb = hist.cumsum(0)
        wf = hist.sum() - wb
        sb = (hist * levels).cumsum(0)
        var = wb * wf * (sb / wb.clamp(min=1) - (sb[-1] - sb) / wf.clamp(min=1)) ** 2
        out.append((img * 255).round() > var.argmax())
    return torch.stack(out)[:, None].float()


def _self_check():
    torch.manual_seed(0)
    train_x, train_y = make_dataset(400, seed=1)
    val_x, val_y = make_dataset(100, seed=2)
    model = UNet()
    train(model, train_x, train_y, epochs=12)
    model.eval()
    with torch.no_grad():
        unet_iou = mask_iou(torch.sigmoid(model(val_x)), val_y).mean().item()
    otsu_iou = mask_iou(otsu_masks(val_x), val_y).mean().item()
    assert UNet(skips=False)(train_x[:2]).shape == (2, 1, 32, 32)
    assert unet_iou > otsu_iou + 0.15, (unet_iou, otsu_iou)
    print(f"unet.py: ok (validation IoU: U-Net {unet_iou:.2f}, Otsu {otsu_iou:.2f})")


if __name__ == "__main__":
    _self_check()
