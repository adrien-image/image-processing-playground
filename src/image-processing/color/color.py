"""
Colour spaces in PyTorch — the same conversions as color.ts, on whole images at once.

RGB ↔ HSV works per pixel with no loops: the max channel decides which sextant of the
hue wheel a pixel is in, and `torch.where` picks the matching formula.
Images are float tensors (3, H, W) with values 0–255; h, s, v are in [0, 1].
"""
import torch


def rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    r, g, b = rgb / 255
    mx, _ = torch.max(rgb / 255, dim=0)
    mn, _ = torch.min(rgb / 255, dim=0)
    d = mx - mn
    safe = torch.where(d == 0, torch.ones_like(d), d)
    h = torch.where(mx == r, ((g - b) / safe) % 6, torch.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4)) / 6
    h = torch.where(d == 0, torch.zeros_like(h), h)
    s = torch.where(mx == 0, torch.zeros_like(mx), d / torch.where(mx == 0, torch.ones_like(mx), mx))
    return torch.stack([h, s, mx])


def hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv
    i = torch.floor(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    i = i.long() % 6
    # Rows: which of (v, q, p, t) each channel takes in each of the 6 sextants.
    r = torch.stack([v, q, p, p, t, v]).gather(0, i[None])[0]
    g = torch.stack([t, v, v, q, p, p]).gather(0, i[None])[0]
    b = torch.stack([p, p, t, v, v, q]).gather(0, i[None])[0]
    return torch.stack([r, g, b]) * 255


def adjust_colors(rgb: torch.Tensor, hue_shift=0.0, sat_scale=1.0, val_scale=1.0, gamma=1.0) -> torch.Tensor:
    """Rotate hue (degrees), scale saturation and value, then apply gamma: out = in^(1/γ)."""
    h, s, v = rgb_to_hsv(rgb)
    out = hsv_to_rgb(torch.stack([(h + hue_shift / 360) % 1, (s * sat_scale).clamp(max=1), (v * val_scale).clamp(max=1)]))
    return (out / 255) ** (1 / gamma) * 255


def _self_check():
    torch.manual_seed(0)
    img = torch.rand(3, 16, 16) * 255
    assert torch.allclose(hsv_to_rgb(rgb_to_hsv(img)), img, atol=1e-3)
    red = torch.tensor([255., 0., 0.])[:, None, None]
    h, s, v = rgb_to_hsv(red).flatten()
    assert (h, s, v) == (0, 1, 1)
    # A 120° hue shift turns pure red into pure green.
    assert torch.allclose(adjust_colors(red, hue_shift=120).flatten(), torch.tensor([0., 255., 0.]), atol=1e-3)
    # Zero saturation gives gray: all channels equal.
    gray = adjust_colors(img, sat_scale=0)
    assert torch.allclose(gray[0], gray[1]) and torch.allclose(gray[1], gray[2])
    print("color.py: ok")


if __name__ == "__main__":
    _self_check()
