"""
Otsu's thresholding in PyTorch — the same algorithm as otsu.ts, vectorised.

For every candidate threshold t at once, cumulative sums give the background weight
ω_B(t), the foreground weight ω_F(t) and both class means; Otsu picks the t that
maximises the between-class variance ω_B · ω_F · (μ_B − μ_F)². Pixels > t are foreground.
"""
import torch


def histogram256(image: torch.Tensor) -> torch.Tensor:
    """256-bin histogram of gray values (rounded to the nearest level)."""
    levels = image.round().clamp(0, 255).long().flatten()
    return torch.bincount(levels, minlength=256).double()


def otsu_threshold(hist: torch.Tensor) -> int:
    """The threshold maximising between-class variance; ties give the middle of the tied range."""
    levels = torch.arange(256, dtype=torch.float64)
    w_b = torch.cumsum(hist, 0)                 # pixels ≤ t
    w_f = hist.sum() - w_b                      # pixels > t
    sum_b = torch.cumsum(hist * levels, 0)
    mean_b = sum_b / w_b.clamp(min=1)
    mean_f = (sum_b[-1] - sum_b) / w_f.clamp(min=1)
    variance = w_b * w_f * (mean_b - mean_f) ** 2
    variance[(w_b == 0) | (w_f == 0)] = -1      # a split needs pixels on both sides
    best = torch.nonzero(variance == variance.max()).flatten()
    return round((best.min().item() + best.max().item()) / 2)


def apply_threshold(image: torch.Tensor, t: int) -> torch.Tensor:
    """Binary mask: 1 for foreground (value > t), 0 for background."""
    return (image.round() > t).to(torch.uint8)


def _self_check():
    torch.manual_seed(0)
    # Two clusters of gray levels: the threshold falls between them.
    img = torch.cat([torch.full((500,), 50.), torch.full((500,), 200.)]) + torch.randn(1000) * 5
    t = otsu_threshold(histogram256(img))
    assert 70 < t < 180, t
    mask = apply_threshold(img, t)
    assert mask[:500].sum() == 0 and mask[500:].sum() == 500
    # Exactly two levels with an empty gap: the cut sits in the middle of the gap.
    two = torch.tensor([10.] * 5 + [30.] * 5)
    assert otsu_threshold(histogram256(two)) == 20
    print("otsu.py: ok")


if __name__ == "__main__":
    _self_check()
