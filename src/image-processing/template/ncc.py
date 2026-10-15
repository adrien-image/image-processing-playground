"""
Template matching by normalised cross-correlation in PyTorch — the same score as ncc.ts.

    NCC(x, y) = Σ (I − Ī)(T − T̄) / (n · σ_I · σ_T)

computed for every placement at once: the numerator is a single convolution of the
image with the zero-mean template (Σ (T − T̄) = 0, so the image mean drops out), and
the local means and variances under the template come from box filters (avg_pool2d).
Scores are in [−1, 1]; 1 means the patch is the template up to brightness and contrast.
"""
import torch
import torch.nn.functional as F

FLAT_STD = 2.0


def ncc_map(image: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """NCC at every placement: shape (H − th + 1, W − tw + 1)."""
    th, tw = template.shape
    t = template.double() - template.double().mean()
    t_std = t.pow(2).mean().sqrt()
    if t_std < FLAT_STD:
        raise ValueError("Template is (nearly) flat: NCC is undefined")
    img = image.double()[None, None]
    num = F.conv2d(img, t[None, None])[0, 0]                    # Σ I·(T − T̄) per placement
    mean = F.avg_pool2d(img, (th, tw), stride=1)[0, 0]
    var = F.avg_pool2d(img ** 2, (th, tw), stride=1)[0, 0] - mean ** 2
    i_std = var.clamp(min=0).sqrt()
    n = th * tw
    return torch.where(i_std >= 1e-6, num / (n * t_std * i_std.clamp(min=1e-6)), torch.zeros_like(num))


def best_match(image: torch.Tensor, template: torch.Tensor):
    scores = ncc_map(image, template)
    idx = scores.argmax()
    y, x = divmod(idx.item(), scores.shape[1])
    return x, y, scores[y, x].item(), scores


def _self_check():
    g = torch.Generator().manual_seed(0)
    image = torch.rand(80, 90, generator=g) * 255
    template = image[30:50, 40:62].clone()
    # Brighter and higher-contrast copy of the patch: NCC still finds it with score 1.
    x, y, score, scores = best_match(image, 1.5 * template + 20)
    assert (x, y) == (40, 30) and abs(score - 1) < 1e-6
    assert scores.shape == (80 - 20 + 1, 90 - 22 + 1)
    assert scores.max() <= 1 + 1e-9 and scores.min() >= -1 - 1e-9
    print("ncc.py: ok")


if __name__ == "__main__":
    _self_check()
