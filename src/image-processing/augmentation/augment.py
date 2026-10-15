"""
Data augmentation in PyTorch — the transforms of augment.ts on image tensors.

Geometric transforms (rotate, shear, crop-and-resize, flip) are one affine matrix each,
applied with affine_grid + grid_sample (the same machinery spatial transformers use).
Photometric ones (brightness, contrast, noise) are elementwise. `random_augment` draws a
random chain of them, as the training pipelines of image classifiers do.
Images are float tensors (C, H, W), values 0–255; uncovered areas are black.
"""
import math
import torch
import torch.nn.functional as F


def _affine(image: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a 2×3 matrix in normalised coordinates ([-1, 1], output → input), zeros outside."""
    grid = F.affine_grid(matrix[None].float(), [1, *image.shape], align_corners=False)
    return F.grid_sample(image[None].float(), grid, align_corners=False, padding_mode="zeros")[0]


def rotate(image: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate about the centre (positive = clockwise on screen, as on a canvas)."""
    a = math.radians(degrees)
    _, h, w = image.shape
    # Normalised coordinates stretch x and y differently on non-square images: correct for it.
    m = torch.tensor([[math.cos(a), math.sin(a) * h / w, 0.], [-math.sin(a) * w / h, math.cos(a), 0.]])
    return _affine(image, m)


def flip(image: torch.Tensor, direction: str) -> torch.Tensor:
    return image.flip(-1 if direction == "horizontal" else -2)


def crop(image: torch.Tensor, scale: float, fx: float, fy: float) -> torch.Tensor:
    """Zoom in on a `scale`-sized window whose position is (fx, fy) ∈ [0, 1]², resized back to full size."""
    tx, ty = (2 * fx - 1) * (1 - scale), (2 * fy - 1) * (1 - scale)
    return _affine(image, torch.tensor([[scale, 0., tx], [0., scale, ty]]))


def shear(image: torch.Tensor, factor: float) -> torch.Tensor:
    return _affine(image, torch.tensor([[1., -factor, 0.], [0., 1., 0.]]))


def brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    return (image * factor).clamp(0, 255)


def contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    return (128 + (image - 128) * factor).clamp(0, 255)


def gaussian_noise(image: torch.Tensor, std: float, generator=None) -> torch.Tensor:
    return (image + std * torch.randn(image.shape, generator=generator)).clamp(0, 255)


def salt_pepper(image: torch.Tensor, density: float, generator=None) -> torch.Tensor:
    r = torch.rand(image.shape[-2:], generator=generator)
    out = image.clone()
    out[:, r < density / 2] = 0
    out[:, (r >= density / 2) & (r < density)] = 255
    return out


def random_augment(image: torch.Tensor, generator=None) -> torch.Tensor:
    """A random chain: each transform is applied with probability ½, with random parameters."""
    u = lambda lo, hi: lo + (hi - lo) * torch.rand((), generator=generator).item()  # noqa: E731
    steps = [
        lambda x: rotate(x, u(-25, 25)),
        lambda x: flip(x, "horizontal"),
        lambda x: crop(x, u(0.7, 0.95), u(0, 1), u(0, 1)),
        lambda x: shear(x, u(-0.3, 0.3)),
        lambda x: brightness(x, u(0.7, 1.3)),
        lambda x: contrast(x, u(0.7, 1.4)),
        lambda x: gaussian_noise(x, u(5, 20), generator),
    ]
    for step in steps:
        if torch.rand((), generator=generator) < 0.5:
            image = step(image)
    return image


def _self_check():
    g = torch.Generator().manual_seed(0)
    img = torch.rand(3, 32, 48, generator=g) * 255
    assert torch.allclose(rotate(img, 0), img, atol=1e-3)
    assert torch.allclose(rotate(rotate(img, 90), -90)[:, 8:-8, 12:-12], img[:, 8:-8, 12:-12], atol=1e-2)  # centre survives
    assert torch.allclose(crop(img, 1.0, 0.5, 0.5), img, atol=1e-3)
    assert torch.equal(flip(flip(img, "horizontal"), "horizontal"), img)
    out = random_augment(img, g)
    assert out.shape == img.shape and out.min() >= 0 and out.max() <= 255
    print("augment.py: ok")


if __name__ == "__main__":
    _self_check()
