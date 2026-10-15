"""
Mathematical morphology in PyTorch — the same operations as morphology.ts.

With a square structuring element, dilation is exactly max-pooling with stride 1, and
erosion is min-pooling, i.e. −maxpool(−x). Padding with −∞ (dilation) or +∞ (erosion)
means pixels outside the image are ignored, as in the TypeScript version.
Images are float tensors (H, W), values 0–255.
"""
import torch
import torch.nn.functional as F


def dilate(image: torch.Tensor, size: int) -> torch.Tensor:
    """Local maximum over a size×size square: bright regions grow."""
    half = size // 2
    x = F.pad(image[None, None], (half, half, half, half), value=float("-inf"))
    return F.max_pool2d(x, size, stride=1)[0, 0]


def erode(image: torch.Tensor, size: int) -> torch.Tensor:
    """Local minimum over a size×size square: bright regions shrink."""
    return -dilate(-image, size)


def morphology(image: torch.Tensor, op: str, size: int) -> torch.Tensor:
    if op == "erode":
        return erode(image, size)
    if op == "dilate":
        return dilate(image, size)
    if op == "open":    # erode then dilate: removes bright specks smaller than the kernel
        return dilate(erode(image, size), size)
    if op == "close":   # dilate then erode: fills dark holes smaller than the kernel
        return erode(dilate(image, size), size)
    raise ValueError(op)


def binarize(image: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.where(image > threshold, 255.0, 0.0)


def _self_check():
    img = torch.zeros(20, 20)
    img[5:15, 5:15] = 255          # a 10×10 square…
    img[2, 2] = 255                # …a bright speck…
    img[9, 9] = 0                  # …and a dark hole inside the square
    assert erode(img, 3).sum() == 255 * 8 * 8 - 255 * 9                # square shrinks by 1 on each side; the hole grows to 3×3
    assert morphology(img, "open", 3)[2, 2] == 0                       # opening removes the speck
    assert morphology(img, "close", 3)[9, 9] == 255                    # closing fills the hole
    assert dilate(img, 3)[1, 1] == 255 and dilate(img, 3)[0, 0] == 0    # the speck grows to 3×3
    print("morphology.py: ok")


if __name__ == "__main__":
    _self_check()
