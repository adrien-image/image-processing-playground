"""
Convolution in PyTorch — the same operation as processors/utils.ts.

`F.conv2d` slides a kernel over an image and sums the element-wise products. Like the
TypeScript version (and every deep-learning library), it computes cross-correlation:
the kernel is not flipped. Borders use replicate padding, so a flat image stays flat.

Images are float tensors of shape (H, W) with values 0–255.
"""
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
SOBEL_Y = SOBEL_X.T.contiguous()


def gaussian_kernel(radius: int, sigma: float) -> torch.Tensor:
    """(2r+1)×(2r+1) Gaussian kernel, normalised to sum to 1."""
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def sharpen_kernel(amount: float) -> torch.Tensor:
    """Centre boosted by 4·amount, 4-neighbours subtracted: the weights sum to 1."""
    return torch.tensor([[0., -amount, 0.], [-amount, 1 + 4 * amount, -amount], [0., -amount, 0.]])


def convolve(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Cross-correlate a (H, W) image with an odd, square kernel, replicate padding."""
    r = kernel.shape[-1] // 2
    x = F.pad(image[None, None], (r, r, r, r), mode="replicate")
    return F.conv2d(x, kernel[None, None]).squeeze(0).squeeze(0)


def to_output(raw: torch.Tensor, mode: str) -> torch.Tensor:
    """'clamp' for kernels summing to 1 (blur, sharpen); 'magnitude' scales |sum| so the max is 255."""
    if mode == "clamp":
        return raw.clamp(0, 255).round()
    peak = raw.abs().max()
    return (raw.abs() * (255 / peak)).round() if peak > 0 else torch.zeros_like(raw)


def step_math(image: torch.Tensor, kernel: torch.Tensor, row: int, col: int):
    """The arithmetic behind one output pixel: neighbourhood, products and their sum."""
    r = kernel.shape[-1] // 2
    padded = F.pad(image[None, None], (r, r, r, r), mode="replicate")[0, 0]
    neighbourhood = padded[row:row + 2 * r + 1, col:col + 2 * r + 1]
    products = neighbourhood * kernel
    return neighbourhood, products, products.sum()


def _self_check():
    torch.manual_seed(0)
    img = torch.rand(12, 10) * 255
    # A flat image stays flat under any kernel that sums to 1 (replicate padding: no dark border).
    flat = torch.full((8, 8), 100.)
    assert torch.allclose(convolve(flat, gaussian_kernel(2, 1.5)), flat)
    assert torch.allclose(convolve(flat, sharpen_kernel(2.0)), flat)
    # The per-pixel arithmetic agrees with the whole-image convolution.
    raw = convolve(img, SOBEL_X)
    for row, col in [(0, 0), (5, 4), (11, 9)]:
        _, _, s = step_math(img, SOBEL_X, row, col)
        assert torch.isclose(s, raw[row, col], atol=1e-3)
    # Sobel responds to a vertical edge, not inside flat areas.
    edge = torch.zeros(6, 6)
    edge[:, 3:] = 255
    gx = to_output(convolve(edge, SOBEL_X), "magnitude")
    assert gx[2, 0] == 0 and gx[2, 3] == 255
    print("convolution.py: ok")


if __name__ == "__main__":
    _self_check()
