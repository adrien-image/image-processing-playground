"""
Learning a convolution kernel by gradient descent in PyTorch — learnKernel.ts with autograd.

The TypeScript version writes the gradient out by hand:
    ∂L/∂K[i, j] = 2/N · Σ (K⋆x − t) · x shifted by (i, j)
Here `loss.backward()` computes exactly the same numbers, and torch.optim.SGD (with
momentum) takes the steps. The hand-written gradient is kept to show they agree.
"""
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float64)


def convolve(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Cross-correlation with replicate padding, (H, W) → (H, W)."""
    r = kernel.shape[-1] // 2
    return F.conv2d(F.pad(image[None, None], (r,) * 4, mode="replicate"), kernel[None, None])[0, 0]


def manual_gradient(image: torch.Tensor, target: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """∂ mean((K⋆x − t)²) / ∂K, written out like the TypeScript version."""
    residual = convolve(image, kernel) - target
    size, r = kernel.shape[0], kernel.shape[0] // 2
    padded = F.pad(image[None, None], (r,) * 4, mode="replicate")[0, 0]
    h, w = image.shape
    grad = torch.empty_like(kernel)
    for ky in range(size):
        for kx in range(size):
            grad[ky, kx] = 2 * (residual * padded[ky:ky + h, kx:kx + w]).mean()
    return grad


def learn_kernel(image: torch.Tensor, target: torch.Tensor, steps=500, lr=0.1, momentum=0.9, size=3):
    """Fit a size×size kernel to (image, target) pairs from zeros. Returns the kernel and the losses."""
    kernel = torch.zeros(size, size, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([kernel], lr=lr, momentum=momentum)
    losses = []
    for _ in range(steps):
        loss = F.mse_loss(convolve(image, kernel), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return kernel.detach(), losses


def _self_check():
    g = torch.Generator().manual_seed(7)
    image = torch.rand(16, 12, generator=g, dtype=torch.float64)
    target = convolve(image, SOBEL_X)
    # Autograd and the hand-written gradient agree.
    k = torch.randn(3, 3, generator=g, dtype=torch.float64, requires_grad=True)
    F.mse_loss(convolve(image, k), target).backward()
    assert torch.allclose(k.grad, manual_gradient(image, target, k.detach()), atol=1e-10)
    # Gradient descent recovers the hidden kernel from random-noise input.
    learned, losses = learn_kernel(image, target, steps=600, lr=0.2, momentum=0.9)
    assert torch.allclose(learned, SOBEL_X, atol=1e-3) and losses[-1] < 1e-8
    # Too large a learning rate diverges.
    _, exploding = learn_kernel(image, target, steps=40, lr=3.0, momentum=0.0)
    assert exploding[-1] > 1e3
    print("learn_kernel.py: ok")


if __name__ == "__main__":
    _self_check()
