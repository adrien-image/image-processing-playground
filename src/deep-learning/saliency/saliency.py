"""
Saliency maps in PyTorch — saliency.ts with autograd and hooks.

  vanilla gradient   |∂ score(class) / ∂x|: set requires_grad on the input and back-propagate
  SmoothGrad         the same, averaged over noisy copies of the input (one batch)
  Grad-CAM           capture the last conv layer's activations A and their gradients with a
                     forward hook; weight each feature map by its mean gradient α_k and take
                     ReLU(Σ α_k A_k), then upsample to the input size

The score is the class logit (before the softmax), as in the Grad-CAM paper: the gradient
of a probability or log-probability vanishes once the model is confident (p ≈ 1).
(The TensorFlow.js model in the browser ends in a softmax, so saliency.ts uses log p.)

Works with the LeNet in classification/lenet.py (any model whose last conv layer you name).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _score(model: nn.Module, x: torch.Tensor, cls: int) -> torch.Tensor:
    """The class logit, summed over the batch (so one backward pass covers every image)."""
    return model(x)[:, cls].sum()


def vanilla_gradient(model: nn.Module, image: torch.Tensor, cls: int) -> torch.Tensor:
    """image: (28, 28) → |gradient| (28, 28)."""
    x = image[None, None].clone().requires_grad_(True)
    _score(model, x, cls).backward()
    return x.grad[0, 0].abs()


def smooth_grad(model: nn.Module, image: torch.Tensor, cls: int, samples=24, sigma=0.15, generator=None) -> torch.Tensor:
    noise = sigma * torch.randn(samples, 1, *image.shape, generator=generator)
    x = (image[None, None] + noise).requires_grad_(True)
    _score(model, x, cls).backward()
    return x.grad.abs().mean(dim=(0, 1))


def grad_cam(model: nn.Module, image: torch.Tensor, cls: int, layer: nn.Module) -> torch.Tensor:
    """Class-specific map from `layer`'s activations, bilinearly upsampled to the input size."""
    captured = {}
    hook = layer.register_forward_hook(lambda _m, _i, out: captured.__setitem__("a", out))
    try:
        x = image[None, None]
        score = _score(model, x, cls)
    finally:
        hook.remove()
    a = captured["a"]                                       # (1, K, h, w)
    grads, = torch.autograd.grad(score, a)
    alpha = grads.mean(dim=(2, 3), keepdim=True)            # one weight per feature map
    cam = F.relu((alpha * a).sum(1, keepdim=True))           # (1, 1, h, w)
    return F.interpolate(cam, size=image.shape, mode="bilinear", align_corners=False)[0, 0].detach()


def _self_check():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("lenet", pathlib.Path(__file__).parents[1] / "classification" / "lenet.py")
    lenet = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lenet)

    torch.manual_seed(0)
    model = lenet.LeNet()
    images = torch.zeros(64, 1, 28, 28)
    labels = torch.arange(64) % 2
    images[labels == 0, :, 4:12, 4:12] = 1       # class 0: square at the top-left
    images[labels == 1, :, 16:24, 16:24] = 1     # class 1: square at the bottom-right
    lenet.train(model, images, labels, epochs=15, batch_size=16, lr=3e-3)
    model.eval()
    img = images[0, 0]
    for m in [vanilla_gradient(model, img, 0), smooth_grad(model, img, 0, generator=torch.Generator().manual_seed(1)),
              grad_cam(model, img, 0, model.conv2)]:
        assert m.shape == (28, 28) and (m >= 0).all()
    cam = grad_cam(model, img, 0, model.conv2)
    assert cam[2:14, 2:14].mean() > cam[16:, 16:].mean()   # Grad-CAM looks where the square is
    print("saliency.py: ok")


if __name__ == "__main__":
    _self_check()
