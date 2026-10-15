"""
Adversarial examples in PyTorch — the attacks of attacks.ts.

  FGSM      x′ = clip(x + ε · sign(∇ₓ L(x, y)), 0, 1)               one step, untargeted
  targeted  step *down* the loss of a chosen class instead
  PGD       several steps of size ε/4, projecting back into the ε-box around x each time

Images are float tensors (N, 1, 28, 28) in [0, 1]; the model returns logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float,
           target: torch.Tensor | None = None, steps: int = 1) -> torch.Tensor:
    model.eval()
    step = epsilon if steps == 1 else epsilon / 4
    towards = target is not None
    x = images.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), target if towards else labels)
        grad, = torch.autograd.grad(loss, x)
        with torch.no_grad():
            x = x + (-step if towards else step) * grad.sign()
            x = torch.min(torch.max(x, images - epsilon), images + epsilon).clamp(0, 1)
    return x.detach()


@torch.no_grad()
def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    return (model(images).argmax(1) == labels).float().mean().item()


def robustness_curve(model, images, labels, epsilons=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)):
    """Accuracy under FGSM for each ε (ε = 0 is the clean accuracy)."""
    return [accuracy(model, attack(model, images, labels, e) if e > 0 else images, labels) for e in epsilons]


def _self_check():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))
    images = torch.rand(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(200):                                  # memorise 32 random images
        opt.zero_grad()
        F.cross_entropy(model(images), labels).backward()
        opt.step()
    assert accuracy(model, images, labels) == 1.0
    for steps in (1, 5):
        adv = attack(model, images, labels, 0.1, steps=steps)
        assert (adv - images).abs().max() <= 0.1 + 1e-6 and adv.min() >= 0 and adv.max() <= 1
    curve = robustness_curve(model, images, labels)
    assert curve[0] == 1.0 and curve[-1] < curve[0]
    target = (labels + 1) % 10
    adv = attack(model, images, labels, 0.2, target=target, steps=10)
    with torch.no_grad():
        assert (model(adv).argmax(1) == target).float().mean() > 0.5
    print(f"attacks.py: ok (accuracy {curve[0]:.2f} → {curve[-1]:.2f} at ε = 0.3)")


if __name__ == "__main__":
    _self_check()
