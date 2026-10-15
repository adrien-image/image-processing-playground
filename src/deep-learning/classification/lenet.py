"""
LeNet-5 on MNIST in PyTorch — the model, training loop and "inside the CNN" tools of
lenet.ts and insight.ts.

  conv 5×5×6 → avg-pool 2 → conv 5×5×16 → avg-pool 2 → dense 120 → dense 84 → 10
  (44,426 weights for 28×28 input; the original LeNet-5 took 32×32)

Inside the network:
  - conv1's six 5×5 kernels are just `model.conv1.weight`
  - activation maps come from forward hooks on each layer
  - occlusion: hide a 6×6 patch at a time and measure how much p(class) drops

To train on real MNIST: `torchvision.datasets.MNIST(root, download=True)`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, 28, 28) in [0, 1] → logits (N, 10). Apply softmax for probabilities."""
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x.flatten(1)))
        x = F.relu(self.fc2(x))
        return self.out(x)


def train(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epochs=10, batch_size=128, lr=5e-4):
    """Adam on cross-entropy, reshuffled every epoch. Returns the mean loss of each epoch."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(images))
        total = 0.0
        for start in range(0, len(images), batch_size):
            idx = order[start:start + batch_size]
            loss = F.cross_entropy(model(images[idx]), labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        history.append(total / len(images))
    return history


@torch.no_grad()
def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor):
    """Accuracy and the 10×10 confusion matrix (rows: true digit, columns: predicted)."""
    model.eval()
    predicted = model(images).argmax(1)
    confusion = torch.zeros(10, 10, dtype=torch.long)
    confusion.index_put_((labels, predicted), torch.ones_like(labels), accumulate=True)
    return (predicted == labels).float().mean().item(), confusion


@torch.no_grad()
def activations(model: LeNet, image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Feature maps of conv1, pool1, conv2, pool2 for one (28, 28) image (conv outputs after ReLU)."""
    maps, hooks = {}, []
    for name in ["conv1", "pool1", "conv2", "pool2"]:
        layer = getattr(model, name)
        relu = name.startswith("conv")
        hooks.append(layer.register_forward_hook(lambda _m, _i, out, name=name, relu=relu: maps.__setitem__(name, (F.relu(out) if relu else out)[0])))
    model.eval()
    model(image[None, None])
    for h in hooks:
        h.remove()
    return maps


@torch.no_grad()
def occlusion_map(model: nn.Module, image: torch.Tensor, cls: int, patch=6, stride=2) -> torch.Tensor:
    """Mean drop in p(cls) over all patches that hid each pixel (all occluded copies in one batch)."""
    size = image.shape[-1]
    positions = list(range(0, size - patch + 1, stride))
    if positions[-1] + patch < size:
        positions.append(size - patch)
    corners = [(y, x) for y in positions for x in positions]
    batch = image.repeat(len(corners) + 1, 1, 1)
    for k, (y, x) in enumerate(corners, start=1):
        batch[k, y:y + patch, x:x + patch] = 0
    model.eval()
    p = F.softmax(model(batch[:, None]), 1)[:, cls]
    total, count = torch.zeros_like(image), torch.zeros_like(image)
    for k, (y, x) in enumerate(corners, start=1):
        total[y:y + patch, x:x + patch] += p[0] - p[k]
        count[y:y + patch, x:x + patch] += 1
    return total / count


def _self_check():
    torch.manual_seed(0)
    model = LeNet()
    assert sum(p.numel() for p in model.parameters()) == 44_426
    # A toy two-class problem: a small bright square near the top-left vs near the bottom-right.
    images = torch.zeros(64, 1, 28, 28)
    labels = torch.arange(64) % 2
    images[labels == 0, :, 6:10, 6:10] = 1
    images[labels == 1, :, 18:22, 18:22] = 1
    history = train(model, images, labels, epochs=15, batch_size=16, lr=3e-3)
    accuracy, confusion = evaluate(model, images, labels)
    assert history[-1] < history[0] and accuracy == 1.0 and confusion.sum() == 64
    maps = activations(model, images[0, 0])
    assert {k: tuple(v.shape) for k, v in maps.items()} == {"conv1": (6, 24, 24), "pool1": (6, 12, 12), "conv2": (16, 8, 8), "pool2": (16, 4, 4)}
    # Hiding the square removes the evidence for class 0; hiding empty background changes nothing.
    occ = occlusion_map(model, images[0, 0], 0)
    assert occ[8, 8] > 1e-3 and occ[24, 4].abs() < 1e-6, (occ[8, 8], occ[24, 4])
    print("lenet.py: ok")


if __name__ == "__main__":
    _self_check()
