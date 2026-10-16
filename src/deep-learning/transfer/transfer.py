"""
Transfer learning in PyTorch: the three strategies of transfer.ts.

    from scratch   a fresh network trained on the few new examples
    linear probe   the pretrained feature extractor frozen (requires_grad = False),
                   only a new last layer trained
    fine-tune      a copy of the pretrained network, all of it trained with a small
                   learning rate

The self-check uses synthetic "glyphs" instead of MNIST: every class is two strokes out
of six (three horizontal bars, three vertical), drawn with random shifts and noise.
Classes 0–4 and 5–9 pair the same strokes differently, so stroke detectors learned on
the first five transfer to the last five.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

STROKES = [(0, 7), (0, 14), (0, 21), (1, 7), (1, 14), (1, 21)]          # (vertical?, position)
CLASSES = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 5), (2, 3), (0, 5), (1, 3), (2, 4), (0, 1)]


def glyphs(classes, per_class, generator):
    images, labels = [], []
    for new_label, c in enumerate(classes):
        for _ in range(per_class):
            img = torch.zeros(28, 28)
            dx, dy = (torch.randint(-2, 3, (2,), generator=generator)).tolist()
            for s in CLASSES[c]:
                vertical, pos = STROKES[s]
                if vertical:
                    img[4 + dy:24 + dy, pos + dx - 1:pos + dx + 2] = 1
                else:
                    img[pos + dy - 1:pos + dy + 2, 4 + dx:24 + dx] = 1
            images.append(img + 0.3 * torch.rand(28, 28, generator=generator))
            labels.append(new_label)
    return torch.stack(images)[:, None], torch.tensor(labels)


def feature_extractor():
    """LeNet without its last layer: 84 features."""
    return nn.Sequential(
        nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 4 * 4, 84), nn.ReLU(),
    )


def train(model, params, x, y, epochs, lr, generator, batch=32):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=generator)
        for s in range(0, len(x), batch):
            idx = order[s:s + batch]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(1) == y).float().mean().item()


def _self_check():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    old_x, old_y = glyphs(range(5), 300, g)
    new_x, new_y = glyphs(range(5, 10), 3, g)               # three examples per new class
    test_x, test_y = glyphs(range(5, 10), 100, g)

    features = feature_extractor()
    pretrain = nn.Sequential(features, nn.Linear(84, 5))
    train(pretrain, pretrain.parameters(), old_x, old_y, 4, 0.003, g)
    assert accuracy(pretrain, *glyphs(range(5), 50, g)) > 0.9

    scratch = nn.Sequential(feature_extractor(), nn.Linear(84, 5))
    train(scratch, scratch.parameters(), new_x, new_y, 60, 0.003, g)

    frozen = copy.deepcopy(features)
    frozen.requires_grad_(False)
    probe = nn.Sequential(frozen, nn.Linear(84, 5))
    before = [p.clone() for p in frozen.parameters()]
    train(probe, [p for p in probe.parameters() if p.requires_grad], new_x, new_y, 60, 0.003, g)
    assert all(torch.equal(a, b) for a, b in zip(before, frozen.parameters()))   # frozen means frozen

    tuned = nn.Sequential(copy.deepcopy(features), nn.Linear(84, 5))
    train(tuned, tuned.parameters(), new_x, new_y, 60, 0.0005, g)

    acc = {name: accuracy(m, test_x, test_y) for name, m in [("scratch", scratch), ("probe", probe), ("fine-tune", tuned)]}
    assert acc["probe"] > acc["scratch"] and acc["fine-tune"] > acc["scratch"], acc
    print("transfer.py: ok (" + ", ".join(f"{k} {v:.2f}" for k, v in acc.items()) + " with 3 examples per class)")


if __name__ == "__main__":
    _self_check()
