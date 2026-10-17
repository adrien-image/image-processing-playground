"""
Contrastive learning (SimCLR) in PyTorch: the NT-Xent loss of contrastive.ts and a small
encoder trained with it, on unlabelled synthetic glyphs.

    ℓ(i, j) = −log( exp(sim(zᵢ, zⱼ)/τ) / Σ_{k ≠ i} exp(sim(zᵢ, z_k)/τ) )

The loss is a cross-entropy over the 2N − 1 other views, where the "class" of each view
is the index of its partner. Augmentations use F.affine_grid / F.grid_sample. Labels are
only used at the end, for a k-nearest-neighbour accuracy of the learned features.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent(z1, z2, temperature):
    n = len(z1)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    logits = z @ z.T / temperature
    logits = logits.masked_fill(torch.eye(2 * n, dtype=torch.bool), float("-inf"))
    partner = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(logits, partner)


def nt_xent_loop(z1, z2, temperature):
    """The same loss written term by term, as in the paper."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n2 = len(z)
    total = 0.0
    for i in range(n2):
        j = (i + n2 // 2) % n2
        sims = torch.stack([z[i] @ z[k] / temperature for k in range(n2) if k != i])
        total = total - (z[i] @ z[j] / temperature - torch.logsumexp(sims, 0))
    return total / n2


def augment(x, generator, degrees=15.0, shift=3.0, noise=0.1):
    n = len(x)
    a = (torch.rand(n, generator=generator) * 2 - 1) * math.radians(degrees)
    s = 0.85 + 0.3 * torch.rand(n, generator=generator)
    t = (torch.rand(n, 2, generator=generator) * 2 - 1) * shift / 14         # pixels → [−1, 1] units
    theta = torch.stack([
        torch.stack([torch.cos(a) / s, torch.sin(a) / s, t[:, 0]], 1),
        torch.stack([-torch.sin(a) / s, torch.cos(a) / s, t[:, 1]], 1),
    ], 1)
    out = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), align_corners=False)
    return (out + noise * torch.randn(out.shape, generator=generator)).clamp(0, 1)


def encoder():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
    )


def knn_accuracy(train_h, train_y, test_h, test_y, k=5):
    sims = F.normalize(test_h, dim=1) @ F.normalize(train_h, dim=1).T
    votes = train_y[sims.topk(k, dim=1).indices]
    return (votes.mode(dim=1).values == test_y).float().mean().item()


def glyphs(n, generator):
    """Ten classes of two-stroke glyphs at random positions (as in transfer.py)."""
    strokes = [(0, 7), (0, 14), (0, 21), (1, 7), (1, 14), (1, 21)]
    classes = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 5), (2, 3), (0, 5), (1, 3), (2, 4), (0, 1)]
    labels = torch.randint(10, (n,), generator=generator)
    images = torch.zeros(n, 1, 28, 28)
    for i, c in enumerate(labels.tolist()):
        for st in classes[c]:
            vertical, pos = strokes[st]
            if vertical:
                images[i, 0, 4:24, pos - 1:pos + 2] = 1
            else:
                images[i, 0, pos - 1:pos + 2, 4:24] = 1
    return images, labels


def _self_check():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    # The vectorised loss is the paper's formula.
    z1, z2 = torch.randn(6, 8, generator=g), torch.randn(6, 8, generator=g)
    assert torch.allclose(nt_xent(z1, z2, 0.5), nt_xent_loop(z1, z2, 0.5), atol=1e-5)
    # Agreeing partners give a low loss, swapped partners a high one.
    e = torch.eye(4)
    assert nt_xent(e, e, 0.5) + 1 < nt_xent(e, e.roll(1, 0), 0.5)

    enc, head = encoder(), nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=2e-3)
    train_x, _ = glyphs(2000, g)                 # unlabelled
    losses = []
    for step in range(150):
        batch = train_x[torch.randint(len(train_x), (64,), generator=g)]
        loss = nt_xent(head(enc(augment(batch, g))), head(enc(augment(batch, g))), 0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Chance is log(2·64 − 1) ≈ 4.84. The loss cannot go near zero: with ten glyph classes a batch holds
    # several copies of each glyph, and the loss counts them as negatives it cannot tell apart.
    early, late = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    assert late < early - 0.3 and late < 0.75 * math.log(127), (early, late)
    # Labels only now: a 5-nearest-neighbour vote on the learned representation.
    with torch.no_grad():
        labelled_x, labelled_y = glyphs(500, g)
        test_x, test_y = glyphs(300, g)
        acc = knn_accuracy(enc(labelled_x), labelled_y, enc(augment(test_x, g)), test_y)
    assert acc > 0.5, acc
    print(f"contrastive.py: ok (loss {early:.2f} → {late:.2f}, kNN accuracy {acc:.2f})")


if __name__ == "__main__":
    _self_check()
