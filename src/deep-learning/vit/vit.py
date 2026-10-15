"""
A tiny Vision Transformer for 28×28 digits — vit.ts in PyTorch.

  1. patchify   16 patches of 7×7 = 49 pixels (unfold)
  2. embed      a linear layer per patch → D-number tokens
  3. + [CLS] token and learned position embeddings
  4. blocks     LayerNorm → multi-head self-attention → +, LayerNorm → MLP → +
  5. head       the [CLS] token → 10 class scores
Self-attention is written out (instead of nn.MultiheadAttention) so the attention weights
softmax(Q·Kᵀ/√d) are visible and returned; attention rollout combines them over blocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH, GRID = 7, 4
PATCHES, TOKENS = GRID * GRID, GRID * GRID + 1


def patchify(images: torch.Tensor) -> torch.Tensor:
    """(N, 1, 28, 28) → (N, 16, 49): patches row by row, pixels row by row inside each."""
    return F.unfold(images, PATCH, stride=PATCH).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.ReLU(), nn.Linear(mlp, dim))

    def forward(self, x: torch.Tensor):
        n, t, d = x.shape
        h = self.ln1(x)
        split = lambda y: y.view(n, t, self.heads, d // self.heads).transpose(1, 2)  # noqa: E731  (N, H, T, dh)
        q, k, v = split(self.q(h)), split(self.k(h)), split(self.v(h))
        weights = torch.softmax(q @ k.transpose(-1, -2) / (d // self.heads) ** 0.5, dim=-1)   # (N, H, T, T)
        mixed = (weights @ v).transpose(1, 2).reshape(n, t, d)
        x = x + self.o(mixed)
        return x + self.mlp(self.ln2(x)), weights


class TinyViT(nn.Module):
    def __init__(self, dim=32, heads=2, layers=2, mlp=64):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, TOKENS, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads, mlp) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, images: torch.Tensor):
        """Returns (logits (N, 10), attention weights per block [(N, H, 17, 17)])."""
        tokens = self.embed(patchify(images))
        x = torch.cat([self.cls.expand(len(images), -1, -1), tokens], 1) + self.pos
        attention = []
        for block in self.blocks:
            x, w = block(x)
            attention.append(w)
        return self.head(self.norm(x)[:, 0]), attention

    def position_similarity(self) -> torch.Tensor:
        """Cosine similarity between the position embeddings of every pair of patches: (16, 16)."""
        p = F.normalize(self.pos[0, 1:], dim=1)
        return p @ p.T


def attention_rollout(attention: list[torch.Tensor]) -> torch.Tensor:
    """How much each patch feeds the [CLS] token through all blocks: average heads, add the
    residual (½A + ½I), renormalise rows, multiply the blocks (Abnar & Zuidema, 2020)."""
    rollout = torch.eye(TOKENS).expand(attention[0].shape[0], -1, -1)
    for a in attention:
        a = 0.5 * a.mean(1) + 0.5 * torch.eye(TOKENS)
        a = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout
    return rollout[:, 0, 1:]                      # (N, 16)


def train(model: TinyViT, images, labels, epochs=4, batch_size=64, lr=2e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(images))
        for start in range(0, len(images), batch_size):
            idx = order[start:start + batch_size]
            loss = F.cross_entropy(model(images[idx])[0], labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _self_check():
    torch.manual_seed(0)
    img = torch.arange(784.).view(1, 1, 28, 28)
    p = patchify(img)
    assert p.shape == (1, 16, 49) and p[0, 9, 1 * 7 + 2] == 15 * 28 + 9   # pixel (x 9, y 15) → patch 9, row 1, column 2
    # A position task only attention can solve: which quadrant holds the bright patch?
    images = torch.zeros(400, 1, 28, 28)
    labels = torch.randint(0, 4, (400,))
    for i, q in enumerate(labels.tolist()):
        r, c = (q // 2) * 14 + 3, (q % 2) * 14 + 3
        images[i, 0, r:r + 7, c:c + 7] = 1
    model = TinyViT()
    train(model, images, labels, epochs=15)
    model.eval()
    with torch.no_grad():
        logits, attention = model(images)
        acc = (logits.argmax(1) == labels).float().mean().item()
        assert acc > 0.9, acc
        assert torch.allclose(attention[0].sum(-1), torch.ones(1))            # rows of attention sum to 1
        roll = attention_rollout(attention)
        assert roll.shape == (400, 16) and model.position_similarity().shape == (16, 16)
    print(f"vit.py: ok (accuracy {acc:.2f})")


if __name__ == "__main__":
    _self_check()
