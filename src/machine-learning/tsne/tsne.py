"""
Exact t-SNE in PyTorch, as in tsne.ts:

    p(j|i) ∝ exp(−‖xᵢ − xⱼ‖² · βᵢ)       βᵢ = 1/2σᵢ² found by binary search: perplexity e^H = target
    pᵢⱼ = (p(j|i) + p(i|j)) / 2n
    qᵢⱼ ∝ (1 + ‖yᵢ − yⱼ‖²)⁻¹               (Student t with one degree of freedom)
    ∂KL/∂yᵢ = 4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)(1 + ‖yᵢ − yⱼ‖²)⁻¹

optimised with early exaggeration (P × 12 for 250 iterations), momentum 0.5 → 0.8 and gains.
The hand-written gradient is checked against autograd on KL(P ‖ Q).
"""
import math
import torch


def conditional_p(D, perplexity, steps=100):
    """Rows p(j|i) of the Gaussian affinities, every row's β found by bisection (all rows at once)."""
    n = len(D)
    target = math.log(perplexity)
    beta = torch.ones(n, dtype=D.dtype)
    lo, hi = torch.zeros(n, dtype=D.dtype), torch.full((n,), math.inf, dtype=D.dtype)
    off = ~torch.eye(n, dtype=torch.bool)
    Dm = D - D.masked_fill(~off, math.inf).min(1, keepdim=True).values   # avoid underflow
    for _ in range(steps):
        logits = (-beta[:, None] * Dm).masked_fill(~off, -math.inf)
        P = logits.softmax(1)
        H = -(P * P.clamp_min(1e-300).log()).sum(1)
        too_flat = H > target            # too many neighbours: narrow the Gaussian (raise β)
        lo = torch.where(too_flat, beta, lo)
        hi = torch.where(too_flat, hi, beta)
        beta = torch.where(hi.isinf(), beta * 2, (lo + hi) / 2)
    return P, H


def joint_p(X, perplexity):
    P, _ = conditional_p(torch.cdist(X, X) ** 2, perplexity)
    P = (P + P.T) / (2 * len(X))
    return P.clamp_min(1e-12).fill_diagonal_(0)


def kl_divergence(P, Y):
    W = 1 / (1 + torch.cdist(Y, Y) ** 2)
    W = W - torch.diag(W.diagonal())
    Q = W / W.sum()
    mask = P > 0
    return (P[mask] * (P[mask] / Q[mask]).log()).sum()


def kl_gradient(P, Y, exaggeration=1.0):
    W = 1 / (1 + torch.cdist(Y, Y) ** 2)
    W.fill_diagonal_(0)
    Q = W / W.sum()
    F = (exaggeration * P - Q) * W                       # n × n
    return 4 * (F.sum(1, keepdim=True) * Y - F @ Y)       # Σⱼ Fᵢⱼ (yᵢ − yⱼ)


def tsne(P, iterations=1000, learning_rate=None, exaggeration=12.0, early=250, generator=None):
    n = len(P)
    learning_rate = learning_rate or max(n / 4, 50.0)     # pᵢⱼ ~ 1/n: bigger sets need bigger steps
    Y = 1e-4 * torch.randn(n, 2, generator=generator, dtype=P.dtype)
    velocity, gains = torch.zeros_like(Y), torch.ones_like(Y)
    kl = []
    for it in range(iterations):
        grad = kl_gradient(P, Y, exaggeration if it < early else 1.0)
        momentum = 0.5 if it < early else 0.8
        gains = torch.where(grad.sign() != velocity.sign(), gains + 0.2, (gains * 0.8).clamp_min(0.01))
        velocity = momentum * velocity - learning_rate * gains * grad
        Y = Y + velocity
        Y = Y - Y.mean(0)
        kl.append(kl_divergence(P, Y).item())
    return Y, kl


def knn_agreement(Y, labels, k=5):
    D = torch.cdist(Y, Y).fill_diagonal_(math.inf)
    nearest = D.topk(k, largest=False).indices
    return (labels[nearest] == labels[:, None]).double().mean().item()


def _self_check():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(torch.initial_seed() + 1)   # follows the global seed
    X = torch.randn(80, 5, generator=g, dtype=torch.float64)
    for perplexity in (5, 20, 50):
        _, H = conditional_p(torch.cdist(X, X) ** 2, perplexity)
        assert torch.allclose(H.exp(), torch.full_like(H, perplexity), rtol=1e-4), perplexity
    P = joint_p(X, 15)
    assert torch.allclose(P, P.T) and abs(P.sum().item() - 1) < 1e-9

    # The analytic gradient equals autograd's gradient of KL(P ‖ Q).
    small = joint_p(X[:7], 3)
    Y = torch.randn(7, 2, generator=g, dtype=torch.float64, requires_grad=True)
    (auto,) = torch.autograd.grad(kl_divergence(small, Y), Y)
    assert torch.allclose(kl_gradient(small, Y.detach()), auto, atol=1e-10)

    # Four well-separated clusters in 10-D stay separated in the 2-D map.
    centres = 8 * torch.randn(4, 10, generator=g, dtype=torch.float64)
    labels = torch.arange(4).repeat_interleave(30)
    X = centres[labels] + torch.randn(120, 10, generator=g, dtype=torch.float64)
    Y, kl = tsne(joint_p(X, 20), iterations=600, generator=g)
    agreement = knn_agreement(Y, labels)
    assert agreement > 0.97, agreement
    assert kl[-1] < kl[260] and kl[-1] < kl[0], (kl[0], kl[260], kl[-1])
    print(f"tsne.py: ok (5-NN label agreement {agreement:.3f}; KL {kl[0]:.2f} → {kl[-1]:.2f})")


if __name__ == "__main__":
    _self_check()
