"""
Gaussian mixtures fitted by EM in PyTorch: the same algorithm as gmm.ts, batched over
points and components.

    E-step  r[i, k] = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)
    M-step  N_k = Σ_i r[i, k],  π_k = N_k / n,  μ_k = Σ_i r[i, k] x_i / N_k,
            Σ_k = (Σ_i r[i, k] (x_i − μ_k)(x_i − μ_k)ᵀ + λI) / N_k

The λI floor keeps a component from collapsing onto one point (where the likelihood is
infinite). It is the exact M-step of log-likelihood − (λ/2) Σ_k tr(Σ_k⁻¹), so EM never
decreases that objective. k-means is the hard, equal, spherical limit of the same steps.
"""
import itertools
import math
import torch

REG = 1e-3


def e_step(x, weights, means, covs):
    """Responsibilities (n, K) and the log-likelihood, via log-sum-exp."""
    dist = torch.distributions.MultivariateNormal(means, covariance_matrix=covs)
    log_joint = dist.log_prob(x[:, None, :]) + weights.clamp_min(1e-300).log()   # (n, K)
    log_p = torch.logsumexp(log_joint, dim=1)
    return (log_joint - log_p[:, None]).exp(), log_p.sum().item()


def m_step(x, resp, reg=REG):
    nk = resp.sum(0).clamp_min(1e-10)                                  # (K,)
    means = resp.T @ x / nk[:, None]                                   # (K, 2)
    d = x[:, None, :] - means[None]                                    # (n, K, 2)
    scatter = torch.einsum("nk,nki,nkj->kij", resp, d, d)
    covs = (scatter + reg * torch.eye(2, dtype=x.dtype)) / nk[:, None, None]
    return nk / len(x), means, covs


def objective(x, weights, means, covs, reg=REG):
    return e_step(x, weights, means, covs)[1] - reg / 2 * torch.linalg.inv(covs).diagonal(dim1=1, dim2=2).sum().item()


def kmeans_pp(x, k):
    """k-means++ starting centres (uses the global torch RNG)."""
    centres = [x[torch.randint(len(x), (1,)).item()]]
    while len(centres) < k:
        d2 = torch.cdist(x, torch.stack(centres)).min(1).values ** 2
        centres.append(x[torch.multinomial(d2, 1).item()])
    return torch.stack(centres)


def start(x, centres):
    k = len(centres)
    s2 = max(x.var(0, unbiased=False).mean().item() / k, REG)
    return torch.full((k,), 1 / k, dtype=x.dtype), centres.clone(), s2 * torch.eye(2, dtype=x.dtype).repeat(k, 1, 1)


def fit(x, centres, iterations=300):
    params = start(x, centres)
    history = [objective(x, *params)]
    for _ in range(iterations):
        params = m_step(x, e_step(x, *params)[0])
        history.append(objective(x, *params))
        if abs(history[-1] - history[-2]) < 1e-9 * abs(history[-2]):
            break
    return params, history


def kmeans(x, centres, iterations=50):
    for _ in range(iterations):
        labels = torch.cdist(x, centres).argmin(1)
        centres = torch.stack([x[labels == k].mean(0) if (labels == k).any() else centres[k] for k in range(len(centres))])
    return centres, torch.cdist(x, centres).argmin(1)


def sample(weights, means, covs, n):
    counts = [round(w * n) for w in weights]
    counts[0] += n - sum(counts)
    parts = [torch.distributions.MultivariateNormal(m, covariance_matrix=c).sample((nk,)) for m, c, nk in zip(means, covs, counts)]
    return torch.cat(parts), torch.cat([torch.full((nk,), k) for k, nk in enumerate(counts)])


def _self_check():
    torch.manual_seed(0)
    dt = torch.float64
    true_w = torch.tensor([0.5, 0.3, 0.2], dtype=dt)
    true_mu = torch.tensor([[-0.6, 0.4], [0.6, 0.5], [0.1, -0.7]], dtype=dt)
    true_cov = torch.tensor([[[0.03, 0.01], [0.01, 0.02]], [[0.015, -0.005], [-0.005, 0.03]], [[0.02, 0], [0, 0.01]]], dtype=dt)
    x, _ = sample(true_w.tolist(), true_mu, true_cov, 2000)

    # The best of four k-means++ starts; EM never lowers the penalised log-likelihood.
    best = None
    for _ in range(4):
        params, history = fit(x, kmeans_pp(x, 3))
        assert all(b >= a - 1e-9 * abs(a) for a, b in zip(history, history[1:])), "objective decreased"
        if best is None or history[-1] > best[1][-1]:
            best = params, history
    (w, mu, cov), history = best
    resp, _ = e_step(x, w, mu, cov)
    assert torch.allclose(resp.sum(1), torch.ones(len(x), dtype=dt), atol=1e-12)
    assert (torch.linalg.eigvalsh(cov) > 0).all()

    # Parameters recovered up to relabelling: match each true mean to the nearest fitted one.
    match = torch.cdist(true_mu, mu).argmin(1)
    assert sorted(match.tolist()) == [0, 1, 2]
    mean_err = (mu[match] - true_mu).norm(dim=1).max().item()
    assert mean_err < 0.04, mean_err
    assert (w[match] - true_w).abs().max() < 0.04
    assert (cov[match] - true_cov).abs().max() < 0.006

    # A component on a lone outlier keeps variance ≈ λ instead of collapsing.
    blob, _ = sample([1.0], torch.zeros(1, 2, dtype=dt), 0.04 * torch.eye(2, dtype=dt)[None], 200)
    y = torch.cat([blob, torch.tensor([[1.3, 1.3]], dtype=dt)])
    params = start(y, torch.tensor([[0, 0], [0.1, -0.1], [1.3, 1.3]], dtype=dt))
    for _ in range(100):
        params = m_step(y, e_step(y, *params)[0])
    assert (torch.linalg.eigvalsh(params[2]) > 0).all() and math.isfinite(objective(y, *params))
    assert params[2][2, 0, 0] > 5e-4

    # Stretched clusters: the GMM finds them, k-means cuts across them.
    t = math.radians(35)
    u, v = torch.tensor([math.cos(t), math.sin(t)], dtype=dt), torch.tensor([-math.sin(t), math.cos(t)], dtype=dt)
    c = 0.42 ** 2 * torch.outer(u, u) + 0.05 ** 2 * torch.outer(v, v)
    s, labels = sample([1 / 3] * 3, torch.stack([-0.38 * v, 0 * v, 0.38 * v]), c.repeat(3, 1, 1), 300)
    fits = [fit(s, kmeans_pp(s, 3)) for _ in range(10)]   # EM finds local optima: keep the best start
    (w, mu, cov), _ = max(fits, key=lambda f: f[1][-1])
    gmm_labels = e_step(s, w, mu, cov)[0].argmax(1)
    _, km_labels = kmeans(s, kmeans_pp(s, 3))

    def agreement(pred):
        return max((torch.tensor(p)[pred] == labels).double().mean().item() for p in itertools.permutations(range(3)))

    gmm_acc, km_acc = agreement(gmm_labels), agreement(km_labels)
    assert gmm_acc > 0.95 and km_acc < 0.9, (gmm_acc, km_acc)
    print(f"gmm.py: ok ({len(history)} EM steps, mean error {mean_err:.4f}; stretched clusters: GMM {gmm_acc:.0%}, k-means {km_acc:.0%})")


if __name__ == "__main__":
    _self_check()
