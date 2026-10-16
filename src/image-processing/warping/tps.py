"""
Thin-plate spline warping in PyTorch: the same spline as tps.ts.

    f(p) = a0 + a1·x + a2·y + Σ w_i · U(‖p − c_i‖),   U(r) = r² log r²

The weights for x and y solve one (n + 3)×(n + 3) system together (two right-hand sides,
torch.linalg.solve). The warp evaluates the spline at every output pixel at once, and
F.grid_sample does the bilinear lookup. Points are in [0, 1] image coordinates.
"""
import torch
import torch.nn.functional as F


def tps_kernel(r2):
    return torch.where(r2 == 0, torch.zeros_like(r2), r2 * torch.log(r2.clamp(min=1e-300)))


def fit_tps(src, dst, lam=0.0):
    """src, dst: (n, 2). Returns (centres, weights (n, 2), affine (3, 2))."""
    n = len(src)
    K = tps_kernel(torch.cdist(src, src) ** 2) + lam * torch.eye(n, dtype=src.dtype)
    P = torch.cat([torch.ones(n, 1, dtype=src.dtype), src], 1)
    A = torch.zeros(n + 3, n + 3, dtype=src.dtype)
    A[:n, :n], A[:n, n:], A[n:, :n] = K, P, P.T
    b = torch.cat([dst, torch.zeros(3, 2, dtype=src.dtype)])
    sol = torch.linalg.solve(A, b)
    return src, sol[:n], sol[n:]


def apply_tps(tps, p):
    centres, w, a = tps
    U = tps_kernel(torch.cdist(p, centres) ** 2)                        # (m, n)
    return torch.cat([torch.ones(len(p), 1, dtype=p.dtype), p], 1) @ a + U @ w


def bending_energy(tps):
    centres, w, _ = tps
    return torch.trace(w.T @ tps_kernel(torch.cdist(centres, centres) ** 2) @ w)


def warp_image(image, inverse):
    """image: (C, H, W). Every output pixel samples the input at inverse(pixel)."""
    c, h, w = image.shape
    size = max(h, w)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=image.dtype), torch.arange(w, dtype=image.dtype), indexing="ij")
    pixels = torch.stack([(xs.reshape(-1) + 0.5) / size, (ys.reshape(-1) + 0.5) / size], 1)
    source = apply_tps(inverse, pixels) * size                          # pixel coordinates (edges at 0 and w)
    grid = torch.stack([source[:, 0] / w * 2 - 1, source[:, 1] / h * 2 - 1], 1).reshape(1, h, w, 2)
    return F.grid_sample(image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0]


def _self_check():
    d = torch.float64
    g = torch.Generator().manual_seed(3)
    src = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9], [0.5, 0.5], [0.3, 0.7], [0.7, 0.35]], dtype=d)
    dst = src + 0.1 * (torch.rand(src.shape, generator=g, dtype=d) - 0.5)
    # λ = 0 interpolates the control points exactly.
    assert torch.allclose(apply_tps(fit_tps(src, dst), src), dst, atol=1e-10)
    # An affine map is reproduced everywhere, with no bending.
    M, t = torch.tensor([[1.2, 0.2], [-0.3, 0.9]], dtype=d), torch.tensor([0.1, -0.05], dtype=d)
    tps = fit_tps(src, src @ M + t)
    probe = torch.tensor([[0.2, 0.4], [0.95, 0.05], [-0.3, 1.4]], dtype=d)
    assert torch.allclose(apply_tps(tps, probe), probe @ M + t, atol=1e-10) and abs(bending_energy(tps)) < 1e-10
    # More smoothing, less bending.
    pulled = src.clone()
    pulled[4] = torch.tensor([0.65, 0.6], dtype=d)
    energies = [bending_energy(fit_tps(src, pulled, lam)).item() for lam in (0, 0.01, 1, 100)]
    assert all(b < a for a, b in zip(energies, energies[1:]))
    # The identity warp leaves an image unchanged.
    image = torch.rand(3, 10, 12, generator=g, dtype=d)
    assert torch.allclose(warp_image(image, fit_tps(src, src)), image, atol=1e-6)
    print(f"tps.py: ok (bending energy {energies[0]:.3f} → {energies[-1]:.2e} as λ grows)")


if __name__ == "__main__":
    _self_check()
