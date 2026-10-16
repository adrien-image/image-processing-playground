"""
Lens distortion and its calibration in PyTorch: the same model and fit as calibration.ts.

    x_d = x · (1 + k1·r² + k2·r⁴)        (normalised coordinates, x = (u − cx) / f)

For a guess of k, the detected corners are undistorted and a homography is fitted to
them by the direct linear transform (the smallest right singular vector, torch.linalg.svd).
Gauss–Newton then adjusts k to shrink the reprojection error; here autograd gives the
Jacobian of the residuals with respect to (k1, k2) instead of finite differences.
"""
import math
import torch


def distort(k, p):
    r2 = (p ** 2).sum(-1, keepdim=True)
    return p * (1 + k[0] * r2 + k[1] * r2 ** 2)


def undistort(k, d, iterations=30):
    p = d
    for _ in range(iterations):
        r2 = (p ** 2).sum(-1, keepdim=True)
        p = d / (1 + k[0] * r2 + k[1] * r2 ** 2)
    return p


def apply_h(H, p):
    q = torch.cat([p, torch.ones_like(p[:, :1])], 1) @ H.T
    return q[:, :2] / q[:, 2:]


def fit_homography(src, dst):
    """Least-squares homography src → dst (normalised DLT)."""
    def normaliser(p):
        mean = p.mean(0)
        s = 2 ** 0.5 / (p - mean).norm(dim=1).mean()
        z, o = torch.zeros((), dtype=p.dtype), torch.ones((), dtype=p.dtype)
        # torch.stack (not torch.tensor) keeps the dependence on p for autograd.
        return torch.stack([torch.stack([s, z, -s * mean[0]]), torch.stack([z, s, -s * mean[1]]), torch.stack([z, z, o])])
    Ts, Td = normaliser(src), normaliser(dst)
    a, b = apply_h(Ts, src), apply_h(Td, dst)
    zeros, ones = torch.zeros_like(a[:, :1]), torch.ones_like(a[:, :1])
    rows_x = torch.cat([-a, -ones, zeros, zeros, zeros, b[:, :1] * a, b[:, :1]], 1)
    rows_y = torch.cat([zeros, zeros, zeros, -a, -ones, b[:, 1:] * a, b[:, 1:]], 1)
    h = torch.linalg.svd(torch.cat([rows_x, rows_y])).Vh[-1].reshape(3, 3)
    H = torch.linalg.inv(Td) @ h @ Ts
    return H / H[2, 2]


def calibrate(board, observed, f, centre, iterations=20):
    """Return (k, H, rms per iteration). board: (N, 2) board units; observed: (N, 2) pixels."""
    observed_n = (observed - centre) / f

    def residuals(k):
        H = fit_homography(board, centre + f * undistort(k, observed_n))
        predicted = centre + f * distort(k, (apply_h(H, board) - centre) / f)
        return (predicted - observed).reshape(-1)

    k = torch.zeros(2, dtype=observed.dtype)
    lam, rms = 1e-3, []
    for _ in range(iterations):
        r = residuals(k)
        rms.append((r.pow(2).sum() / (len(r) / 2)).sqrt().item())
        J = torch.autograd.functional.jacobian(residuals, k)           # (2N, 2)
        A, g = J.T @ J, J.T @ r
        for _ in range(10):
            step = torch.linalg.solve(A + lam * torch.diag(A.diag()), -g)
            if residuals(k + step).pow(2).sum() < r.pow(2).sum():
                k, lam = k + step, max(lam / 3, 1e-7)
                break
            lam *= 10
        else:
            break
    H = fit_homography(board, centre + f * undistort(k, observed_n))
    return k, H, rms


def _self_check():
    dtype = torch.float64
    f, centre = 160.0, torch.tensor([80.0, 60.0], dtype=dtype)
    true_k = torch.tensor([-0.3, 0.08], dtype=dtype)
    j, i = torch.meshgrid(torch.arange(1, 7, dtype=dtype), torch.arange(1, 10, dtype=dtype), indexing="ij")
    board = torch.stack([i.reshape(-1), j.reshape(-1)], 1)
    c, s = 11 * math.cos(0.2), 11 * math.sin(0.2)
    pose = torch.tensor([[c, -s, 30.0], [s, c, 18.0], [0.002, 0.0, 1.0]], dtype=dtype)
    observed = centre + f * distort(true_k, (apply_h(pose, board) - centre) / f)

    # undistort inverts distort
    p = torch.tensor([[0.3, -0.2], [-0.45, 0.33]], dtype=dtype)
    assert torch.allclose(undistort(true_k, distort(true_k, p)), p, atol=1e-10)
    # The corners alone give back the distortion and the pose.
    k, H, rms = calibrate(board, observed, f, centre)
    assert torch.allclose(k, true_k, atol=1e-3) and rms[-1] < 1e-3 < rms[0]
    assert torch.allclose(apply_h(H, board), apply_h(pose, board), atol=1e-2)
    # With 0.3 px detection noise the estimate stays close.
    g = torch.Generator().manual_seed(4)
    k_noisy, _, rms_noisy = calibrate(board, observed + 0.3 * torch.randn(observed.shape, generator=g, dtype=dtype), f, centre)
    assert abs(k_noisy[0] - true_k[0]) < 0.05 and rms_noisy[-1] < 0.5
    print(f"calibration.py: ok (k1 {k[0]:.4f}, k2 {k[1]:.4f}; RMS {rms[0]:.2f} → {rms[-1]:.1e} px)")


if __name__ == "__main__":
    _self_check()
