"""
Deconvolution in PyTorch — deblur.ts with torch.fft.

A blurred image is the sharp one convolved with the point spread function (PSF) plus noise:
g = h ⊛ f + n, i.e. G = H · F + N in the frequency domain.

  inverse filter    F̂ = G / H                    perfect without noise, explodes with it
  Wiener filter     F̂ = H* · G / (|H|² + K)       K ≈ noise-to-signal ratio
  Richardson–Lucy   f ← f · (h̃ ⊛ (g / (h ⊛ f)))   iterative, keeps the estimate positive

Convolutions are circular (via the FFT). Images are float tensors (H, W), values 0–255.
"""
import math
import torch


def make_psf(kind: str, size: int, param: float, angle_deg: float = 0.0) -> torch.Tensor:
    """A normalised size×size PSF: 'motion' (length), 'gaussian' (σ) or 'defocus' (radius)."""
    c = (size - 1) / 2
    psf = torch.zeros(size, size, dtype=torch.float64)
    if kind == "motion":
        a = math.radians(angle_deg)
        t = torch.linspace(-0.5, 0.5, max(2, round(param * 8)), dtype=torch.float64) * param
        x = torch.round(c + t * math.cos(a)).long()
        y = torch.round(c - t * math.sin(a)).long()
        ok = (x >= 0) & (y >= 0) & (x < size) & (y < size)
        psf.index_put_((y[ok], x[ok]), torch.ones(int(ok.sum()), dtype=torch.float64), accumulate=True)
    else:
        ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float64), torch.arange(size, dtype=torch.float64), indexing="ij")
        d2 = (xs - c) ** 2 + (ys - c) ** 2
        psf = torch.exp(-d2 / (2 * param ** 2)) if kind == "gaussian" else (d2 <= param ** 2).double()
    return psf / psf.sum()


def psf_spectrum(psf: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """H: the PSF zero-padded to the image size with its centre moved to (0, 0)."""
    h = torch.zeros(shape, dtype=torch.float64)
    k = psf.shape[0]
    h[:k, :k] = psf
    return torch.fft.fft2(torch.roll(h, shifts=(-(k // 2), -(k // 2)), dims=(0, 1)))


def blur(image: torch.Tensor, H: torch.Tensor, sigma: float = 0.0, generator=None) -> torch.Tensor:
    out = torch.fft.ifft2(torch.fft.fft2(image.double()) * H).real
    return out + sigma * torch.randn(out.shape, generator=generator, dtype=out.dtype) if sigma > 0 else out


def wiener(blurred: torch.Tensor, H: torch.Tensor, K: float) -> torch.Tensor:
    """K = 0 is the plain inverse filter (with a tiny floor against exact zeros of H)."""
    G = torch.fft.fft2(blurred.double())
    return torch.fft.ifft2(H.conj() * G / (H.abs() ** 2 + max(K, 1e-12))).real


def richardson_lucy(blurred: torch.Tensor, H: torch.Tensor, iterations: int) -> torch.Tensor:
    conv = lambda x, h: torch.fft.ifft2(torch.fft.fft2(x) * h).real  # noqa: E731
    g = blurred.double().clamp(min=0) + 1               # strictly positive
    f = torch.full_like(g, g.mean().item())
    for _ in range(iterations):
        f = f * conv(g / conv(f, H).clamp(min=1e-6), H.conj())   # the flipped PSF's spectrum is H*
    return f - 1


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.double() - b.double().clamp(0, 255)) ** 2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def _self_check():
    g = torch.Generator().manual_seed(1)
    ys, xs = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    image = torch.where((xs > 16) & (xs < 44) & (ys > 20) & (ys < 40), 200.0, 40.0) + 30 * torch.sin(xs / 3)
    for psf in (make_psf("motion", 15, 9, 30), make_psf("gaussian", 15, 2), make_psf("defocus", 15, 4)):
        assert abs(psf.sum().item() - 1) < 1e-12
    mild = psf_spectrum(make_psf("gaussian", 9, 1.2), image.shape)
    assert psnr(image, wiener(blur(image, mild), mild, 0)) > 60               # no noise: inverse filter is exact
    H = psf_spectrum(make_psf("gaussian", 13, 2), image.shape)
    noisy = blur(image, H, 5, g)
    assert psnr(image, wiener(noisy, H, 0)) < psnr(image, noisy) - 10        # noise ÷ tiny |H| explodes
    assert psnr(image, wiener(noisy, H, 0.03)) > psnr(image, noisy) + 1.5     # Wiener copes
    assert psnr(image, richardson_lucy(noisy, H, 15)) > psnr(image, noisy) + 1.5
    print(f"deblur.py: ok (blurred {psnr(image, noisy):.1f} dB → Wiener {psnr(image, wiener(noisy, H, 0.03)):.1f} dB)")


if __name__ == "__main__":
    _self_check()
