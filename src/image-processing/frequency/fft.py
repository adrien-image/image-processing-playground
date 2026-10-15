"""
Frequency-domain filtering in PyTorch — what fft.ts does, with torch.fft.

torch.fft.fft2 gives the 2-D spectrum; fftshift moves the zero frequency (DC, the mean
brightness) to the centre so filters can be described by their distance from it.
Multiply the spectrum by a gain mask, transform back, keep the real part.
Images are float tensors (H, W), values 0–255.
"""
import torch


def magnitude_spectrum(image: torch.Tensor) -> torch.Tensor:
    """log(1 + |F|), centred and scaled to 0–255 for display."""
    spectrum = torch.fft.fftshift(torch.fft.fft2(image.double()))
    mag = torch.log1p(spectrum.abs())
    return mag * (255 / mag.max())


def filter_mask(h: int, w: int, kind: str, shape: str, r1: float, r2: float | None = None) -> torch.Tensor:
    """Centred gain (0–1) per frequency. kind: low/high/band-pass/band-stop; shape: ideal/gaussian.
    For Gaussian filters the radius is the σ of the Gaussian."""
    r2 = r1 if r2 is None else r2
    inner, outer = min(r1, r2), max(r1, r2)
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    d = torch.hypot(xs - w // 2, ys - h // 2)

    def lowpass(r: float) -> torch.Tensor:
        return (d <= r).double() if shape == "ideal" else torch.exp(-d ** 2 / (2 * r ** 2))

    if kind == "low":
        return lowpass(r1)
    if kind == "high":
        return 1 - lowpass(r1)
    inside = lowpass(inner - 1e-9) if shape == "ideal" else lowpass(inner)
    if kind == "band-pass":
        return (lowpass(outer) - inside).clamp(min=0)
    if kind == "band-stop":
        return (1 - lowpass(outer) + inside).clamp(max=1)
    raise ValueError(kind)


def apply_filter(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Multiply the centred spectrum by `mask` and return to the image domain."""
    spectrum = torch.fft.fftshift(torch.fft.fft2(image.double()))
    return torch.fft.ifft2(torch.fft.ifftshift(spectrum * mask)).real


def _self_check():
    h, w = 64, 64
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    smooth = 128 + 40 * torch.sin(2 * torch.pi * xs / 32)       # low frequency (2 cycles)
    fine = 30 * torch.sin(2 * torch.pi * xs * 16 / w)            # high frequency (16 cycles)
    img = smooth + fine
    low = apply_filter(img, filter_mask(h, w, "low", "ideal", 8))
    high = apply_filter(img, filter_mask(h, w, "high", "ideal", 8))
    assert torch.allclose(low, smooth, atol=1e-6)               # the low-pass keeps only the slow wave
    assert torch.allclose(high, fine, atol=1e-6)                # the high-pass keeps only the fine one
    assert torch.allclose(low + high, img, atol=1e-6)
    assert filter_mask(8, 8, "low", "gaussian", 2)[4, 4] == 1   # DC passes unchanged
    print("fft.py: ok")


if __name__ == "__main__":
    _self_check()
