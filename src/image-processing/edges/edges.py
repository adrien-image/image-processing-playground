"""
Edge detectors in PyTorch — the same pipeline as edgeDetectors.ts.

Sobel, Prewitt and Laplacian are single convolutions (F.conv2d with replicate padding).
Canny = Gaussian blur → Sobel → non-maximum suppression → hysteresis:
  - NMS quantises each gradient direction to 0°, 45°, 90° or 135° and compares every
    pixel with its two neighbours along that direction, using shifted copies of the image
  - hysteresis grows the strong edges into connected weak ones by repeated 3×3 dilation
    (max-pooling) restricted to the weak pixels, until nothing changes
Images are float tensors (H, W), values 0–255.
"""
import math
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
PREWITT_X = torch.tensor([[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]])
LAPLACIAN = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]])


def convolve(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r = kernel.shape[-1] // 2
    x = F.pad(image[None, None], (r, r, r, r), mode="replicate")
    return F.conv2d(x, kernel[None, None])[0, 0]


def gaussian_kernel(radius: int, sigma: float) -> torch.Tensor:
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def normalise(values: torch.Tensor) -> torch.Tensor:
    """|values| scaled so the strongest response is 255."""
    peak = values.abs().max()
    return values.abs() * (255 / peak) if peak > 0 else torch.zeros_like(values)


def _shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """out[y, x] = in[y + dy, x + dx], replicating the border."""
    p = F.pad(x[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    h, w = x.shape
    return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


def non_max_suppression(mag: torch.Tensor, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    angle = torch.rad2deg(torch.atan2(gy, gx)) % 180
    sector = torch.where((angle < 22.5) | (angle >= 157.5), 0, torch.where(angle < 67.5, 1, torch.where(angle < 112.5, 2, 3)))
    # Neighbour offsets (dx, dy) along the gradient for each sector (y points down).
    offsets = [((-1, 0), (1, 0)), ((-1, -1), (1, 1)), ((0, -1), (0, 1)), ((1, -1), (-1, 1))]
    keep = torch.zeros_like(mag, dtype=torch.bool)
    for s, ((ax, ay), (bx, by)) in enumerate(offsets):
        # ">=" on one side only, so a two-pixel plateau keeps exactly one pixel.
        keep |= (sector == s) & (mag >= _shift(mag, ax, ay)) & (mag > _shift(mag, bx, by))
    return torch.where(keep & (mag > 0), mag, torch.zeros_like(mag))


def hysteresis(nms: torch.Tensor, low: float, high: float) -> torch.Tensor:
    weak = (nms >= low) & (nms > 0)
    edges = (nms >= high) & (nms > 0)
    while True:
        grown = F.max_pool2d(edges[None, None].float(), 3, stride=1, padding=1)[0, 0].bool() & weak
        grown |= edges
        if torch.equal(grown, edges):
            return edges.float() * 255
        edges = grown


def canny(image: torch.Tensor, low_pct=20.0, high_pct=50.0, sigma=1.0) -> torch.Tensor:
    blurred = convolve(image, gaussian_kernel(2, sigma))
    gx, gy = convolve(blurred, SOBEL_X), convolve(blurred, SOBEL_X.T.contiguous())
    nms = non_max_suppression(torch.hypot(gx, gy), gx, gy)
    peak = nms.max()
    return hysteresis(nms, peak * low_pct / 100, peak * high_pct / 100)


def detect_edges(image: torch.Tensor, method: str) -> torch.Tensor:
    if method == "sobel-gx":
        return normalise(convolve(image, SOBEL_X))
    if method == "sobel-gy":
        return normalise(convolve(image, SOBEL_X.T.contiguous()))
    if method == "sobel-mag":
        return normalise(torch.hypot(convolve(image, SOBEL_X), convolve(image, SOBEL_X.T.contiguous())))
    if method == "prewitt":
        return normalise(torch.hypot(convolve(image, PREWITT_X), convolve(image, PREWITT_X.T.contiguous())))
    if method == "laplacian":
        return normalise(convolve(image, LAPLACIAN))
    if method == "canny":
        return canny(image)
    raise ValueError(method)


def _self_check():
    ys, xs = torch.meshgrid(torch.arange(40.), torch.arange(40.), indexing="ij")
    disc = torch.where((xs - 20) ** 2 + (ys - 20) ** 2 < 12 ** 2, 220.0, 30.0)
    edges = canny(disc)
    on = torch.nonzero(edges)
    radius = torch.hypot(on[:, 1].float() - 20, on[:, 0].float() - 20)
    assert len(on) > 40 and (radius - 12).abs().max() < 2.5     # a thin ring around the disc
    assert edges[20, 20] == 0 and edges[0, 0] == 0
    assert detect_edges(torch.full((10, 10), 99.), "sobel-mag").max() == 0   # flat image: no edges, no frame
    print(f"edges.py: ok ({len(on)} edge pixels)")


if __name__ == "__main__":
    _self_check()
