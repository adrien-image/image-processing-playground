"""
The lossy core of JPEG in PyTorch — jpeg.ts, on all 8×8 blocks at once.

  blocks    reshape the (padded) image into (blocks, 8, 8)
  DCT       C · B · Cᵀ for every block in one batched matrix product (C: orthonormal DCT-II)
  quantise  divide by the quality-scaled table and round: the only lossy step
  decode    multiply back, Cᵀ · D · C, shift by +128, clamp
Size is estimated from the entropy of JPEG's run-length symbols (see jpeg.ts).
Images are float tensors (H, W), values 0–255.
"""
import math
import torch
import torch.nn.functional as F

LUMA_QUANT = torch.tensor([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=torch.float64).view(8, 8)

n = torch.arange(8, dtype=torch.float64)
DCT = torch.cos((2 * n[None, :] + 1) * n[:, None] * math.pi / 16) * torch.where(n == 0, math.sqrt(1 / 8), math.sqrt(2 / 8))[:, None]

# Zigzag: walk the anti-diagonals, alternating direction.
ZIGZAG = [y * 8 + x for s in range(15) for y, x in
          (sorted(((y, s - y) for y in range(8) if 0 <= s - y < 8), reverse=(s % 2 == 0)))]


def quant_table(quality: int) -> torch.Tensor:
    q = min(100, max(1, round(quality)))
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return torch.floor((LUMA_QUANT * scale + 50) / 100).clamp(1, 255)


def to_blocks(image: torch.Tensor) -> torch.Tensor:
    """(H, W) → (blocks, 8, 8), padding the right and bottom edges by repetition."""
    h, w = image.shape
    ph, pw = -h % 8, -w % 8
    x = F.pad(image[None, None].double(), (0, pw, 0, ph), mode="replicate")[0, 0]
    return x.unfold(0, 8, 8).unfold(1, 8, 8).reshape(-1, 8, 8)


def from_blocks(blocks: torch.Tensor, h: int, w: int) -> torch.Tensor:
    by, bx = math.ceil(h / 8), math.ceil(w / 8)
    return blocks.reshape(by, bx, 8, 8).permute(0, 2, 1, 3).reshape(by * 8, bx * 8)[:h, :w]


def jpeg(image: torch.Tensor, quality: int):
    """Returns (decoded image, quantised coefficients (blocks, 8, 8))."""
    table = quant_table(quality)
    coeffs = DCT @ (to_blocks(image) - 128) @ DCT.T
    quantised = torch.round(coeffs / table)
    decoded = DCT.T @ (quantised * table) @ DCT + 128
    return from_blocks(decoded, *image.shape).round().clamp(0, 255), quantised


def category(v: torch.Tensor) -> torch.Tensor:
    """Bits needed for |v| (JPEG's size category); 0 for 0."""
    a = v.abs()
    return torch.where(a == 0, torch.zeros_like(a), torch.floor(torch.log2(a.clamp(min=1))) + 1)


def estimate_bits(quantised: torch.Tensor) -> float:
    zz = quantised.reshape(-1, 64)[:, ZIGZAG].long()
    dc = torch.diff(zz[:, 0], prepend=torch.zeros(1, dtype=torch.long))
    symbols, extra = [f"dc{int(c)}" for c in category(dc)], category(dc).sum().item()
    for row in zz[:, 1:].tolist():
        last = max((i for i, v in enumerate(row) if v != 0), default=-1)
        run = 0
        for v in row[:last + 1]:
            if v == 0:
                run += 1
                if run == 16:
                    symbols.append("zrl")
                    run = 0
                continue
            size = int(category(torch.tensor(float(v))))
            symbols.append(f"{run}/{size}")
            extra += size
            run = 0
        if last < 62:
            symbols.append("eob")
    dc_syms = [s for s in symbols if s.startswith("dc")]
    ac_syms = [s for s in symbols if not s.startswith("dc")]

    def entropy_bits(syms):
        counts = {}
        for s in syms:
            counts[s] = counts.get(s, 0) + 1
        return sum(c * max(1.0, -math.log2(c / len(syms))) for c in counts.values())

    return entropy_bits(dc_syms) + entropy_bits(ac_syms) + extra


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.double() - b.double()) ** 2).mean().item()
    return math.inf if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def _self_check():
    g = torch.Generator().manual_seed(3)
    block = torch.rand(8, 8, generator=g, dtype=torch.float64) * 255 - 128
    assert torch.allclose(DCT.T @ (DCT @ block @ DCT.T) @ DCT, block)          # orthonormal
    assert torch.equal(quant_table(50), LUMA_QUANT) and (quant_table(100) == 1).all()
    assert ZIGZAG[:6] == [0, 1, 8, 16, 9, 2] and sorted(ZIGZAG) == list(range(64))
    ys, xs = torch.meshgrid(torch.arange(36.), torch.arange(40.), indexing="ij")
    image = 128 + 80 * torch.sin(xs / 5) * torch.cos(ys / 7)
    high, q_high = jpeg(image, 90)
    low, q_low = jpeg(image, 10)
    assert high.shape == image.shape and psnr(image, high) > 35 and psnr(image, high) > psnr(image, low)
    assert estimate_bits(q_low) < estimate_bits(q_high)
    flat, _ = jpeg(torch.full((64, 64), 200.), 50)
    assert (flat == 200).all()
    print(f"jpeg.py: ok (quality 90: {psnr(image, high):.1f} dB, quality 10: {psnr(image, low):.1f} dB)")


if __name__ == "__main__":
    _self_check()
