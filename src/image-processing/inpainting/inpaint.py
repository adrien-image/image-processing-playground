"""
Inpainting in PyTorch — inpaint.ts with tensors.

Diffusion: repeat u ← mean of 4 neighbours (a conv2d with a cross kernel) inside the hole,
keeping known pixels fixed (Jacobi iterations of Laplace's equation).
Exemplar-based (Criminisi): the patch search is vectorised with F.unfold: every candidate
patch of the image becomes a column, and the masked SSD to the target patch is one
matrix expression. The fill loop itself stays sequential: each step depends on the last.
"""
import torch
import torch.nn.functional as F


def diffusion_inpaint(img: torch.Tensor, mask: torch.Tensor, iterations: int = 500) -> torch.Tensor:
    """img (C, H, W), mask (H, W) bool (True = missing) → filled image."""
    known = ~mask
    border = F.max_pool2d(mask[None, None].float(), 3, 1, 1)[0, 0].bool() & known
    u = img.clone()
    u[:, mask] = img[:, border].mean(1, keepdim=True)            # start at the mean border colour
    cross = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=img.dtype).expand(img.shape[0], 1, 3, 3)
    count = F.conv2d(F.pad(torch.ones(1, 1, *mask.shape, dtype=img.dtype), (1, 1, 1, 1)), cross[:1])[0, 0]
    for _ in range(iterations):
        avg = F.conv2d(F.pad(u[None], (1, 1, 1, 1)), cross, groups=img.shape[0])[0] / count
        u = torch.where(mask, avg, img)
    return u


def exemplar_inpaint(img: torch.Tensor, mask: torch.Tensor, radius: int = 4, use_priority: bool = True) -> torch.Tensor:
    """Criminisi inpainting of img (C, H, W) where mask (H, W) is True."""
    c, h, w = img.shape
    k = 2 * radius + 1
    out = img.clone()
    known = ~mask.clone()
    confidence = known.to(img.dtype)
    # Candidate sources: all fully known patches (as columns of `patches`).
    full = F.unfold(known[None, None].to(img.dtype), k)[0].min(0).values.bool()          # (L,)
    patches = F.unfold(img[None], k)[0][:, full]                                         # (C·k², S)
    centres = torch.nonzero(full)[:, 0]
    ow = w - k + 1
    src_y, src_x = centres // ow + radius, centres % ow + radius
    gray = lambda t: 0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2] if c == 3 else t[0]
    cross = torch.ones(1, 1, 3, 3, dtype=img.dtype)

    while not known.all():
        front = ~known & (F.conv2d(known[None, None].to(img.dtype), cross, padding=1)[0, 0] > 0)
        ys, xs = torch.nonzero(front, as_tuple=True)
        if use_priority:
            conf = F.avg_pool2d(confidence[None, None], k, 1, radius, count_include_pad=False)[0, 0][ys, xs]
            # Data term |∇I⊥ · n|: the gradient comes from the strongest-gradient known neighbour whose own
            # neighbours are known (max_pool2d with indices finds it); n is the normal of the front.
            gy, gx = torch.gradient(gray(out))
            reliable = -F.max_pool2d(-known[None, None].to(img.dtype), 3, 1, 1)[0, 0] > 0   # known 3×3 neighbourhood
            strength = torch.where(reliable, gx ** 2 + gy ** 2, torch.full_like(gx, -1.0))
            _, where = F.max_pool2d(strength[None, None], 3, 1, 1, return_indices=True)
            where = where[0, 0][ys, xs]
            ny, nx = torch.gradient(known.to(img.dtype))                              # points into the known side
            norm = torch.sqrt(nx ** 2 + ny ** 2).clamp(min=1e-9)
            gxs, gys = gx.flatten()[where], gy.flatten()[where]
            usable = strength.flatten()[where] >= 0
            data = torch.where(usable, (-gys * nx[ys, xs] + gxs * ny[ys, xs]).abs() / norm[ys, xs], torch.zeros(())) / 255 + 1e-3
            pick = (conf * data).argmax()
        else:
            pick = 0
        ty, tx = ys[pick].item(), xs[pick].item()
        # Target patch (zero-padded at the image border) and which of its pixels are known.
        pad = F.pad(out[None], (radius,) * 4)[0][:, ty:ty + k, tx:tx + k].reshape(c, -1)
        valid = F.pad(known[None, None].to(img.dtype), (radius,) * 4)[0, 0][ty:ty + k, tx:tx + k].reshape(-1)
        cost = (((patches.reshape(c, k * k, -1) - pad[:, :, None]) ** 2) * valid[None, :, None]).sum((0, 1))
        best = cost.argmin()
        sy, sx = src_y[best].item(), src_x[best].item()
        patch_conf = confidence[max(0, ty - radius):ty + radius + 1, max(0, tx - radius):tx + radius + 1].mean()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y, x = ty + dy, tx + dx
                if 0 <= y < h and 0 <= x < w and not known[y, x]:
                    out[:, y, x] = img[:, sy + dy, sx + dx]
                    known[y, x] = True
                    confidence[y, x] = patch_conf
    return out


def _self_check():
    h, w = 40, 48
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    hole = ((xs - 24).abs() < 6) & ((ys - 20).abs() < 6)
    rms = lambda a, b: ((a - b)[:, hole] ** 2).mean().sqrt().item()

    ramp = (40 + 3 * xs + 2 * ys).expand(3, h, w)
    assert rms(diffusion_inpaint(ramp, hole, 800), ramp) < 1, "diffusion should solve a linear ramp exactly"

    stripes = torch.where(xs % 6 < 3, 220.0, 30.0).to(torch.float64).expand(3, h, w).clone()
    blurred, copied = rms(diffusion_inpaint(stripes, hole, 800), stripes), rms(exemplar_inpaint(stripes, hole, 3), stripes)
    assert blurred > 60 and copied < 10, (blurred, copied)
    print(f"inpaint.py: ok (stripes, RMS error in the hole: diffusion {blurred:.0f}, exemplar {copied:.1f})")


if __name__ == "__main__":
    _self_check()
