"""
Connected components and region measurements in PyTorch — components.ts, vectorised.

Instead of a flood fill, labelling here uses label propagation, which maps well onto
tensor ops: give every foreground pixel a unique number, then repeatedly replace each
label with the largest label among its neighbours (a 3×3 max-pool, restricted to the
foreground) until nothing changes. Every pixel of a blob ends up with the same label.
Labels are then renumbered 1..count in raster order of each region's first pixel, like
the TypeScript version. 4-connectivity uses a cross-shaped neighbourhood instead of 3×3.

Masks are bool tensors (H, W).
"""
import math
import torch
import torch.nn.functional as F


def _neighbour_max(labels: torch.Tensor, connectivity: int) -> torch.Tensor:
    x = labels[None, None].double()
    if connectivity == 8:
        return F.max_pool2d(x, 3, stride=1, padding=1)[0, 0].long()
    # 4-connectivity: the max of the pixel and its left/right and up/down neighbours.
    rows = F.max_pool2d(x, (1, 3), stride=1, padding=(0, 1))
    cols = F.max_pool2d(x, (3, 1), stride=1, padding=(1, 0))
    return torch.maximum(rows, cols)[0, 0].long()


def label_components(mask: torch.Tensor, connectivity: int = 8):
    """Returns (labels (H, W) with 0 = background and 1..count = regions, count)."""
    h, w = mask.shape
    labels = torch.where(mask, torch.arange(1, h * w + 1).reshape(h, w), torch.zeros(h, w, dtype=torch.long))
    while True:
        grown = torch.where(mask, _neighbour_max(labels, connectivity), labels)
        if torch.equal(grown, labels):
            break
        labels = grown
    # Renumber 1..count in raster order of each region's first pixel.
    flat = labels.flatten()
    fg = flat > 0
    first = torch.full((h * w + 1,), h * w, dtype=torch.long)
    first.scatter_reduce_(0, flat[fg], torch.arange(h * w)[fg], reduce="amin")
    present = torch.unique(flat[fg])
    order = present[first[present].argsort()]
    remap = torch.zeros(h * w + 1, dtype=torch.long)
    remap[order] = torch.arange(1, len(order) + 1)
    return remap[labels], len(order)


def region_properties(labels: torch.Tensor, count: int, gray: torch.Tensor | None = None):
    """Area, centroid (pixel centres), bounding box, boundary-pixel perimeter, circularity, mean gray."""
    h, w = labels.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    padded = F.pad(labels[None, None].double(), (1, 1, 1, 1), value=-1)[0, 0]
    # A pixel is on the boundary if any 4-neighbour has a different label (or is outside the image).
    boundary = (padded[:-2, 1:-1] != labels) | (padded[2:, 1:-1] != labels) | (padded[1:-1, :-2] != labels) | (padded[1:-1, 2:] != labels)
    regions = []
    for label in range(1, count + 1):
        m = labels == label
        area = m.sum().item()
        perimeter = (boundary & m).sum().item()
        regions.append({
            "label": label,
            "area": area,
            "centroid": ((xs[m] + 0.5).mean().item(), (ys[m] + 0.5).mean().item()),
            "bbox": (int(xs[m].min()), int(ys[m].min()), int(xs[m].max()), int(ys[m].max())),
            "perimeter": perimeter,
            "circularity": 4 * math.pi * area / perimeter ** 2,
            "mean_intensity": gray[m].mean().item() if gray is not None else None,
        })
    return regions


def _self_check():
    mask = torch.zeros(12, 14, dtype=torch.bool)
    mask[1:4, 1:5] = True              # region 1: a 3×4 block
    mask[6:10, 8:12] = True            # region 2: a 4×4 block
    mask[10, 12] = True                # touches region 2 only diagonally
    labels, count = label_components(mask, 8)
    assert count == 2 and labels[10, 12] == labels[7, 9] == 2 and labels[2, 2] == 1
    _, count4 = label_components(mask, 4)
    assert count4 == 3                 # with 4-connectivity the diagonal pixel is its own region
    props = region_properties(labels, count)
    assert props[0]["area"] == 12 and props[0]["centroid"] == (3.0, 2.5) and props[0]["bbox"] == (1, 1, 4, 3)
    assert props[0]["perimeter"] == 10  # every pixel of a 3×4 block except the 2 inner ones
    print("components.py: ok")


if __name__ == "__main__":
    _self_check()
