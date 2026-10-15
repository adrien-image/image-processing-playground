"""
Object-detection bookkeeping in PyTorch — detection.ts with batched tensor ops.

  box_iou        IoU of every box in A with every box in B at once ((N, M) matrix)
  nms            greedy non-max suppression (torchvision.ops.nms does the same, in C++)
  average_prec.  match detections to ground truth by score order, PASCAL-VOC style AP
  anchors        boxes of several sizes and aspect ratios centred on every grid cell,
                 labelled positive / ignored / background from their best IoU

Boxes are (N, 4) tensors of (x0, y0, x1, y1).
"""
import torch


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)  # noqa: E731
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area(a)[:, None] + area(b)[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, score_threshold: float = 0.0) -> torch.Tensor:
    """Indices of the kept boxes, best first."""
    order = torch.nonzero(scores >= score_threshold).flatten()
    order = order[scores[order].argsort(descending=True)]
    iou = box_iou(boxes, boxes)
    keep, alive = [], torch.ones(len(boxes), dtype=torch.bool)
    for i in order.tolist():
        if not alive[i]:
            continue
        keep.append(i)
        alive &= iou[i] <= iou_threshold      # drop everything overlapping the kept box (and the box itself)
    return torch.tensor(keep, dtype=torch.long)


def average_precision(boxes, scores, truth, iou_match=0.5):
    """AP, plus the precision and recall after each detection in score order."""
    order = scores.argsort(descending=True)
    iou = box_iou(boxes[order], truth)
    matched = torch.zeros(len(truth), dtype=torch.bool)
    tp = torch.zeros(len(order))
    for k in range(len(order)):
        candidates = torch.where(~matched & (iou[k] >= iou_match), iou[k], torch.full_like(iou[k], -1))
        best = candidates.argmax() if len(truth) else None
        if best is not None and candidates[best] >= iou_match:
            matched[best], tp[k] = True, 1
    cum_tp = tp.cumsum(0)
    precision = cum_tp / torch.arange(1, len(order) + 1)
    recall = cum_tp / max(len(truth), 1)
    # Make precision monotonic (best precision at any later rank), then integrate over recall.
    interp = precision.flip(0).cummax(0).values.flip(0)
    ap = (torch.diff(recall, prepend=torch.zeros(1)) * interp).sum().item()
    return ap, precision, recall


def anchors(image_size: int, grid: int, scales=(24, 48, 96), ratios=(0.5, 1.0, 2.0)) -> torch.Tensor:
    cell = image_size / grid
    centres = (torch.arange(grid) + 0.5) * cell
    cy, cx = torch.meshgrid(centres, centres, indexing="ij")
    shapes = torch.tensor([(s * r ** 0.5, s / r ** 0.5) for s in scales for r in ratios])   # (w, h)
    c = torch.stack([cx, cy], -1).reshape(-1, 1, 2)
    half = shapes[None] / 2
    return torch.cat([c - half, c + half], -1).reshape(-1, 4)


def label_anchors(anchor_boxes, truth, positive=0.5, negative=0.4):
    """1 = positive, 0 = background, −1 = ignored; every object also claims its best anchor."""
    iou = box_iou(anchor_boxes, truth)
    best_iou, best_truth = iou.max(1)
    labels = torch.full((len(anchor_boxes),), -1)
    labels[best_iou < negative] = 0
    labels[best_iou >= positive] = 1
    labels[iou.argmax(0)] = 1
    return labels, best_truth


def _self_check():
    b = lambda *xs: torch.tensor(xs, dtype=torch.float32)  # noqa: E731
    assert torch.isclose(box_iou(b([0, 0, 2, 2]), b([1, 1, 3, 3]))[0, 0], torch.tensor(1 / 7))
    boxes = b([0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [51, 50, 61, 60], [90, 90, 99, 99])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.95, 0.1])
    assert nms(boxes, scores, 0.5, 0.2).tolist() == [3, 0]
    truth = b([0, 0, 10, 10], [50, 50, 60, 60])
    ap, _, _ = average_precision(boxes, scores, truth)
    assert abs(ap - 1.0) < 1e-6                      # both objects found before any false positive
    ap_bad, _, _ = average_precision(b([80, 80, 90, 90], [0, 0, 10, 10]), torch.tensor([0.99, 0.5]), truth[:1])
    assert abs(ap_bad - 0.5) < 1e-6                  # a confident false positive ranks first
    a = anchors(256, 8)
    assert a.shape == (8 * 8 * 9, 4)
    labels, _ = label_anchors(a, b([10, 10, 50, 40], [120, 100, 220, 240]))
    assert (labels == 1).sum() >= 2 and (labels == 0).sum() > 300
    print("detection.py: ok")


if __name__ == "__main__":
    _self_check()
