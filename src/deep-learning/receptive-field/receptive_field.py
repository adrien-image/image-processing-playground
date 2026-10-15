"""
Receptive fields — receptiveField.ts in Python, plus a check with autograd.

The formulas (per layer with kernel k, stride s, padding p, dilation d):
    size   n_l = ⌊(n_{l−1} + 2p − d(k−1) − 1) / s⌋ + 1
    jump   j_l = j_{l−1} · s
    field  r_l = r_{l−1} + d(k−1) · j_{l−1}
    start  c_l = c_{l−1} + (d(k−1)/2 − p) · j_{l−1}

PyTorch can also *measure* a receptive field: build the network with every weight set
to 1, back-propagate from one output unit to the input, and the input gradient is
non-zero exactly on the pixels that unit depends on. Its values count the paths from
each pixel to the unit — the "effective receptive field" shading in the lesson.
"""
import torch
import torch.nn as nn


def layer_geometry(input_size: int, layers: list[dict]) -> list[dict]:
    """Geometry of the input (first entry) and of every layer's output."""
    geometry = [{"size": input_size, "jump": 1, "field": 1, "start": 0.0}]
    for layer in layers:
        k, s, p, d = layer["kernel"], layer["stride"], layer["padding"], layer.get("dilation", 1)
        prev = geometry[-1]
        span = d * (k - 1)
        geometry.append({
            "size": (prev["size"] + 2 * p - span - 1) // s + 1,
            "jump": prev["jump"] * s,
            "field": prev["field"] + span * prev["jump"],
            "start": prev["start"] + (span / 2 - p) * prev["jump"],
        })
    return geometry


def build(layers: list[dict]) -> nn.Sequential:
    """The same stack as real layers, every weight 1 and no bias, so gradients count paths."""
    modules = []
    for layer in layers:
        k, s, p, d = layer["kernel"], layer["stride"], layer["padding"], layer.get("dilation", 1)
        if layer["kind"] == "conv":
            conv = nn.Conv2d(1, 1, k, stride=s, padding=p, dilation=d, bias=False)
            nn.init.ones_(conv.weight)
            modules.append(conv)
        else:  # sum-pooling has the same geometry as average pooling, and integer path counts
            conv = nn.Conv2d(1, 1, k, stride=s, padding=p, bias=False)
            nn.init.ones_(conv.weight)
            modules.append(conv)
    return nn.Sequential(*modules).double()


def measured_field(layers: list[dict], input_size: int, unit: tuple[int, int]) -> torch.Tensor:
    """Path counts from every input pixel to output unit (row, col) of the last layer."""
    x = torch.zeros(1, 1, input_size, input_size, dtype=torch.float64, requires_grad=True)
    out = build(layers)(x)
    out[0, 0, unit[0], unit[1]].backward()
    return x.grad[0, 0]


LENET = [
    {"kind": "conv", "kernel": 5, "stride": 1, "padding": 0},
    {"kind": "pool", "kernel": 2, "stride": 2, "padding": 0},
    {"kind": "conv", "kernel": 5, "stride": 1, "padding": 0},
    {"kind": "pool", "kernel": 2, "stride": 2, "padding": 0},
]


def _self_check():
    g = layer_geometry(28, LENET)
    assert [l["size"] for l in g] == [28, 24, 12, 8, 4]
    assert [l["field"] for l in g] == [1, 5, 6, 14, 16]
    # Autograd agrees with the formula: unit (2, 2) of LeNet's last layer sees a 16×16 window.
    counts = measured_field(LENET, 28, (2, 2))
    rows, cols = torch.nonzero(counts, as_tuple=True)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (8, 23, 8, 23)
    centre = g[-1]["start"] + 2 * g[-1]["jump"]
    assert centre == 15.5   # pixel-index coordinates: the middle of 8…23
    # Five padded 3×3 convolutions: 11 pixels wide, peaked in the middle (binomial-like).
    stack = [{"kind": "conv", "kernel": 3, "stride": 1, "padding": 1}] * 5
    profile = measured_field(stack, 32, (16, 16))[16]
    assert torch.equal(profile[11:22] / profile[16] * 51, torch.tensor([1., 5, 15, 30, 45, 51, 45, 30, 15, 5, 1], dtype=torch.float64))
    print("receptive_field.py: ok")


if __name__ == "__main__":
    _self_check()
