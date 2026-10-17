"""
A recurrent network that reads an image row by row, in PyTorch: the GRU of rnn.ts.

PyTorch's GRU cell, written out (rnn.ts gives the same equations without the biases):
    r = σ(W_ir·x + b_ir + W_hr·h + b_hr)
    z = σ(W_iz·x + b_iz + W_hz·h + b_hz)
    n = tanh(W_in·x + b_in + r ⊙ (W_hn·h + b_hn))
    h' = (1 − z) ⊙ n + z ⊙ h
(here z is the fraction of the old state kept). nn.GRU applies it along the sequence; a
linear layer reads a prediction from every step. The self-check compares the equations
with nn.GRUCell, then trains on a task that needs memory.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gru_cell(x, h, cell: nn.GRUCell):
    """One step of nn.GRUCell from its own weights, gate by gate."""
    w_ir, w_iz, w_in = cell.weight_ih.chunk(3)
    w_hr, w_hz, w_hn = cell.weight_hh.chunk(3)
    b_ir, b_iz, b_in = cell.bias_ih.chunk(3)
    b_hr, b_hz, b_hn = cell.bias_hh.chunk(3)
    r = torch.sigmoid(x @ w_ir.T + b_ir + h @ w_hr.T + b_hr)
    z = torch.sigmoid(x @ w_iz.T + b_iz + h @ w_hz.T + b_hz)
    n = torch.tanh(x @ w_in.T + b_in + r * (h @ w_hn.T + b_hn))
    return (1 - z) * n + z * h


class RowReader(nn.Module):
    """GRU over the 28 rows (or columns) of an image, with a prediction after every step."""
    def __init__(self, units=64):
        super().__init__()
        self.gru = nn.GRU(28, units, batch_first=True)
        self.readout = nn.Linear(units, 10)

    def forward(self, images, order="rows"):
        seq = images.view(-1, 28, 28)
        if order == "columns":
            seq = seq.transpose(1, 2)
        hidden, _ = self.gru(seq)                       # (n, 28, units): h after every step
        return self.readout(hidden)                     # (n, 28, 10): logits after every step


def _self_check():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    # The written-out equations are exactly nn.GRUCell.
    cell = nn.GRUCell(28, 16).double()
    x, h = torch.randn(5, 28, generator=g, dtype=torch.float64), torch.randn(5, 16, generator=g, dtype=torch.float64)
    assert torch.allclose(gru_cell(x, h, cell), cell(x, h), atol=1e-12)

    # Memory: class c is a bright row at row 2 + 3c, the rest is noise; only the last step is scored.
    n = 300
    labels = torch.randint(3, (n,), generator=g)
    images = 0.1 * torch.rand(n, 28, 28, generator=g)
    images[torch.arange(n), 2 + 3 * labels] = 1
    model = RowReader(32)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(300):
        loss = F.cross_entropy(model(images)[:, -1], labels)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # the gradient through 25 steps can spike
        opt.step()
    with torch.no_grad():
        logits = model(images)
        last = (logits[:, -1].argmax(1) == labels).float().mean().item()
        early = (logits[:, 1].argmax(1) == labels).float().mean().item()   # after 2 rows the answer isn't visible yet
    assert last > 0.95 and early < 0.6, (last, early)
    print(f"rnn.py: ok (accuracy after 2 rows {early:.2f}, after all 28 {last:.2f})")


if __name__ == "__main__":
    _self_check()
