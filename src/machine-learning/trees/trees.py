"""
Decision trees (CART) and random forests in PyTorch, the algorithm of trees.ts.

A node tries every feature and every threshold halfway between two consecutive distinct
values, and keeps the split with the largest impurity decrease

    gain = I(parent) − (n_L / n)·I(left) − (n_R / n)·I(right)
    Gini I = 1 − Σ p_c²,   entropy I = −Σ p_c log₂ p_c,   regression: I = variance of y

The sweep over thresholds is vectorised: sort the node's points by the feature, and a
cumulative sum of one-hot labels gives the class counts left of every threshold at once.
A random forest grows each tree on a bootstrap sample, with one random feature per split
(more if that one cannot split), and averages the trees' class proportions.
"""
import math

import torch


def impurity(counts, kind="gini"):
    """Impurity of nodes from their class counts (last dimension)."""
    n = counts.sum(-1, keepdim=True).clamp_min(1e-12)
    p = counts / n
    if kind == "gini":
        return 1 - (p * p).sum(-1)
    return -(p * torch.log2(p.clamp_min(1e-300))).sum(-1)


def best_split(X, rows, node_stats, features, min_leaf, max_features):
    """The best (gain, feature, threshold, sorted rows, n_left) among `features`, tried in order."""
    best = None
    for k, f in enumerate(features):
        if best is not None and k >= max_features:
            break
        values, order = torch.sort(X[rows, f], stable=True)
        gain = node_stats(rows[order])
        m = len(rows)
        n_left = torch.arange(1, m)
        valid = (values[1:] > values[:-1]) & (n_left >= min_leaf) & (m - n_left >= min_leaf)
        if not valid.any():
            continue
        gain = torch.where(valid, gain, torch.full_like(gain, -math.inf))
        i = int(gain.argmax())
        g = gain[i].item()
        better = best is None or g > best[0] + 1e-12 or (g > best[0] - 1e-12 and f < best[1])
        if g > 1e-12 and better:
            best = (g, f, (values[i] + values[i + 1]).item() / 2, rows[order], i + 1)
    return best


def fit_tree(X, y, max_depth, min_leaf=1, kind="gini", classes=None, rows=None, max_features=None, generator=None, depth=0):
    """A tree as nested dicts. kind = "gini" | "entropy" (labels y) or "variance" (targets y)."""
    rows = torch.arange(len(y)) if rows is None else rows
    d = X.shape[1]
    if kind == "variance":
        t = y[rows].double()
        node = {"depth": depth, "n": len(rows), "value": t.mean().item(), "impurity": t.var(unbiased=False).item()}

        def stats(sorted_rows):
            # Weighted child variances from running sums of y and y² (n·var = Σy² − (Σy)²/n).
            s = y[sorted_rows].double()
            m = len(s)
            cs, cs2 = s.cumsum(0)[:-1], (s * s).cumsum(0)[:-1]
            nl = torch.arange(1, m, dtype=torch.float64)
            left = cs2 - cs * cs / nl
            right = (s * s).sum() - cs2 - (s.sum() - cs) ** 2 / (m - nl)
            return node["impurity"] - (left + right) / m
    else:
        counts = torch.bincount(y[rows], minlength=classes).double()
        node = {"depth": depth, "n": len(rows), "counts": counts, "value": int(counts.argmax()), "impurity": impurity(counts, kind).item()}

        def stats(sorted_rows):
            onehot = torch.nn.functional.one_hot(y[sorted_rows], classes).double()
            left = onehot.cumsum(0)[:-1]
            right = counts - left
            m = len(sorted_rows)
            nl = torch.arange(1, m, dtype=torch.float64)
            return node["impurity"] - nl / m * impurity(left, kind) - (m - nl) / m * impurity(right, kind)

    if node["impurity"] < 1e-14 or depth >= max_depth or len(rows) < 2 * min_leaf:
        return node
    features = torch.randperm(d, generator=generator).tolist() if generator is not None else list(range(d))
    best = best_split(X, rows, stats, features, min_leaf, max_features or d)
    if best is None:
        return node
    gain, f, threshold, sorted_rows, n_left = best
    node.update(feature=f, threshold=threshold, gain=gain)
    args = dict(min_leaf=min_leaf, kind=kind, classes=classes, max_features=max_features, generator=generator, depth=depth + 1)
    node["left"] = fit_tree(X, y, max_depth, rows=sorted_rows[:n_left], **args)
    node["right"] = fit_tree(X, y, max_depth, rows=sorted_rows[n_left:], **args)
    return node


def leaf(node, x):
    while "feature" in node:
        node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
    return node


def predict_proba(node, X):
    return torch.stack([(lambda l: l["counts"] / l["n"])(leaf(node, x)) for x in X])


def predict_value(node, X):
    return torch.tensor([leaf(node, x)["value"] for x in X])


def depth_of(node):
    return max(depth_of(node["left"]), depth_of(node["right"])) if "feature" in node else node["depth"]


def fit_forest(X, y, classes, trees, max_depth, min_leaf=1, kind="gini", bootstrap=True, max_features=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(y)
    forest = []
    for _ in range(trees):
        rows = torch.randint(0, n, (n,), generator=g) if bootstrap else torch.arange(n)
        forest.append(fit_tree(X, y, max_depth, min_leaf, kind, classes, rows, max_features, g))
    return forest


def forest_proba(forest, X):
    return torch.stack([predict_proba(t, X) for t in forest]).mean(0)


def make_moons(n, noise):
    """The two moons of the lesson (shared/datasets.ts), from torch's global generator."""
    c = torch.arange(n) % 2
    t = math.pi * torch.rand(n)
    x = torch.where(c == 0, torch.cos(t) - 0.5, 0.5 - torch.cos(t)) * 0.75
    y = torch.where(c == 0, torch.sin(t) - 0.25, 0.25 - torch.sin(t)) * 0.75
    return torch.stack([x, y], 1) + noise * torch.randn(n, 2), c


def _self_check():
    torch.manual_seed(0)
    # A pure node is a leaf.
    pure = fit_tree(torch.rand(6, 2), torch.ones(6, dtype=torch.long), 10, classes=2)
    assert "feature" not in pure and pure["impurity"] == 0
    # On hand data, feature 0 separates the classes perfectly at 0.45: Gini 0.5 → 0, the best possible gain.
    X = torch.tensor([[0.1, 0.9], [0.2, 0.1], [0.3, 0.5], [0.6, 0.8], [0.7, 0.2], [0.9, 0.4]])
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    stump = fit_tree(X, y, 1, classes=2)
    assert stump["feature"] == 0 and abs(stump["threshold"] - 0.45) < 1e-6 and abs(stump["gain"] - 0.5) < 1e-12
    for f in range(2):
        for t in torch.linspace(-0.05, 1, 22):
            left = X[:, f] <= t
            n_l = int(left.sum())
            gain = 0.5 - n_l / 6 * impurity(torch.bincount(y[left], minlength=2).double()).item() \
                - (6 - n_l) / 6 * impurity(torch.bincount(y[~left], minlength=2).double()).item()
            assert gain <= stump["gain"] + 1e-12
    # A regression tree splits where the step is and predicts the leaf means.
    reg = fit_tree(torch.arange(6.0)[:, None], torch.tensor([1.0, 1, 1, 5, 5, 5]), 3, kind="variance")
    assert abs(reg["threshold"] - 2.5) < 1e-9 and predict_value(reg, torch.tensor([[0.5], [4.2]])).tolist() == [1.0, 5.0]

    # The depth limit holds, and a tree without one memorises the noisy training set.
    Xtr, ytr = make_moons(300, 0.25)
    Xte, yte = make_moons(300, 0.25)
    for depth in (1, 3, 6):
        assert depth_of(fit_tree(Xtr, ytr, depth, classes=2)) <= depth
    deep = fit_tree(Xtr, ytr, 30, classes=2)
    assert (predict_proba(deep, Xtr).argmax(1) == ytr).all()
    # A forest of one tree, without bootstrap and with every feature at each split, is that tree.
    tree6 = fit_tree(Xtr, ytr, 6, 2, "entropy", 2)
    one = fit_forest(Xtr, ytr, 2, 1, 6, 2, "entropy", bootstrap=False, max_features=2)
    assert torch.equal(forest_proba(one, Xte), predict_proba(tree6, Xte))
    # Averaging decorrelated deep trees beats one deep tree on noisy test data (two training sets, 1000 test points).
    accs = []
    Xte, yte = make_moons(1000, 0.25)
    for rep in range(2):
        Xtr, ytr = make_moons(300, 0.25)
        deep = fit_tree(Xtr, ytr, 30, classes=2)
        forest = fit_forest(Xtr, ytr, 2, 40, 30, seed=rep)
        accs.append([(predict_proba(deep, Xte).argmax(1) == yte).double().mean().item(),
                     (forest_proba(forest, Xte).argmax(1) == yte).double().mean().item()])
    tree_acc, forest_acc = (sum(a[i] for a in accs) / 2 for i in range(2))
    assert all(f > t for t, f in accs) and forest_acc > tree_acc + 0.02, accs
    print(f"trees.py: ok (moons, noise 0.25: deep tree test {tree_acc:.3f}, forest of 40 {forest_acc:.3f})")

if __name__ == "__main__":
    _self_check()
