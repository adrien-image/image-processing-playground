"""
Evaluation metrics for a binary classifier in PyTorch: the same definitions as metrics.ts.

    confusion at threshold t    TP, FP, FN, TN with "positive" meaning score ≥ t
    precision = TP / (TP + FP)  recall = TP / (TP + FN)  specificity = TN / (TN + FP)
    ROC    (FPR, TPR) at every distinct score, highest first; AUC by the trapezoid rule
    AP     Σₙ (Rₙ − Rₙ₋₁) Pₙ   (scikit-learn's average_precision_score)

The self-check works a small case by hand, checks that AUC equals the Mann–Whitney
probability P(score₊ > score₋) + ½ P(tie), and shows how class imbalance flatters
accuracy and lowers AP while leaving ROC AUC alone.
"""
import math
import torch


def confusion(scores, labels, t):
    predicted, actual = scores >= t, labels == 1
    return {"tp": int((predicted & actual).sum()), "fp": int((predicted & ~actual).sum()),
            "fn": int((~predicted & actual).sum()), "tn": int((~predicted & ~actual).sum())}


def rates(c):
    ratio = lambda a, b: a / b if b else math.nan
    return {"accuracy": ratio(c["tp"] + c["tn"], sum(c.values())), "precision": ratio(c["tp"], c["tp"] + c["fp"]),
            "recall": ratio(c["tp"], c["tp"] + c["fn"]), "specificity": ratio(c["tn"], c["tn"] + c["fp"]),
            "f1": ratio(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])}


def sweep(scores, labels):
    """Cumulative TP and FP at every distinct score, highest first (tied scores enter together)."""
    order = torch.argsort(scores, descending=True)
    s, y = scores[order], labels[order].double()
    tp, fp = torch.cumsum(y, 0), torch.cumsum(1 - y, 0)
    last = torch.ones(len(s), dtype=torch.bool)
    last[:-1] = s[1:] != s[:-1]  # the last example of each run of equal scores
    return s[last], tp[last], fp[last]


def roc_auc(scores, labels):
    _, tp, fp = sweep(scores, labels)
    zero = torch.zeros(1, dtype=tp.dtype)
    return torch.trapezoid(torch.cat([zero, tp / tp[-1]]), torch.cat([zero, fp / fp[-1]])).item()


def average_precision(scores, labels):
    _, tp, fp = sweep(scores, labels)
    recall, precision = tp / tp[-1], tp / (tp + fp)
    return (torch.diff(recall, prepend=torch.zeros(1, dtype=recall.dtype)) * precision).sum().item()


def mann_whitney(scores, labels):
    """P(random positive outscores random negative) + ½ P(tie), from every pair."""
    pos, neg = scores[labels == 1][:, None], scores[labels == 0][None, :]
    return ((pos > neg).double().mean() + 0.5 * (pos == neg).double().mean()).item()


def make_problem(n, positive_rate, separation):
    """Positives at +separation/2 along the diagonal, negatives at −separation/2, unit variance."""
    labels = (torch.arange(n) < round(n * positive_rate)).double()
    centre = (labels[:, None] - 0.5) * separation / math.sqrt(2)
    return centre + torch.randn(n, 2, dtype=torch.float64), labels


def logistic_scores(train, test, steps=400, lr=1.0):
    """Logistic regression by gradient descent on (X, y) = train, scoring the test inputs."""
    (X, y), X_test = train, test
    w = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y)
        gw, gb = torch.autograd.grad(loss, [w, b])
        with torch.no_grad():
            w -= lr * gw
            b -= lr * gb
    return torch.sigmoid(X_test @ w + b).detach()


def _self_check():
    torch.manual_seed(0)

    # The hand-worked case: positives 0.9, 0.8, 0.6; negatives 0.7, 0.55, 0.4.
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.55, 0.4], dtype=torch.float64)
    labels = torch.tensor([1, 1, 0, 1, 0, 0])
    assert confusion(scores, labels, 0.6) == {"tp": 3, "fp": 1, "fn": 0, "tn": 2}
    assert abs(rates(confusion(scores, labels, 0.6))["precision"] - 0.75) < 1e-12
    assert math.isnan(rates(confusion(scores, labels, 2.0))["precision"])
    assert abs(roc_auc(scores, labels) - 8 / 9) < 1e-12 and abs(mann_whitney(scores, labels) - 8 / 9) < 1e-12
    assert abs(average_precision(scores, labels) - 11 / 12) < 1e-12  # sklearn: 0.9167

    # AUC is the Mann–Whitney probability, ties included.
    y = (torch.rand(500) < 0.3).long()
    tied = torch.round((torch.rand(500, dtype=torch.float64) + 0.3 * y) * 10) / 10
    assert abs(roc_auc(tied, y) - mann_whitney(tied, y)) < 1e-12

    # Perfect and random classifiers.
    y = (torch.rand(20000) < 0.1).long()
    perfect = y + 0.5 * torch.rand(20000, dtype=torch.float64)
    assert abs(roc_auc(perfect, y) - 1) < 1e-12 and abs(average_precision(perfect, y) - 1) < 1e-12
    coin = torch.rand(20000, dtype=torch.float64)
    rate = y.double().mean().item()
    assert abs(roc_auc(coin, y) - 0.5) < 0.02 and abs(average_precision(coin, y) - rate) < 0.02

    # Imbalance: 50% vs 2% positives, same overlap (d′ = 2), logistic regression scores.
    results = {}
    for share in (0.5, 0.02):
        test_X, test_y = make_problem(4000, share, 2.0)
        s = logistic_scores(make_problem(4000, share, 2.0), test_X)
        results[share] = (roc_auc(s, test_y), average_precision(s, test_y), rates(confusion(s, test_y, 0.5))["accuracy"])
    (auc_b, ap_b, _), (auc_r, ap_r, acc_r) = results[0.5], results[0.02]
    assert abs(auc_b - auc_r) < 0.05, (auc_b, auc_r)
    assert ap_r < ap_b - 0.3, (ap_b, ap_r)
    assert abs(acc_r - 0.98) < 0.01, acc_r  # no better than always answering "negative"
    print(f"metrics.py: ok (hand case AUC 8/9, AP 11/12; d′ = 2: AUC {auc_b:.3f} → {auc_r:.3f}, "
          f"AP {ap_b:.3f} → {ap_r:.3f} from 50% to 2% positives, accuracy {acc_r:.3f} vs 0.980 for 'always negative')")


if __name__ == "__main__":
    _self_check()
