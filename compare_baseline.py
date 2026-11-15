"""Validation comparison: starter baseline (3 seeds) vs. the exported model.pt.

Baseline = the starter CNN and training loop unchanged (Adam 1e-3, 10 epochs, batch 64).
Reports mean +- std of val balanced accuracy and the 95 % bootstrap interval of the
difference (same method as bootstrap_difference in the starter notebook).

Usage:  python compare_baseline.py --model model.pt
"""
import argparse
import json

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from train import NUM_CLASSES, load, metrics, predict_logits


def make_baseline():
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, NUM_CLASSES))


def train_baseline(X_train, y_train, seed, epochs=10, lr=1e-3):
    torch.manual_seed(seed)
    model = make_baseline()
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def bootstrap_difference(y_true, pred_a, pred_b, n_draws=2000, seed=0):
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_draws):
        idx = torch.randint(0, len(y_true), (len(y_true),), generator=g)
        diffs.append(balanced_accuracy_score(y_true[idx], pred_b[idx])
                     - balanced_accuracy_score(y_true[idx], pred_a[idx]))
    low, high = torch.quantile(torch.tensor(diffs), torch.tensor([0.025, 0.975]))
    return round(low.item(), 3), round(high.item(), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    args = ap.parse_args()
    X_train, y_train = load("train")
    X_val, y_val = load("val")

    final = torch.jit.load(args.model)
    pred_b = predict_logits(final, X_val).argmax(1)
    mb = metrics(y_val, pred_b)
    print(f"model.pt: val balanced accuracy {mb['balanced_accuracy']:.3f}  macro-F1 {mb['macro_F1']:.3f}")
    print("  sensitivity per class:", [round(s, 2) for s in mb["sensitivity"]])

    res = {"model": mb, "baseline": []}
    for seed in [0, 1, 2]:
        pred_a = predict_logits(train_baseline(X_train, y_train, seed), X_val).argmax(1)
        ma = metrics(y_val, pred_a)
        ci = bootstrap_difference(y_val, pred_a, pred_b)
        res["baseline"].append(dict(ma, seed=seed, ci_model_minus_baseline=ci))
        print(f"baseline seed {seed}: bacc {ma['balanced_accuracy']:.3f}  macro-F1 {ma['macro_F1']:.3f}  "
              f"95% interval of (model.pt - baseline): {ci}")
    a = torch.tensor([r["balanced_accuracy"] for r in res["baseline"]])
    print(f"baseline: {a.mean():.3f} +- {a.std():.3f}")
    json.dump(res, open("compare_baseline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
