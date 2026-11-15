"""Report-only evaluation on the official TEST split (allowed for final models, never used for choices).

Evaluates the frozen model.pt and the starter baseline (seeds 0, 1, 2) on test, and also collects
val per-class sensitivities for the poster figure. Writes results/test_results.json.

Usage:  python evaluate_test.py --model model.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from compare_baseline import bootstrap_difference, train_baseline
from train import load, metrics, predict_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    args = ap.parse_args()
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    X_test, y_test = load("test")

    final = torch.jit.load(args.model)
    out = {"model": {"val": metrics(y_val, predict_logits(final, X_val).argmax(1))}}
    pred_test = predict_logits(final, X_test).argmax(1)
    out["model"]["test"] = metrics(y_test, pred_test)
    out["model"]["test"]["accuracy"] = float((pred_test == y_test).float().mean())

    out["baseline"] = []
    for seed in [0, 1, 2]:
        base = train_baseline(X_train, y_train, seed)
        pv = predict_logits(base, X_val).argmax(1)
        pt = predict_logits(base, X_test).argmax(1)
        r = {"seed": seed, "val": metrics(y_val, pv), "test": metrics(y_test, pt)}
        r["test"]["accuracy"] = float((pt == y_test).float().mean())
        r["test_ci_model_minus_baseline"] = bootstrap_difference(y_test, pt, pred_test)
        out["baseline"].append(r)
        print(f"baseline seed {seed}: val {r['val']['balanced_accuracy']:.3f}  test {r['test']['balanced_accuracy']:.3f}"
              f"  test CI(model - baseline) {r['test_ci_model_minus_baseline']}")

    m = out["model"]
    print(f"model.pt  val bacc {m['val']['balanced_accuracy']:.3f}  test bacc {m['test']['balanced_accuracy']:.3f}"
          f"  test macro-F1 {m['test']['macro_F1']:.3f}  test acc {m['test']['accuracy']:.3f}")
    print("model.pt test sensitivity:", [round(s, 2) for s in m["test"]["sensitivity"]])
    b = np.array([r["test"]["balanced_accuracy"] for r in out["baseline"]])
    print(f"baseline test bacc {b.mean():.3f} +- {b.std(ddof=1):.3f}")
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/test_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
