"""Results table of the one-change-at-a-time ablation (run ablation.sh first).

For every step: balanced accuracy and macro-F1 on the val split, mean +- std over the 3 seeds
(last epoch, no epoch picking), and the 95 % bootstrap interval of the difference to the baseline A0.

Bootstrap: 2000 resamples of the 1003 val images (with replacement). In every resample we compute the
seed-averaged balanced accuracy of the step and of A0 on the same images, and take the difference.
We also report the starter notebook's bootstrap_difference for each seed pair (step seed s vs. A0 seed s).

Usage:  python ablation_report.py        -> results/ablation.json, results/ablation_table.tex, printed table
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from compare_baseline import bootstrap_difference
from train import load

STEPS = [
    ("A0_baseline", "A0 Starter baseline", "val_logits_seed{}.npy"),
    ("A1_logit_adj", "A1 + logit-adjusted loss", "val_logits_seed{}.npy"),
    ("A1w_weighted_ce", "A1' class-weighted CE instead (alternative)", "val_logits_seed{}.npy"),
    ("A2_cosine40", "A2 + 40 epochs, warm-up + cosine LR", "val_logits_seed{}.npy"),
    ("A3_augment", "A3 + data augmentation", "val_logits_seed{}.npy"),
    ("A4_resnet", "A4 + ResNet-style CNN", "val_logits_seed{}.npy"),
    ("A4_resnet", "A5 + test-time augmentation (8 views)", "val_logits_tta_seed{}.npy"),
]
SEEDS = [0, 1, 2]
N_DRAWS = 2000


def main():
    _, y = load("val")
    y = y.numpy()
    preds = {}
    for folder, name, pattern in STEPS:
        preds[name] = [np.load(os.path.join("runs/ablation", folder, pattern.format(s))).argmax(1) for s in SEEDS]

    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(y), size=(N_DRAWS, len(y)))
    base = preds[STEPS[0][1]]

    def boot_mean(p_list):
        return np.array([np.mean([balanced_accuracy_score(y[i], p[i]) for p in p_list]) for i in draws])

    base_boot = boot_mean(base)
    rows = []
    for _, name, _ in STEPS:
        p = preds[name]
        bacc = np.array([balanced_accuracy_score(y, q) for q in p])
        f1 = np.array([f1_score(y, q, average="macro") for q in p])
        row = {"step": name, "bacc_mean": float(bacc.mean()), "bacc_std": float(bacc.std(ddof=1)),
               "bacc_per_seed": bacc.round(4).tolist(),
               "f1_mean": float(f1.mean()), "f1_std": float(f1.std(ddof=1))}
        if name != STEPS[0][1]:
            diff = boot_mean(p) - base_boot
            row["diff_vs_A0"] = float(bacc.mean() - np.mean([balanced_accuracy_score(y, q) for q in base]))
            row["ci95_vs_A0"] = [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))]
            row["ci95_per_seed_pair"] = [bootstrap_difference(torch.tensor(y), torch.tensor(b), torch.tensor(q))
                                         for b, q in zip(base, p)]
        rows.append(row)

    # each ladder step vs. the step before it (A1' is an alternative to A1, so it is skipped)
    ladder = [n for _, n, _ in STEPS if not n.startswith("A1'")]
    boots = {n: boot_mean(preds[n]) for n in ladder}
    for prev, cur in zip(ladder[:-1], ladder[1:]):
        d = boots[cur] - boots[prev]
        r = next(r for r in rows if r["step"] == cur)
        r["ci95_vs_previous"] = [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]

    # final model.pt (ensemble of 5 + 8 views) vs. A0, same bootstrap
    if os.path.exists("model.pt"):
        X_val, _ = load("val")
        with torch.no_grad():
            final = torch.jit.load("model.pt")(X_val).argmax(1).numpy()
        fb = np.array([balanced_accuracy_score(y[i], final[i]) for i in draws]) - base_boot
        rows.append({"step": "Final model.pt (5-net ensemble + 8 views)",
                     "bacc_mean": float(balanced_accuracy_score(y, final)), "bacc_std": None,
                     "f1_mean": float(f1_score(y, final, average="macro")), "f1_std": None,
                     "diff_vs_A0": float(balanced_accuracy_score(y, final) - np.mean([balanced_accuracy_score(y, q) for q in base])),
                     "ci95_vs_A0": [float(np.quantile(fb, 0.025)), float(np.quantile(fb, 0.975))],
                     "note": "member epochs chosen on val, so the val number is optimistic"})

    os.makedirs("results", exist_ok=True)
    json.dump(rows, open("results/ablation.json", "w"), indent=2)

    print(f"{'step':48s} {'bal. acc. (val)':>16s} {'macro-F1':>14s} {'diff':>7s}  95% CI vs A0")
    tex = []
    for r in rows:
        ci = r.get("ci95_vs_A0")
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        d = f"{r['diff_vs_A0']:+.3f}" if ci else ""
        sd = f" +- {r['bacc_std']:.3f}" if r["bacc_std"] is not None else "        "
        fsd = f" +- {r['f1_std']:.3f}" if r["f1_std"] is not None else "        "
        pv = r.get("ci95_vs_previous")
        pvs = f"  prev: [{pv[0]:+.3f}, {pv[1]:+.3f}]" if pv else ""
        print(f"{r['step']:48s} {r['bacc_mean']:.3f}{sd}   {r['f1_mean']:.3f}{fsd} {d:>7s}  {cis}{pvs}")
        tex.append(f"{r['step']} & {r['bacc_mean']:.3f}" + (f" $\\pm$ {r['bacc_std']:.3f}" if r["bacc_std"] is not None else "")
                   + " & " + (f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "--") + " \\\\")
    open("results/ablation_table.tex", "w").write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
