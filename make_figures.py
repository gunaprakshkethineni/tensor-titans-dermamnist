"""Figures for the report: example images per class, and per-class test sensitivity (baseline vs. final).

Needs data/dermamnist.npz (or run train.py once with medmnist) and results/test_results.json
(written by evaluate_test.py).   Usage:  python make_figures.py   -> figures/*.pdf
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = ["actinic\nkeratoses", "basal cell\ncarcinoma", "benign\nkeratosis", "dermato-\nfibroma",
         "melanoma", "nevi\n(moles)", "vascular\nlesions"]
COUNTS = [228, 359, 769, 80, 779, 4693, 99]
GREY, BLUE = "#b4b2aa", "#2a78d6"
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)

# ---- Fig. 1: one training image per class
d = np.load("data/dermamnist.npz")
X, y = d["train_images"], d["train_labels"].ravel()
fig, axes = plt.subplots(1, 7, figsize=(3.45, 0.95))
for c, ax in enumerate(axes):
    ax.imshow(X[np.where(y == c)[0][3]], interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(SHORT[c], fontsize=5.6, pad=2)
    ax.set_xlabel(f"n = {COUNTS[c]}", fontsize=5.6, labelpad=2)
fig.subplots_adjust(left=0.01, right=0.99, top=0.72, bottom=0.14, wspace=0.12)
fig.savefig("figures/examples.pdf")

# ---- Fig. 3: sensitivity per class on the test set
nums = json.load(open("results/test_results.json"))
base = np.mean([r["test"]["sensitivity"] for r in nums["baseline"]], axis=0)
base_sd = np.std([r["test"]["sensitivity"] for r in nums["baseline"]], axis=0, ddof=1)
ours = np.array(nums["model"]["test"]["sensitivity"])
fig, ax = plt.subplots(figsize=(3.45, 1.9))
idx = np.arange(7)
ax.barh(idx - 0.2, base, height=0.38, color=GREY, xerr=base_sd, error_kw={"lw": 0.6, "capsize": 1.5},
        label="baseline (3 seeds)")
ax.barh(idx + 0.2, ours, height=0.38, color=BLUE, label="final model")
for i, v in enumerate(ours):
    ax.text(v + 0.015, i + 0.2, f"{v:.2f}", va="center", fontsize=5.6)
ax.set_yticks(idx, [s.replace("-\n", "").replace("\n", " ") for s in SHORT])
ax.invert_yaxis()
ax.set_xlim(0, 1.08)
ax.set_xlabel("sensitivity on the test set (recall per class)")
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=5.6, frameon=False, handlelength=1.2)
ax.grid(axis="x", lw=0.3, color="#dddddd")
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig("figures/sensitivity_test.pdf")
print("figures written")
