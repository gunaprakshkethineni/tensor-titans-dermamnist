"""DermaMNIST challenge: train an improved CNN and export model.pt (TorchScript).

Keeps the input format of the starter notebook:
images (N, 3, 28, 28), values 0...1, normalised with the starter MEAN / STD.
model.pt maps normalised images (N, 3, 28, 28) -> logits (N, 7).

Rules respected: trains on the train split only, chooses settings on the val split only.
The test split is never loaded here.

Usage:
    python train.py --seeds 0 1 2 --epochs 40 --loss logit_adjusted --out runs/la40
    python export.py --runs runs/la40 --out model.pt
(the full commands for the submitted model and for the ablation are in README.md)
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

# exactly as in day4_challenge_starter.ipynb - do not change
MEAN = [0.7631, 0.5381, 0.5614]
STD = [0.1366, 0.1543, 0.1692]
CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]
NUM_CLASSES = len(CLASSES)
_MEAN_T = torch.tensor(MEAN).view(1, 3, 1, 1)
_STD_T = torch.tensor(STD).view(1, 3, 1, 1)


# --------------------------------------------------------------------------- data
def load(split, root="data"):
    """DermaMNIST split as tensors like the starter: images (N, 3, 28, 28) normalised, labels (N,)."""
    try:                                                    # same loader as the starter notebook
        from medmnist import DermaMNIST
        os.makedirs(root, exist_ok=True)
        d = DermaMNIST(split=split, download=True, size=28, root=root)
        imgs, labels = d.imgs, d.labels
    except ImportError:                                     # offline fallback: the official dermamnist.npz
        d = np.load(os.path.join(root, "dermamnist.npz"))
        imgs, labels = d[f"{split}_images"], d[f"{split}_labels"]
    X = torch.tensor(imgs).permute(0, 3, 1, 2).float() / 255
    y = torch.tensor(labels).reshape(-1).long()
    return (X - _MEAN_T) / _STD_T, y


def augment(x):
    """Batch augmentation on normalised images (a different random draw for every image).

    Dermoscopy images have no canonical orientation, so all 8 flips/rot90s are valid.
    Plus a random shift (reflect pad 3 + crop 28) and mild brightness/contrast jitter.
    """
    n = x.shape[0]
    # flips and 90 degree rotations (dihedral group D4)
    f = torch.rand(n) < 0.5
    x = torch.where(f[:, None, None, None], x.flip(-1), x)
    f = torch.rand(n) < 0.5
    x = torch.where(f[:, None, None, None], x.flip(-2), x)
    f = torch.rand(n) < 0.5
    x = torch.where(f[:, None, None, None], x.transpose(-1, -2), x)
    # random shift by up to 3 pixels
    p = 3
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    dx = torch.randint(0, 2 * p + 1, (n,))
    dy = torch.randint(0, 2 * p + 1, (n,))
    rows = (dy[:, None] + torch.arange(28)[None, :])                       # (n, 28)
    cols = (dx[:, None] + torch.arange(28)[None, :])
    idx_n = torch.arange(n)[:, None, None, None]
    idx_c = torch.arange(3)[None, :, None, None]
    x = xp[idx_n, idx_c, rows[:, None, :, None], cols[:, None, None, :]]
    # brightness / contrast jitter in 0...1 space, then normalise again
    raw = x * _STD_T + _MEAN_T
    b = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.3                             # brightness 0.85...1.15
    c = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.3                             # contrast  0.85...1.15
    m = raw.mean(dim=(1, 2, 3), keepdim=True)
    raw = ((raw - m) * c + m) * b
    return (raw.clamp(0, 1) - _MEAN_T) / _STD_T


# --------------------------------------------------------------------------- model
class BasicBlock(nn.Module):
    """ResNet basic block (He et al., CVPR 2016) with BatchNorm."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


class SmallResNet(nn.Module):
    """ResNet for 28 x 28 inputs: stem 28x28 -> stages at 28, 14, 7 -> global average pool -> 7 logits."""

    def __init__(self, width: int = 32, blocks=(1, 1, 1), dropout: float = 0.2, num_classes: int = NUM_CLASSES):
        super().__init__()
        w = [width, 2 * width, 4 * width]
        self.stem = nn.Sequential(nn.Conv2d(3, w[0], 3, 1, 1, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU())
        layers = []
        cin = w[0]
        for i, (cout, nb) in enumerate(zip(w, blocks)):
            for j in range(nb):
                layers.append(BasicBlock(cin, cout, 2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)

    def forward(self, x):
        x = self.features(self.stem(x))
        x = torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)
        return self.fc(self.drop(x))


def StarterCNN(num_classes: int = NUM_CLASSES):
    """The baseline CNN of day4_challenge_starter.ipynb, unchanged."""
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, num_classes))


def build_model(args):
    if getattr(args, "arch", "resnet") == "starter":
        return StarterCNN()
    return SmallResNet(width=args.width, blocks=tuple(args.blocks), dropout=args.dropout)


# --------------------------------------------------------------------------- training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_logits(model, X, batch_size=500, tta=False):
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        x = X[i:i + batch_size]
        if tta:   # average over the 8 dihedral views
            views = [x, x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)]
            views += [v.transpose(-1, -2) for v in views]
            outs.append(torch.stack([model(v) for v in views]).mean(0))
        else:
            outs.append(model(x))
    return torch.cat(outs)


def metrics(y_true, y_pred):
    return {"balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_F1": float(f1_score(y_true, y_pred, average="macro")),
            "sensitivity": [float(s) for s in recall_score(y_true, y_pred, average=None,
                                                           labels=range(NUM_CLASSES), zero_division=0)]}


def train_one(seed, args, X_train, y_train, X_val, y_val, log):
    set_seed(seed)
    model = build_model(args)
    writer = None
    if args.tb:                                              # experiment tracking (TensorBoard)
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.tb, f"{os.path.basename(args.out.rstrip('/'))}_seed{seed}"))
        except ImportError:
            log("tensorboard not installed - no tracking")
    counts = torch.bincount(y_train, minlength=NUM_CLASSES).float()
    prior = counts / counts.sum()

    # class imbalance: logit-adjusted loss (Menon et al., ICLR 2021 "Long-tail learning via logit adjustment")
    # = balanced softmax (Ren et al., NeurIPS 2020): add tau * log(prior) to the logits during training only,
    # so the plain logits at test time estimate the *balanced* posterior -> good balanced accuracy.
    la_bias = args.tau * torch.log(prior) if args.loss == "logit_adjusted" else torch.zeros(NUM_CLASSES)
    weight = (counts.sum() / (NUM_CLASSES * counts)) if args.loss == "weighted_ce" else None

    if args.optim == "adam":                                 # the starter: Adam, constant learning rate
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:                                                    # AdamW (decoupled weight decay)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(X_train) / args.batch_size)
    total = args.epochs * steps_per_epoch
    warm = min(steps_per_epoch * 2, total // 4)                # linear warm-up, then cosine decay
    if args.sched == "cosine":
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm
            else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)

    best = {"balanced_accuracy": -1}
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(X_train))
        tot_loss = 0.0
        for i in range(0, len(X_train), args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, y = X_train[idx], y_train[idx]
            if args.augment:
                x = augment(x)
            logits = model(x) + la_bias
            loss = F.cross_entropy(logits, y, weight=weight, label_smoothing=args.label_smoothing)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
        # model selection on val only (early stopping on val balanced accuracy, evaluated every epoch)
        pred = predict_logits(model, X_val).argmax(1)
        m = metrics(y_val, pred)
        if m["balanced_accuracy"] > best["balanced_accuracy"]:
            best = dict(m, epoch=epoch + 1)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if writer is not None:
            writer.add_scalar("train/loss", tot_loss / len(X_train), epoch + 1)
            writer.add_scalar("val/balanced_accuracy", m["balanced_accuracy"], epoch + 1)
            writer.add_scalar("val/macro_F1", m["macro_F1"], epoch + 1)
            writer.add_scalar("train/learning_rate", opt.param_groups[0]["lr"], epoch + 1)
        log(f"seed {seed} epoch {epoch + 1:3d} loss {tot_loss / len(X_train):.3f} "
            f"val_bacc {m['balanced_accuracy']:.3f} best {best['balanced_accuracy']:.3f}@{best['epoch']} "
            f"({time.time() - t0:.1f}s)")
    if args.select == "last":
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        best = dict(metrics(y_val, predict_logits(model, X_val).argmax(1)), epoch=args.epochs)
    model.load_state_dict(best_state)
    if writer is not None:
        writer.close()
    return model, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--arch", choices=["starter", "resnet"], default="resnet")
    ap.add_argument("--optim", choices=["adam", "adamw"], default="adamw")
    ap.add_argument("--sched", choices=["const", "cosine"], default="cosine")
    ap.add_argument("--tb", default="runs/tensorboard", help="TensorBoard log folder ('' = off)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--blocks", type=int, nargs=3, default=[1, 1, 1])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss", choices=["ce", "weighted_ce", "logit_adjusted"], default="logit_adjusted")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--select", choices=["best", "last"], default="best")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="runs/exp")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log(json.dumps(vars(args)))
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    log(f"train {tuple(X_train.shape)} val {tuple(X_val.shape)}")

    results = []
    for seed in args.seeds:
        model, best = train_one(seed, args, X_train, y_train, X_val, y_val, log)
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "seed": seed, "val": best},
                   os.path.join(args.out, f"seed{seed}.pt"))
        np.save(os.path.join(args.out, f"val_logits_seed{seed}.npy"), predict_logits(model, X_val).numpy())
        np.save(os.path.join(args.out, f"val_logits_tta_seed{seed}.npy"), predict_logits(model, X_val, tta=True).numpy())
        results.append(best["balanced_accuracy"])
        log(f"seed {seed} best val balanced accuracy {best['balanced_accuracy']:.3f} at epoch {best['epoch']}")
    r = np.array(results)
    summary = {"val_bacc_per_seed": results, "mean": float(r.mean()), "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0}
    log(json.dumps(summary))
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
