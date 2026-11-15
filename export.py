"""Export the trained models as one TorchScript file model.pt.

model.pt: normalised images (N, 3, 28, 28) float -> logits (N, 7)   (starter notebook, section 4)

What is inside:
  * an ensemble of the trained seeds (logits averaged),
  * test-time augmentation: the logits are averaged over the 8 flips / 90 degree rotations
    (lesions have no fixed orientation), all inside forward(), so no extra code is needed.

Usage:
    python export.py --runs runs/final --out model.pt
"""
import argparse
import glob
import os
from typing import List

import torch
import torch.nn as nn

from train import build_model, load, metrics, predict_logits


class Exported(nn.Module):
    def __init__(self, models: List[nn.Module], tta: bool = True):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.tta = tta

    def _ensemble(self, x: torch.Tensor) -> torch.Tensor:
        outs: List[torch.Tensor] = []
        for m in self.models:
            outs.append(m(x))
        return torch.stack(outs).mean(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.tta:
            return self._ensemble(x)
        out = self._ensemble(x)
        out = out + self._ensemble(x.flip(-1))
        out = out + self._ensemble(x.flip(-2))
        out = out + self._ensemble(x.flip(-1).flip(-2))
        xt = x.transpose(-1, -2)
        out = out + self._ensemble(xt)
        out = out + self._ensemble(xt.flip(-1))
        out = out + self._ensemble(xt.flip(-2))
        out = out + self._ensemble(xt.flip(-1).flip(-2))
        return out / 8.0


def load_models(run_dirs):
    models = []
    for d in run_dirs:
        for f in sorted(glob.glob(os.path.join(d, "seed*.pt"))):
            ck = torch.load(f, map_location="cpu")
            a = ck["args"]
            m = build_model(argparse.Namespace(**a))
            m.load_state_dict(ck["state_dict"])
            m.eval()
            models.append(m)
            print(f"loaded {f}  (val balanced accuracy {ck['val']['balanced_accuracy']:.3f})")
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--tta", type=int, default=1)
    args = ap.parse_args()

    models = load_models(args.runs)
    model = Exported(models, tta=bool(args.tta)).eval()
    scripted = torch.jit.script(model)
    scripted.save(args.out)

    # check: load it again, like the evaluation script does (starter notebook, section 4)
    X_val, y_val = load("val")
    loaded = torch.jit.load(args.out)
    p_loaded = predict_logits(loaded, X_val).argmax(1)
    p_model = predict_logits(model, X_val).argmax(1)
    print("model.pt saved -", os.path.getsize(args.out) // 1024, "KB - same predictions after loading:",
          torch.equal(p_loaded, p_model))
    print("output shape for 2 images:", tuple(loaded(X_val[:2]).shape))
    m = metrics(y_val, p_loaded)
    print(f"val balanced accuracy {m['balanced_accuracy']:.3f}   macro-F1 {m['macro_F1']:.3f}")
    print("val sensitivity per class:", [round(s, 2) for s in m["sensitivity"]])


if __name__ == "__main__":
    main()
