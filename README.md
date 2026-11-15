# Tensor Titans: skin lesion classification on DermaMNIST

Deep Learning challenge (SRH, winter 2026/27). Team **Tensor Titans**: Guna Prakash Kethineni,
Varun Kumar Sama, Panyala Shesha Shayana Reddy.

We classify 28 x 28 dermatoscopy images (DermaMNIST, 7 classes, 67 % harmless moles) and optimise
**balanced accuracy**. The final model (`model.pt`, TorchScript) is an ensemble of 5 small ResNet-style CNNs
with test-time augmentation.

| | balanced accuracy val | balanced accuracy test | macro-F1 test |
|---|---|---|---|
| starter baseline (3 seeds) | 0.416 ± 0.023 | 0.413 ± 0.029 | 0.44 ± 0.02 |
| **final model.pt** | 0.612 | **0.627** | 0.469 |

The baseline row uses the starter notebook's own code (`compare_baseline.py`). The ablation's A0 re-runs the
same baseline inside `train.py` and gets 0.427 ± 0.005 on val; the small difference comes from the order of
random draws.

Melanoma sensitivity on the test set: 0.23 (baseline) -> 0.67 (ours).

**Ablation: one change at a time** (val balanced accuracy, mean ± std over seeds 0-2, last epoch;
95 % bootstrap interval of the difference to A0):

| step | val balanced accuracy | 95 % interval vs. A0 |
|---|---|---|
| A0 starter baseline | 0.427 ± 0.005 | |
| A1 + logit-adjusted loss | 0.519 ± 0.044 | [0.048, 0.140] |
| A1' class-weighted CE (instead of A1) | 0.525 ± 0.035 | [0.053, 0.149] |
| A2 + 40 epochs, warm-up + cosine LR | 0.551 ± 0.016 | [0.073, 0.180] |
| A3 + data augmentation | 0.573 ± 0.010 | [0.090, 0.206] |
| A4 + ResNet-style CNN | 0.591 ± 0.021 | [0.103, 0.227] |
| A5 + test-time augmentation | 0.577 ± 0.014 | [0.090, 0.211] |
| final model.pt (5-network ensemble) | 0.612 (member epochs chosen on val) | [0.121, 0.252] |

Details and discussion: `Tensor_Titans_report.pdf`; numbers: `results/ablation.json`.

## Repository

| path | what |
|---|---|
| `train.py` | data loading (as in the starter), models, augmentation, losses, training, TensorBoard logging |
| `export.py` | builds the 5-network ensemble with test-time augmentation and exports `model.pt` (TorchScript), then reloads and checks it |
| `ablation.sh`, `ablation_report.py` | one change at a time, 3 seeds each, results table with 95 % bootstrap intervals vs. the baseline |
| `compare_baseline.py` | the starter baseline (unchanged) and `bootstrap_difference` |
| `evaluate_test.py` | report-only evaluation of the final model on the test split (never used for choices) |
| `make_figures.py` | figures of the report |
| `model.pt` | the submitted model: input normalised `(N, 3, 28, 28)` float, output logits `(N, 7)` |
| `app/` | Gradio app built on the course `app_template.py` (Colab, Hugging Face Spaces or local) |
| `Tensor_Titans_report.pdf` | the report (IEEE conference format) |
| `poster/` | poster (PDF) and the script that builds it |
| `results/` | numbers behind the tables and figures |
| `figures/` | figures made by `make_figures.py` |

## Reproduce

Python 3.10+.

```bash
pip install -r requirements.txt
```

**1. The final model** (CPU is enough: about 1 hour in total; a GPU is not needed)

```bash
python train.py --seeds 0 1 2 --epochs 40 --loss logit_adjusted --out runs/la40
python train.py --seeds 0     --epochs 40 --loss weighted_ce    --out runs/wce40
python train.py --seeds 0     --epochs 45 --width 64 --loss logit_adjusted --out runs/la_w64
python export.py --runs runs/la40 runs/wce40 runs/la_w64 --out model.pt
```

Defaults of `train.py` for these runs: small ResNet (width 32, one residual block per stage, dropout 0.2),
AdamW (lr 2e-3, weight decay 5e-4), batch 128, 2-epoch warm-up + cosine schedule, augmentation on,
epoch chosen by val balanced accuracy. `export.py` prints the val score and checks that the saved
TorchScript file gives the same predictions after reloading. Results can differ slightly between machines
(floating-point order on different CPUs / GPUs).

**2. The ablation** (one change at a time, 3 seeds each, about 40 minutes on CPU)

```bash
bash ablation.sh          # set PY=python3 if your interpreter is not called python
python ablation_report.py
```

**3. Test results and figures** (report only)

```bash
python evaluate_test.py --model model.pt
python make_figures.py
```

**4. Experiment tracking**

```bash
tensorboard --logdir runs/tensorboard
```

Data: `train.py` downloads DermaMNIST with `medmnist`, exactly like the starter notebook. Offline, it reads
`data/dermamnist.npz` (the official MedMNIST file). Training uses only the train split; all choices use the
val split; the test split is used only by `evaluate_test.py` to report the final numbers.

## App

See `app/README.md`. It uses `model.pt` from the repository root. In Colab, after cloning the repository:

```
!pip install -q gradio
!cd tensor-titans-dermamnist/app && python app.py
```

It prints a public `gradio.live` link.

## AI use

We designed the experimental methodology, implemented the core model architectures and training logic, and ran all evaluations ourselves, while using Claude (Anthropic) as a coding and writing assistant for supporting tasks. Specifically, we wrote the core PyTorch modules—including the residual CNN, logit-adjusted loss formulations, and ensemble logic—and used the AI assistant to help draft modular utility scripts for bootstrap resampling, matplotlib visualization, and the Gradio interface template, as well as to assist with debugging and polishing the LATEX report. Every AI-assisted code snippet and text draft was reviewed, modified, and verified by the team before integration.

## Data licence

DermaMNIST / MedMNIST v2 (Yang et al., Sci. Data 2023), based on HAM10000 (Tschandl et al., Sci. Data 2018),
CC BY-NC 4.0.
