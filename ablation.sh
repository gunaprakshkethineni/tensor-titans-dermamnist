#!/usr/bin/env bash
# One change at a time, 3 seeds each, evaluated on the val split at the LAST epoch (no epoch picking).
# Each step adds exactly one improvement to the step before it.
#   A0  starter baseline (starter CNN, cross-entropy, Adam 1e-3, 10 epochs, batch 64)
#   A1  + logit-adjusted loss                     (alternative A1w: class-weighted cross-entropy)
#   A2  + longer training: 40 epochs with warm-up + cosine learning-rate schedule
#   A3  + data augmentation (flips / 90 degree rotations, shifts, brightness / contrast)
#   A4  + ResNet-style CNN (residual blocks, BatchNorm, dropout) instead of the starter CNN
#   A5  + test-time augmentation (8 views) - evaluation only, computed from A4 by ablation_report.py
# Then: python ablation_report.py   (table, mean +- std, 95 % bootstrap interval vs. A0)
set -e
PY=${PY:-python}
COMMON="--seeds 0 1 2 --select last --batch_size 64 --lr 1e-3 --optim adam --tb runs/tensorboard"

$PY train.py $COMMON --out runs/ablation/A0_baseline     --arch starter --loss ce             --sched const  --epochs 10 --augment 0
$PY train.py $COMMON --out runs/ablation/A1_logit_adj    --arch starter --loss logit_adjusted --sched const  --epochs 10 --augment 0
$PY train.py $COMMON --out runs/ablation/A1w_weighted_ce --arch starter --loss weighted_ce    --sched const  --epochs 10 --augment 0
$PY train.py $COMMON --out runs/ablation/A2_cosine40     --arch starter --loss logit_adjusted --sched cosine --epochs 40 --augment 0
$PY train.py $COMMON --out runs/ablation/A3_augment      --arch starter --loss logit_adjusted --sched cosine --epochs 40 --augment 1
$PY train.py $COMMON --out runs/ablation/A4_resnet       --arch resnet  --loss logit_adjusted --sched cosine --epochs 40 --augment 1
