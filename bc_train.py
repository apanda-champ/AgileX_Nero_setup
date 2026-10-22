#!/usr/bin/env python3
"""
Vanilla behaviour cloning on proprioception only. MLP, MSE loss, single-step.

    python3 bc_train.py episodes/ --epochs 300

  observation : follower qpos (7 joints + gripper)
  target      : leader action (7 joints + gripper), same timestep
  loss        : MSE

Two things that are not stylistic choices:

  * Validation is held out BY EPISODE. Splitting by timestep leaks, because
    frame t and frame t+1 are nearly identical -- you would get a beautiful
    validation curve that means nothing.
  * The identity baseline is printed alongside the loss. With no vision, if the
    follower tracks the leader closely then predicting "action = qpos" already
    scores well. A policy that does not beat that number has learned nothing,
    however low its loss looks.
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bc_dataset import ProprioBCDataset, compute_stats, list_episodes


class BCPolicy(nn.Module):
    """qpos -> action. Plain MLP; there is nothing to encode without images."""

    def __init__(self, qpos_dim=8, action_dim=8, hidden=1024, layers=2,
                 dropout=0.1):
        super().__init__()
        blocks, d = [], qpos_dim
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        blocks += [nn.Linear(d, action_dim)]
        self.net = nn.Sequential(*blocks)

    def forward(self, qpos):
        return self.net(qpos)


def split_episodes(paths, val_frac, seed):
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac)) if len(shuffled) > 1 else 0
    return shuffled[n_val:], shuffled[:n_val]


def run_epoch(model, loader, device, optimiser=None):
    train = optimiser is not None
    model.train(train)
    total, count = 0.0, 0
    for qpos, action in loader:
        qpos = qpos.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            loss = ((model(qpos) - action) ** 2).mean()      # MSE
        if train:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        total += loss.item() * qpos.size(0)
        count += qpos.size(0)
    return total / max(count, 1)


def identity_baseline(dataset):
    """MSE of predicting the (normalised) observation as the action.

    Kept in the SAME units as the training loss so the two are comparable."""
    return float(((dataset.a - dataset.q) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--rate", type=float, default=None,
                    help="nominal data rate in Hz, stored in the checkpoint "
                         "(default 50 for _joints.h5, 30 for _aligned.h5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    paths = list_episodes(args.directory)
    if not paths:
        raise SystemExit(f"No *_joints.h5 or *_aligned.h5 in {args.directory}")
    rate = args.rate or (50.0 if paths[0].endswith("_joints.h5") else 30.0)

    train_paths, val_paths = split_episodes(paths, args.val_frac, args.seed)
    print(f"{len(train_paths)} train episode(s), {len(val_paths)} val episode(s), "
          f"nominal {rate:g}Hz")
    if len(train_paths) < 10:
        print("[warn] fewer than 10 training episodes — expect poor "
              "generalisation whatever the loss does.")

    # Stats from the TRAINING split only; including val leaks its distribution.
    stats = compute_stats(train_paths)
    train_ds = ProprioBCDataset(train_paths, stats)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          drop_last=len(train_ds) > args.batch)
    val_ds = ProprioBCDataset(val_paths, stats) if val_paths else None
    val_ld = (DataLoader(val_ds, batch_size=args.batch, shuffle=False)
              if val_ds else None)

    base_tr = identity_baseline(train_ds)
    base_va = identity_baseline(val_ds) if val_ds else None
    print(f"identity baseline (MSE) — train {base_tr:.6f}"
          + (f", val {base_va:.6f}" if base_va is not None else ""))
    print("Beat that number, or the policy has learned nothing.\n")

    dim = stats["dim"]
    model = BCPolicy(qpos_dim=dim, action_dim=dim, hidden=args.hidden,
                     layers=args.layers, dropout=args.dropout).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    os.makedirs(args.out, exist_ok=True)
    config = {"obs": "proprio_only", "action_dim": dim, "qpos_dim": dim,
              "hidden": args.hidden, "layers": args.layers,
              "dropout": args.dropout, "rate_hz": rate, "loss": "mse"}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_ld, device, optimiser)
        va = run_epoch(model, val_ld, device) if val_ld else None
        sched.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            line = f"epoch {epoch:4d}/{args.epochs}  train {tr:.4f}"
            if va is not None:
                line += f"  val {va:.4f}"
            line += f"  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.1f}s"
            print(line)

        score = va if va is not None else tr
        ckpt = {"model": model.state_dict(), "stats": stats, "config": config,
                "epoch": epoch, "train_loss": tr, "val_loss": va,
                "identity_baseline_train": base_tr,
                "identity_baseline_val": base_va}
        torch.save(ckpt, os.path.join(args.out, "last.pt"))
        if score < best:
            best = score
            torch.save(ckpt, os.path.join(args.out, "best.pt"))

    ref = base_va if base_va is not None else base_tr
    print(f"\nbest {'val' if val_paths else 'train'} MSE {best:.6f}  "
          f"(identity baseline {ref:.6f})")
    if best >= ref:
        print("The policy did NOT beat the identity baseline. It has not "
              "learned the task — do not deploy it expecting behaviour.")
    print(f"checkpoints in {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
