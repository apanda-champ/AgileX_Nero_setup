#!/usr/bin/env python3
"""
Proprioception-only dataset for vanilla behaviour cloning.

No images. Input is the follower's measured state, target is the leader's
command at the same instant:

    qpos   (8,)  follower joints (7) + gripper  <- observation
    action (8,)  leader joints (7) + gripper    <- target

Reads EITHER file the rig produces:
    *_joints.h5   straight from nero_teleop_spacebar.py, 50Hz, no pairing needed
    *_aligned.h5  from pair_episodes.py, 30Hz

With no camera in the loop the raw _joints.h5 files are the simpler source and
give 50Hz instead of 30.

    python3 bc_dataset.py episodes/        # inspect and write stats.json
"""

import glob
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def list_episodes(directory):
    """Prefer *_joints.h5; fall back to *_aligned.h5."""
    joints = sorted(glob.glob(os.path.join(directory, "*_joints.h5")))
    if joints:
        return joints
    return sorted(glob.glob(os.path.join(directory, "*_aligned.h5")))


def read_episode(path):
    """Returns (qpos, action), both (T, 8) float32, for either file layout."""
    with h5py.File(path, "r") as f:
        if "action" in f:                                   # aligned layout
            action = f["action"][:].astype(np.float32)
            if "observations/qpos" not in f:
                raise ValueError(f"{path}: no observations/qpos")
            qpos = f["observations/qpos"][:].astype(np.float32)
        else:                                               # raw joints layout
            lj = f["leader/joint_angles"][:].astype(np.float32)
            lw = f["leader/gripper"][:].astype(np.float32)
            action = np.concatenate([lj, lw[:, None]], axis=1)
            if "follower" not in f:
                raise ValueError(f"{path}: no follower group — collection must "
                                 f"run with RECORD_FOLLOWER = True")
            fj = f["follower/joint_angles"][:].astype(np.float32)
            fw = f["follower/gripper"][:].astype(np.float32)
            qpos = np.concatenate([fj, fw[:, None]], axis=1)

    n = min(len(qpos), len(action))
    qpos, action = qpos[:n], action[:n]
    # NaNs come from follower reads that returned nothing at capture time.
    good = np.isfinite(qpos).all(axis=1) & np.isfinite(action).all(axis=1)
    if not good.all():
        qpos, action = qpos[good], action[good]
    return qpos, action


def compute_stats(paths):
    qs, acts = [], []
    for p in paths:
        q, a = read_episode(p)
        qs.append(q)
        acts.append(a)
    q = np.concatenate(qs, axis=0)
    a = np.concatenate(acts, axis=0)
    return {
        "action_mean": a.mean(0).tolist(),
        "action_std": np.clip(a.std(0), 1e-3, None).tolist(),
        "qpos_mean": q.mean(0).tolist(),
        "qpos_std": np.clip(q.std(0), 1e-3, None).tolist(),
        "n_episodes": len(paths),
        "n_timesteps": int(a.shape[0]),
        "dim": int(a.shape[1]),
    }


class ProprioBCDataset(Dataset):
    """One (qpos, action) pair per timestep. Held entirely in RAM — without
    images the whole dataset is a few megabytes."""

    def __init__(self, paths, stats):
        qs, acts = [], []
        for p in paths:
            q, a = read_episode(p)
            qs.append(q)
            acts.append(a)
        self.q = np.concatenate(qs, axis=0)
        self.a = np.concatenate(acts, axis=0)

        qm = np.asarray(stats["qpos_mean"], dtype=np.float32)
        qsd = np.asarray(stats["qpos_std"], dtype=np.float32)
        am = np.asarray(stats["action_mean"], dtype=np.float32)
        asd = np.asarray(stats["action_std"], dtype=np.float32)
        self.q = ((self.q - qm) / qsd).astype(np.float32)
        self.a = ((self.a - am) / asd).astype(np.float32)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        return torch.from_numpy(self.q[i]), torch.from_numpy(self.a[i])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = list_episodes(args.directory)
    if not paths:
        raise SystemExit(f"No *_joints.h5 or *_aligned.h5 in {args.directory}")

    kind = "joints" if paths[0].endswith("_joints.h5") else "aligned"
    print(f"{len(paths)} episode(s), {kind} layout")
    qs, acts, total = [], [], 0
    for p in paths:
        q, a = read_episode(p)
        qs.append(q)
        acts.append(a)
        total += len(q)
        print(f"  {os.path.basename(p):48} {len(q):5d} steps")
    print(f"  total {total} steps")

    stats = compute_stats(paths)
    out = args.out or os.path.join(args.directory, "stats.json")
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote {out}")
    print("  action_std:", np.round(stats["action_std"], 4))
    print("  qpos_std  :", np.round(stats["qpos_std"], 4))

    tiny = [i for i, s in enumerate(stats["action_std"]) if s < 5e-3]
    if tiny:
        print(f"  [warn] dims {tiny} barely vary across the dataset — there is "
              f"nothing for the policy to learn there.")

    # Without vision, qpos is the ONLY input. If the follower tracks the leader
    # closely, action is nearly equal to qpos and the task degenerates into
    # copying the input -- low loss, no useful behaviour.
    q = np.concatenate(qs, axis=0)
    a = np.concatenate(acts, axis=0)
    lag = np.abs(a - q).mean(0)
    print("  mean |action - qpos| per dim:", np.round(lag, 4))
    if lag[:7].mean() < 0.01:
        print("  [warn] leader and follower track almost perfectly, so "
              "action ~= qpos. A policy that learns the identity function will "
              "score a very low loss and do nothing useful. Compare any result "
              "against that baseline before believing it.")


if __name__ == "__main__":
    main()
