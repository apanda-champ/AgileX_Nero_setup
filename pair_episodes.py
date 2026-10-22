#!/usr/bin/env python3
"""
Offline pairing of separately-recorded joint and camera episodes.

    python3 pair_episodes.py episodes/ --rate 30 --out aligned/

Reads   ep_NNNN_<stamp>_joints.h5  +  ep_NNNN_<stamp>_camera.h5
Writes  ep_NNNN_<stamp>_aligned.h5   (ALOHA-style layout)

HOW PAIRING WORKS
  Both files carry timestamps in seconds since the same t0_monotonic, because
  DepthAI device stamps are synced to the host steady clock. So no clock
  estimation is needed -- this resamples both streams onto one timebase.

    joints  ~50Hz  -> linearly interpolated onto the target grid
    camera  ~30Hz  -> NEAREST frame, never interpolated (you cannot average
                      two JPEGs into a real observation)

  The gripper defaults to nearest-neighbour too: interpolating across a grasp
  transition invents half-open states that never occurred.

  Every output row reports how far its camera frame actually was. If that
  exceeds --tolerance the row is dropped, so a dataset never silently contains
  an image paired with joints from a different moment.

OUTPUT LAYOUT
  /timestamps                        (T,)      seconds from t0
  /action                            (T, 8)    leader joints + gripper
  /observations/qpos                 (T, 8)    follower joints + gripper
  /observations/images/cam_high      (T,)      vlen uint8 JPEG
  /observations/depth_mm             (T, H, W) uint16, if present
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np

PAIR_RE = re.compile(r"^(ep_\d+_\d{8}_\d{6})_joints\.h5$")


def find_pairs(directory):
    pairs = []
    for jpath in sorted(glob.glob(os.path.join(directory, "ep_*_joints.h5"))):
        m = PAIR_RE.match(os.path.basename(jpath))
        if not m:
            continue
        cpath = os.path.join(directory, m.group(1) + "_camera.h5")
        pairs.append((m.group(1), jpath, cpath if os.path.exists(cpath) else None))
    return pairs


def nearest_index(grid, source):
    """For each t in grid, the index of the closest entry in source."""
    idx = np.searchsorted(source, grid)
    idx = np.clip(idx, 1, len(source) - 1)
    take_left = np.abs(grid - source[idx - 1]) < np.abs(grid - source[idx])
    idx[take_left] -= 1
    return idx


def interp_columns(grid, src_t, values):
    """Linear interpolation of an (N, D) array onto grid."""
    values = np.atleast_2d(values.T).T if values.ndim > 1 else values[:, None]
    out = np.empty((len(grid), values.shape[1]), dtype=np.float64)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(grid, src_t, values[:, c])
    return out


def pair_episode(name, jpath, cpath, rate, tolerance, outdir,
                 gripper_interp=False, compression="gzip", dry=False):
    with h5py.File(jpath, "r") as jf:
        jt = jf["timestamps"][:]
        lj = jf["leader/joint_angles"][:]
        lw = jf["leader/gripper"][:]
        has_follower = "follower" in jf
        fj = jf["follower/joint_angles"][:] if has_follower else None
        fw = jf["follower/gripper"][:] if has_follower else None
        j_t0 = jf.attrs.get("t0_monotonic", None)
        j_episode = int(jf.attrs.get("episode", -1))
        j_dropped = int(jf.attrs.get("dropped_samples", 0))

    if cpath is None:
        print(f"  {name}: NO CAMERA FILE — skipped")
        return None

    with h5py.File(cpath, "r") as cf:
        ct = cf["timestamps"][:]
        c_t0 = cf.attrs.get("t0_monotonic", None)
        has_depth = "depth_mm" in cf
        n_frames = len(ct)
        jpegs = [np.asarray(cf["color_jpeg"][i], dtype=np.uint8)
                 for i in range(n_frames)]
        depth = cf["depth_mm"][:] if has_depth else None

    if j_t0 is not None and c_t0 is not None and abs(float(j_t0) - float(c_t0)) > 1e-6:
        print(f"  {name}: t0 MISMATCH between files ({j_t0} vs {c_t0}) — skipped")
        return None
    if len(jt) < 2 or len(ct) < 2:
        print(f"  {name}: too few samples ({len(jt)} joints, {len(ct)} frames)")
        return None

    # Common timebase over the overlap only.
    lo = max(jt[0], ct[0])
    hi = min(jt[-1], ct[-1])
    if hi <= lo:
        print(f"  {name}: streams do not overlap — skipped")
        return None
    grid = np.arange(lo, hi, 1.0 / rate)

    cam_idx = nearest_index(grid, ct)
    cam_gap = np.abs(grid - ct[cam_idx])
    keep = cam_gap <= tolerance
    n_dropped_rows = int((~keep).sum())
    grid, cam_idx, cam_gap = grid[keep], cam_idx[keep], cam_gap[keep]
    if len(grid) == 0:
        print(f"  {name}: every row exceeded tolerance — skipped")
        return None

    # Joints: interpolate the arm, nearest for the gripper unless asked.
    action_j = interp_columns(grid, jt, lj)
    qpos_j = interp_columns(grid, jt, fj) if has_follower else None
    if gripper_interp:
        action_g = np.interp(grid, jt, lw)
        qpos_g = np.interp(grid, jt, fw) if has_follower else None
    else:
        ji = nearest_index(grid, jt)
        action_g = lw[ji]
        qpos_g = fw[ji] if has_follower else None

    action = np.concatenate([action_j, action_g[:, None]], axis=1)
    qpos = (np.concatenate([qpos_j, qpos_g[:, None]], axis=1)
            if has_follower else None)

    joint_gap = np.abs(grid - jt[nearest_index(grid, jt)])
    reused = len(cam_idx) - len(np.unique(cam_idx))

    print(f"  {name}: {len(grid)} rows @ {rate}Hz over {grid[-1] - grid[0]:.1f}s "
          f"| cam gap max {cam_gap.max() * 1000:.1f}ms mean "
          f"{cam_gap.mean() * 1000:.1f}ms | joint gap max "
          f"{joint_gap.max() * 1000:.1f}ms | {reused} frames reused"
          f"{f' | {n_dropped_rows} rows over tolerance' if n_dropped_rows else ''}")

    if dry:
        return None

    os.makedirs(outdir, exist_ok=True)
    opath = os.path.join(outdir, name + "_aligned.h5")
    kw = {"compression": compression} if compression else {}
    with h5py.File(opath, "w") as f:
        f.create_dataset("timestamps", data=grid, **kw)
        f.create_dataset("action", data=action.astype(np.float32), **kw)
        obs = f.create_group("observations")
        if qpos is not None:
            obs.create_dataset("qpos", data=qpos.astype(np.float32), **kw)
        imgs = obs.create_group("images")
        vlen = h5py.vlen_dtype(np.uint8)
        ds = imgs.create_dataset("cam_high", (len(cam_idx),), dtype=vlen)
        for row, src in enumerate(cam_idx):
            ds[row] = jpegs[src]
        if depth is not None:
            obs.create_dataset("depth_mm", data=depth[cam_idx], **kw)

        f.attrs["episode"] = j_episode
        f.attrs["rate_hz"] = rate
        f.attrs["n_timesteps"] = len(grid)
        f.attrs["duration_s"] = float(grid[-1] - grid[0])
        f.attrs["tolerance_s"] = tolerance
        f.attrs["cam_gap_max_s"] = float(cam_gap.max())
        f.attrs["cam_gap_mean_s"] = float(cam_gap.mean())
        f.attrs["joint_gap_max_s"] = float(joint_gap.max())
        f.attrs["frames_reused"] = reused
        f.attrs["rows_over_tolerance"] = n_dropped_rows
        f.attrs["joints_dropped_at_capture"] = j_dropped
        f.attrs["gripper_interpolated"] = bool(gripper_interp)
        f.attrs["source_joints"] = os.path.basename(jpath)
        f.attrs["source_camera"] = os.path.basename(cpath)
        f.attrs["layout"] = ("action = leader joints + gripper; "
                             "observations/qpos = follower joints + gripper; "
                             "images are JPEG bytes, cv2.imdecode to recover")
        f.attrs["units"] = "joints: rad; gripper: m; depth: mm; time: s"
    return opath


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="folder holding the recorded episodes")
    ap.add_argument("--rate", type=float, default=30.0,
                    help="output rate in Hz (default 30, matching the camera)")
    ap.add_argument("--tolerance", type=float, default=0.020,
                    help="max allowed camera-to-grid gap in seconds (default 0.020)")
    ap.add_argument("--out", default=None,
                    help="output folder (default <directory>/aligned)")
    ap.add_argument("--gripper-interp", action="store_true",
                    help="linearly interpolate gripper width instead of nearest")
    ap.add_argument("--no-compression", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="report timing quality without writing files")
    args = ap.parse_args()

    outdir = args.out or os.path.join(args.directory, "aligned")
    pairs = find_pairs(args.directory)
    if not pairs:
        raise SystemExit(f"No ep_*_joints.h5 found in {args.directory}")

    print(f"{len(pairs)} episode(s) in {args.directory}, target {args.rate:g}Hz, "
          f"tolerance {args.tolerance * 1000:.0f}ms\n")
    written = 0
    for name, jpath, cpath in pairs:
        try:
            if pair_episode(name, jpath, cpath, args.rate, args.tolerance, outdir,
                            gripper_interp=args.gripper_interp,
                            compression=None if args.no_compression else "gzip",
                            dry=args.dry_run):
                written += 1
        except Exception as exc:
            print(f"  {name}: FAILED {exc.__class__.__name__}: {exc}")

    if args.dry_run:
        print(f"\nDry run — nothing written. Check the gap columns above.")
    else:
        print(f"\n{written}/{len(pairs)} written to {os.path.abspath(outdir)}/")
        print("Camera gaps well under 17ms and joint gaps under 10ms means the "
              "pairing is tight.")


if __name__ == "__main__":
    main()
