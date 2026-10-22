#!/usr/bin/env python3
"""
Play back a recorded camera episode.

    python3 view_episode.py episodes/ep_0003_20260921_141500_camera.h5
    python3 view_episode.py episodes/ep_0003_..._camera.h5 --depth
    python3 view_episode.py episodes/ep_0003_..._camera.h5 --export out.mp4

  SPACE  pause / resume        , .  step one frame while paused
  [ ]    slower / faster       r    restart
  q      quit

Playback follows the RECORDED timestamps, not a nominal frame rate, so a
stretch where frames were dropped plays back as the pause it actually was.
That is the point: a smooth-looking 30fps replay would hide exactly the
problem you would want to see.
"""

import argparse
import os

import cv2
import h5py
import numpy as np


def colourise_depth(depth_mm, max_mm=2000):
    d = np.clip(depth_mm.astype(np.float32), 0, max_mm)
    vis = cv2.applyColorMap((d / max_mm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[depth_mm == 0] = 0                 # invalid pixels stay black
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="an *_camera.h5 file")
    ap.add_argument("--depth", action="store_true",
                    help="show depth beside the colour frame")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--export", default=None, metavar="OUT.mp4",
                    help="write an mp4 instead of displaying")
    ap.add_argument("--depth-max", type=float, default=2000,
                    help="depth colour scale ceiling in mm (default 2000)")
    args = ap.parse_args()

    with h5py.File(args.path, "r") as f:
        if "color_jpeg" not in f:
            raise SystemExit(f"{args.path} has no color_jpeg — is this a "
                             f"_camera.h5 file?")
        ts = f["timestamps"][:]
        n = len(ts)
        has_depth = "depth_mm" in f
        print(f"{os.path.basename(args.path)}")
        print(f"  {n} frames, {ts[-1] - ts[0]:.1f}s, "
              f"{n / max(ts[-1] - ts[0], 1e-9):.1f} fps mean")
        for k in ("episode", "start_wall_time", "jpeg_quality", "rgb_size"):
            if k in f.attrs:
                print(f"  {k}: {f.attrs[k]}")
        if args.depth and not has_depth:
            print("  [warn] no depth in this file — colour only")

        gaps = np.diff(ts)
        if len(gaps):
            print(f"  frame gap: min {gaps.min()*1000:.1f}ms  "
                  f"median {np.median(gaps)*1000:.1f}ms  "
                  f"max {gaps.max()*1000:.1f}ms")
            stalls = int((gaps > 3 * np.median(gaps)).sum())
            if stalls:
                print(f"  [warn] {stalls} gap(s) over 3x the median — "
                      f"frames were dropped during recording")

        frames = [np.asarray(f["color_jpeg"][i], dtype=np.uint8) for i in range(n)]
        depth = f["depth_mm"][:] if (args.depth and has_depth) else None

    def render(i):
        img = cv2.imdecode(frames[i], cv2.IMREAD_COLOR)
        if depth is not None:
            img = np.hstack([img, colourise_depth(depth[i], args.depth_max)])
        cv2.putText(img, f"{i+1}/{n}  t={ts[i]-ts[0]:6.2f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, f"{i+1}/{n}  t={ts[i]-ts[0]:6.2f}s",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return img

    # ---- export ------------------------------------------------------------
    if args.export:
        first = render(0)
        h, w = first.shape[:2]
        fps = n / max(ts[-1] - ts[0], 1e-9)
        writer = cv2.VideoWriter(args.export,
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"could not open {args.export} for writing")
        for i in range(n):
            writer.write(render(i))
        writer.release()
        print(f"\nwrote {args.export}  ({w}x{h}, {fps:.1f}fps, {n} frames)")
        print("  Note: mp4 is fixed-rate, so recording stalls are smoothed away.")
        return

    # ---- interactive -------------------------------------------------------
    print("\n  SPACE pause   , . step   [ ] speed   r restart   q quit\n")
    speed = args.speed
    paused = False
    i = 0
    while True:
        cv2.imshow("episode", render(i))
        if paused:
            wait = 0
        else:
            gap = (ts[i + 1] - ts[i]) if i + 1 < n else 0.033
            wait = max(1, int(gap * 1000 / max(speed, 1e-3)))
        key = cv2.waitKey(wait) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            i = 0
        elif key == ord("."):
            paused, i = True, min(i + 1, n - 1)
        elif key == ord(","):
            paused, i = True, max(i - 1, 0)
        elif key == ord("["):
            speed = max(0.1, speed / 1.5)
            print(f"  speed x{speed:.2f}")
        elif key == ord("]"):
            speed = min(16.0, speed * 1.5)
            print(f"  speed x{speed:.2f}")
        elif not paused:
            i += 1
            if i >= n:
                i = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
