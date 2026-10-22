#!/usr/bin/env python3
"""
Run a proprioception-only BC policy on the follower arm. No camera.

    python3 bc_deploy.py checkpoints/best.pt --dry-run
    python3 bc_deploy.py checkpoints/best.pt

  SPACE  start / pause the rollout
  x      EMERGENCY STOP — halts commanding immediately, holds the arm
  q      quit

THE LEADER IS NOT USED. The policy outputs what the leader would have
commanded, and that goes to the follower over the same move_js path teleop
uses.

SAFETY. A BC policy with no vision has no idea whether the scene matches what
it was trained on, and it will command confidently regardless. Four gates:

  1. MAX_STEP_RAD   -- per-tick joint delta is clamped. At the deploy rate this
                       caps velocity no matter what the network emits, and it
                       is the single most effective limit here.
  2. START_TOLERANCE-- refuses to start unless the arm is already near the
                       first predicted action, so it cannot lunge on tick one.
  3. MAX_RUNTIME_S  -- rollouts stop on their own.
  4. x key          -- stops commanding within one tick.

These limit speed. They do not make a bad policy safe. Keep a hand on the
e-stop for the first rollouts.
"""

import argparse
import select
import sys
import termios
import time
import tty

import numpy as np
import torch

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

try:
    from pyAgxArm import resolve_firmware_profile
except ImportError:
    resolve_firmware_profile = None

from bc_train import BCPolicy

# ----------------------------------------------------------------- safety ---
MAX_STEP_RAD      = 0.03      # per joint per tick
MAX_GRIPPER_STEP  = 0.010     # metres per tick
START_TOLERANCE   = 0.15      # rad; refuse to start if further than this
MAX_RUNTIME_S     = 60.0

FOLLOWER_CHANNEL  = "can0"
FALLBACK_PROFILE  = NeroFW.V121
CONNECT_SETTLE_S  = 2.0
FOLLOWER_GRIP_FORCE = 1.0
GRIPPER_MIN_M     = 0.000
GRIPPER_MAX_M     = 0.100


def _try(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as exc:
        print(f"[warn] {label}: {exc.__class__.__name__}: {exc}")
        return None


def unwrap(obj):
    return getattr(obj, "msg", None) if obj is not None else None


class Keyboard:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.saved = None
        self.ok = sys.stdin.isatty()

    def __enter__(self):
        if self.ok:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            print("[warn] stdin is not a terminal — keys disabled.")
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self):
        if self.ok and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


def detect_profile(channel):
    probe = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT,
        interface="socketcan", channel=channel))
    ver = None
    try:
        probe.connect()
        time.sleep(CONNECT_SETTLE_S)
        fw = probe.get_firmware()
        ver = fw.get("software_version") if isinstance(fw, dict) else str(fw)
    except Exception as exc:
        print(f"[warn] firmware probe: {exc}")
    finally:
        _try("probe disconnect", probe.disconnect)
    if ver and resolve_firmware_profile:
        try:
            prof = resolve_firmware_profile(ArmModel.NERO, ver)
            print(f"firmware {ver} -> profile {prof!r}")
            return prof
        except Exception:
            pass
    return FALLBACK_PROFILE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--channel", default=FOLLOWER_CHANNEL)
    ap.add_argument("--rate", type=float, default=None,
                    help="deploy rate in Hz (default: the rate it was trained at)")
    ap.add_argument("--max-step", type=float, default=MAX_STEP_RAD)
    ap.add_argument("--runtime", type=float, default=MAX_RUNTIME_S)
    ap.add_argument("--dry-run", action="store_true",
                    help="predict and print, never command the arm")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg, stats = ckpt["config"], ckpt["stats"]
    a_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    a_std = np.asarray(stats["action_std"], dtype=np.float32)
    q_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    q_std = np.asarray(stats["qpos_std"], dtype=np.float32)
    rate = args.rate or cfg.get("rate_hz", 30.0)

    model = BCPolicy(qpos_dim=cfg["qpos_dim"], action_dim=cfg["action_dim"],
                     hidden=cfg.get("hidden", 1024), layers=cfg.get("layers", 2),
                     dropout=cfg.get("dropout", 0.1)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"loaded {args.checkpoint}  epoch {ckpt['epoch']}  "
          f"val {ckpt.get('val_loss')}  device {device}")
    base = ckpt.get("identity_baseline_val") or ckpt.get("identity_baseline_train")
    score = ckpt.get("val_loss") or ckpt.get("train_loss")
    if base is not None and score is not None and score >= base:
        print(f"[warn] this checkpoint ({score:.4f}) did not beat the identity "
              f"baseline ({base:.4f}). It has not learned the task.")

    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=detect_profile(args.channel),
        interface="socketcan", channel=args.channel))
    arm.connect()
    time.sleep(CONNECT_SETTLE_S)
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

    print("Enabling in 3 s. CLEAR THE WORKSPACE.")
    time.sleep(3)
    arm.enable()
    time.sleep(0.5)
    arm.set_follower_mode()
    time.sleep(0.5)

    dt = 1.0 / rate
    print("\n" + "=" * 60)
    print("  SPACE = start/pause rollout    x = EMERGENCY STOP    q = quit")
    print(f"  {rate:g}Hz, max step {args.max_step} rad/tick, "
          f"{args.runtime:.0f}s limit")
    if args.dry_run:
        print("  DRY RUN — nothing is commanded")
    print("=" * 60 + "\n")

    running = False
    stopped = False
    started_at = None
    first_tick = True

    try:
        with Keyboard() as kb:
            next_tick = time.perf_counter()
            while True:
                key = kb.get()
                if key in ("q", "Q"):
                    break
                if key in ("x", "X"):
                    stopped, running = True, False
                    print("[STOP] emergency stop — commanding halted")
                if key == " " and not stopped:
                    running = not running
                    first_tick = True
                    started_at = time.perf_counter() if running else None
                    print(f"[ROLLOUT] {'started' if running else 'paused'}")

                if running and started_at and \
                        time.perf_counter() - started_at > args.runtime:
                    running = False
                    print(f"[ROLLOUT] stopped — {args.runtime:.0f}s limit")

                if running:
                    state = unwrap(arm.get_joint_angles())
                    gmsg = unwrap(gripper.get_gripper_status())
                    if state is None or gmsg is None:
                        print("[warn] no state this tick — skipped")
                    else:
                        qpos = np.array(list(state) + [gmsg.value],
                                        dtype=np.float32)
                        t_q = torch.from_numpy(
                            ((qpos - q_mean) / q_std)[None]).to(device)
                        with torch.no_grad():
                            pred = model(t_q)[0].cpu().numpy()
                        target = pred * a_std + a_mean

                        if first_tick:
                            gap = float(np.abs(target[:7] - qpos[:7]).max())
                            if gap > START_TOLERANCE:
                                running = False
                                print(f"[ABORT] first action is {gap:.3f} rad from "
                                      f"the current pose (limit {START_TOLERANCE}). "
                                      f"Move the arm nearer the demo start pose.")
                                continue
                            first_tick = False

                        joints = np.clip(target[:7], qpos[:7] - args.max_step,
                                         qpos[:7] + args.max_step)
                        grip = float(np.clip(target[7],
                                             qpos[7] - MAX_GRIPPER_STEP,
                                             qpos[7] + MAX_GRIPPER_STEP))
                        grip = float(np.clip(grip, GRIPPER_MIN_M, GRIPPER_MAX_M))

                        if args.dry_run:
                            print(f"  would command {np.round(joints, 3)} "
                                  f"grip {grip:.4f}")
                        else:
                            _try("move_js", arm.move_js,
                                 [float(v) for v in joints])
                            _try("move_gripper_m", gripper.move_gripper_m,
                                 value=grip, force=FOLLOWER_GRIP_FORCE)

                next_tick += dt
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        here = unwrap(_try("get_joint_angles", arm.get_joint_angles))
        if here is not None:
            _try("hold", arm.move_js, list(here))
        print("\n*** SUPPORT THE ARM — motors cut in 3 s ***")
        for i in (3, 2, 1):
            print(f"  {i}...")
            time.sleep(1.0)
        _try("gripper disable", gripper.disable_gripper)
        _try("arm disable", arm.disable)
        _try("arm disconnect", arm.disconnect)
        print("Disconnected.")


if __name__ == "__main__":
    main()
