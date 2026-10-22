#!/usr/bin/env python3
"""
Replay a recorded episode's leader command stream on the follower arm.

    python3 replay_episode.py episodes/ep_0003_20260914_121500_joints.h5 --dry-run
    python3 replay_episode.py episodes/ep_0003_20260914_121500_joints.h5

  SPACE  begin  (home -> 3s countdown -> replay)  /  pause once playing
  x      EMERGENCY STOP
  q      quit

SEQUENCE
  Pressing SPACE runs three phases without further input:
    1. HOMING     the follower is walked to the episode's first pose at a
                  capped rate, one small step per tick. It is never commanded
                  straight to a distant target -- move_js has no trajectory
                  planning, so a far command is executed as fast as the arm
                  can move.
    2. COUNTDOWN  3 seconds, so you can get clear and see what is about to run.
    3. REPLAY     the recorded command stream, on the recorded timeline.

  "Home" defaults to the episode's own first pose, which is what open-loop
  replay actually requires. --home overrides it with explicit joint values.

WHY REPLAY BEFORE TRAINING ANYTHING
  /leader/joint_angles is the exact command stream that drove the follower
  during the demo. Feeding it back through move_js uses the same call, the
  same arm, the same path. If replay reproduces the demo, your timestamps,
  action stream, gripper widths and command path are all sound, and any later
  failure belongs to the policy. If replay does NOT reproduce it, no amount of
  training will fix that.

  It also answers a question BC cannot: whether the task is reproducible
  OPEN-LOOP at all. Replay has no feedback -- errors accumulate rather than
  wash out. If the exact recorded commands fail, the task needs closed-loop
  vision and a proprioception-only policy will not rescue it.

  The LEADER stream is replayed, never the follower's. /follower is what
  resulted and already contains tracking lag; commanding it applies that lag
  twice.

ONLY THE FOLLOWER (--channel, default can0) IS OPENED. The leader arm is not
connected, enabled or commanded by this script. Do not run it while
nero_teleop_spacebar.py is running -- two processes commanding one arm fight
over every frame.

TIMING
  Playback follows the recorded timestamps, not a nominal rate. The control
  loop slips under bus load, so assuming 50Hz would drift against the original.
"""

import argparse
import select
import sys
import termios
import time
import tty

import h5py
import numpy as np

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

try:
    from pyAgxArm import resolve_firmware_profile
except ImportError:
    resolve_firmware_profile = None

# ----------------------------------------------------------------- safety ---
LOOP_HZ           = 50.0
MAX_STEP_RAD      = 0.05      # replay: per joint per tick
HOME_STEP_RAD     = 0.02      # homing: per joint per tick (~1 rad/s at 50Hz)
HOME_GRIP_STEP    = 0.004
HOME_TOLERANCE    = 0.02      # rad; converged when every joint is inside this
HOME_TIMEOUT_S    = 30.0
COUNTDOWN_S       = 3
MAX_GRIPPER_STEP  = 0.015     # replay: metres per tick
GRIPPER_MIN_M     = 0.000
GRIPPER_MAX_M     = 0.100

FOLLOWER_CHANNEL  = "can0"
FALLBACK_PROFILE  = NeroFW.V121
CONNECT_SETTLE_S  = 2.0
FOLLOWER_GRIP_FORCE = 1.0

IDLE, HOMING, COUNTDOWN, PLAYING, DONE = "idle", "homing", "countdown", "playing", "done"


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


def load_episode(path):
    """Returns (t, joints, gripper, attrs) from either file layout."""
    with h5py.File(path, "r") as f:
        t = f["timestamps"][:].astype(np.float64)
        if "action" in f:                          # aligned layout
            a = f["action"][:].astype(np.float64)
            joints, grip = a[:, :7], a[:, 7]
        else:                                      # raw joints layout
            joints = f["leader/joint_angles"][:].astype(np.float64)
            grip = f["leader/gripper"][:].astype(np.float64)
        meta = {k: f.attrs[k] for k in f.attrs}
    n = min(len(t), len(joints), len(grip))
    t, joints, grip = t[:n], joints[:n], grip[:n]
    good = np.isfinite(joints).all(axis=1) & np.isfinite(grip)
    return t[good], joints[good], grip[good], meta


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
    ap.add_argument("episode", help="an *_joints.h5 or *_aligned.h5 file")
    ap.add_argument("--channel", default=FOLLOWER_CHANNEL)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (0.5 = half speed)")
    ap.add_argument("--max-step", type=float, default=MAX_STEP_RAD)
    ap.add_argument("--home", default=None,
                    help="comma-separated joint values to home to; "
                         "default is the episode's first pose")
    ap.add_argument("--home-timeout", type=float, default=HOME_TIMEOUT_S)
    ap.add_argument("--skip-home", action="store_true",
                    help="start replay without homing (arm must already be there)")
    ap.add_argument("--no-gripper", action="store_true")
    ap.add_argument("--loop", action="store_true", help="repeat until stopped")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the whole sequence and print, never command the arm")
    args = ap.parse_args()

    t, joints, grip, meta = load_episode(args.episode)
    if len(t) < 2:
        raise SystemExit("episode has too few samples")
    dur = t[-1] - t[0]

    if args.home:
        home = np.array([float(v) for v in args.home.split(",")], dtype=np.float64)
        if home.shape != (7,):
            raise SystemExit("--home needs exactly 7 comma-separated values")
        home_grip = float(grip[0])
    else:
        home = joints[0].copy()
        home_grip = float(grip[0])

    print(f"{args.episode}")
    print(f"  {len(t)} samples, {dur:.1f}s, {len(t) / dur:.1f}Hz mean, "
          f"speed x{args.speed:g}")
    print(f"  gripper range {grip.min():.4f} to {grip.max():.4f} m")
    if "episode" in meta:
        print(f"  recorded {meta.get('start_wall_time', '?')} "
              f"as episode {meta['episode']}")
    print(f"  home target: {np.round(home, 4)} grip {home_grip:.4f}"
          f"{' (from --home)' if args.home else ' (episode first pose)'}")

    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=detect_profile(args.channel),
        interface="socketcan", channel=args.channel))
    arm.connect()
    time.sleep(CONNECT_SETTLE_S)
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)

    print("\nEnabling in 3 s. CLEAR THE WORKSPACE.")
    time.sleep(3)
    arm.enable()
    time.sleep(0.5)
    arm.set_follower_mode()
    time.sleep(0.5)

    here = unwrap(_try("get_joint_angles", arm.get_joint_angles))
    if here is not None:
        gap = float(np.abs(np.asarray(here[:7]) - home).max())
        secs = gap / (HOME_STEP_RAD * LOOP_HZ)
        print(f"\n  arm is {gap:.3f} rad from home "
              f"(~{secs:.1f}s to walk there at {HOME_STEP_RAD} rad/tick)")

    print("\n" + "=" * 62)
    print("  SPACE = begin (home -> countdown -> replay)   x = STOP   q = quit")
    if args.skip_home:
        print("  --skip-home: replay starts immediately, no homing")
    if args.dry_run:
        print("  DRY RUN — nothing is commanded")
    print("=" * 62 + "\n")

    phase = IDLE
    stopped = False
    i = 0
    play_t0 = None
    home_t0 = None
    count_end = None
    count_shown = None
    errors = 0
    dt = 1.0 / LOOP_HZ

    try:
        with Keyboard() as kb:
            next_tick = time.perf_counter()
            while True:
                key = kb.get()
                if key in ("q", "Q"):
                    break
                if key in ("x", "X"):
                    stopped, phase = True, IDLE
                    print("[STOP] emergency stop — commanding halted")
                if key == " " and not stopped:
                    if phase == IDLE:
                        i = 0
                        if args.skip_home:
                            phase = COUNTDOWN
                            count_end = time.perf_counter() + COUNTDOWN_S
                            count_shown = None
                        else:
                            phase = HOMING
                            home_t0 = time.perf_counter()
                            print("[HOME] moving to the episode's first pose...")
                    elif phase == PLAYING:
                        phase = IDLE
                        print(f"[REPLAY] paused at sample {i} ({t[i] - t[0]:.1f}s)")

                # ---- HOMING --------------------------------------------------
                if phase == HOMING:
                    state = unwrap(arm.get_joint_angles())
                    if state is None:
                        errors += 1
                    else:
                        now_j = np.asarray(state[:7], dtype=np.float64)
                        err = home - now_j
                        if float(np.abs(err).max()) <= HOME_TOLERANCE:
                            print(f"[HOME] reached "
                                  f"(max error {np.abs(err).max():.4f} rad)")
                            phase = COUNTDOWN
                            count_end = time.perf_counter() + COUNTDOWN_S
                            count_shown = None
                        else:
                            cmd = now_j + np.clip(err, -HOME_STEP_RAD,
                                                  HOME_STEP_RAD)
                            if args.dry_run:
                                pass
                            else:
                                _try("move_js", arm.move_js,
                                     [float(v) for v in cmd])
                                if not args.no_gripper:
                                    gmsg = unwrap(gripper.get_gripper_status())
                                    if gmsg is not None:
                                        g = float(np.clip(
                                            home_grip,
                                            gmsg.value - HOME_GRIP_STEP,
                                            gmsg.value + HOME_GRIP_STEP))
                                        g = float(np.clip(g, GRIPPER_MIN_M,
                                                          GRIPPER_MAX_M))
                                        _try("move_gripper_m",
                                             gripper.move_gripper_m,
                                             value=g,
                                             force=FOLLOWER_GRIP_FORCE)
                            if args.dry_run and time.perf_counter() - home_t0 > 1.0:
                                print("[HOME] (dry run — assuming reached)")
                                phase = COUNTDOWN
                                count_end = time.perf_counter() + COUNTDOWN_S
                                count_shown = None
                    if phase == HOMING and \
                            time.perf_counter() - home_t0 > args.home_timeout:
                        phase = IDLE
                        print(f"[HOME] TIMEOUT after {args.home_timeout:.0f}s — "
                              f"not started. Check the arm is free to move.")

                # ---- COUNTDOWN -----------------------------------------------
                elif phase == COUNTDOWN:
                    remaining = count_end - time.perf_counter()
                    whole = int(np.ceil(max(remaining, 0)))
                    if whole != count_shown and whole > 0:
                        count_shown = whole
                        print(f"  replay in {whole}...")
                    if remaining <= 0:
                        phase = PLAYING
                        play_t0 = time.perf_counter()
                        i = 0
                        print(f"[REPLAY] started — {len(t)} samples, {dur:.1f}s")

                # ---- PLAYING -------------------------------------------------
                elif phase == PLAYING:
                    elapsed = (time.perf_counter() - play_t0) * args.speed
                    target_i = int(np.searchsorted(t - t[0], elapsed))
                    if target_i > i:
                        i = min(target_i, len(t) - 1)
                        state = unwrap(arm.get_joint_angles())
                        gmsg = unwrap(gripper.get_gripper_status())
                        if state is None:
                            errors += 1
                        else:
                            now_j = np.asarray(state[:7], dtype=np.float64)
                            cmd = np.clip(joints[i], now_j - args.max_step,
                                          now_j + args.max_step)
                            if args.dry_run:
                                if i % 25 == 0:
                                    print(f"  t={t[i] - t[0]:6.2f}s  "
                                          f"j1={cmd[0]:+.3f}  grip={grip[i]:.4f}")
                            else:
                                _try("move_js", arm.move_js,
                                     [float(v) for v in cmd])
                                if not args.no_gripper and gmsg is not None:
                                    g = float(np.clip(
                                        grip[i], gmsg.value - MAX_GRIPPER_STEP,
                                        gmsg.value + MAX_GRIPPER_STEP))
                                    g = float(np.clip(g, GRIPPER_MIN_M,
                                                      GRIPPER_MAX_M))
                                    _try("move_gripper_m",
                                         gripper.move_gripper_m,
                                         value=g, force=FOLLOWER_GRIP_FORCE)

                    if i >= len(t) - 1:
                        if args.loop:
                            phase = HOMING
                            home_t0 = time.perf_counter()
                            print("[REPLAY] finished — re-homing for next pass")
                        else:
                            phase = IDLE
                            print(f"[REPLAY] finished ({len(t)} samples, "
                                  f"{errors} state read failures)")

                next_tick += dt
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    next_tick = time.perf_counter()

    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        last = unwrap(_try("get_joint_angles", arm.get_joint_angles))
        if last is not None:
            _try("hold", arm.move_js, list(last))
        print("\n*** SUPPORT THE ARM — motors cut in 3 s ***")
        for k in (3, 2, 1):
            print(f"  {k}...")
            time.sleep(1.0)
        _try("gripper disable", gripper.disable_gripper)
        _try("arm disable", arm.disable)
        _try("arm disconnect", arm.disconnect)
        print("Disconnected.")


if __name__ == "__main__":
    main()
