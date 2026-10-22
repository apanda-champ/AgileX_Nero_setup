#!/usr/bin/env python3
"""
Leader-follower teleoperation, AgileX Nero x2, with a SPACEBAR gravity-comp clutch.

  h      go to HOME_POSE (both arms)
  r      start a demo          e  end the demo and write it
  d      discard the demo in progress
  p      print the leader's current pose as a HOME_POSE line
  SPACE  toggle gravity compensation by hand
  q      quit (Ctrl+C also works)

  Normal cycle:  h -> r -> demonstrate -> e -> h -> r -> ...

  can1 = LEADER    can0 = FOLLOWER    firmware 1.21 -> profile v121

  Flags: --no-home  --no-camera  --no-record  --dry-home  --help

START POSE

  Both arms are driven to HOME_POSE once at startup, before the clutch loop,
  so every session begins from the same configuration. The move is an
  interpolated move_js ramp generated on the host at LOOP_HZ -- move_js is MIT
  passthrough with no trajectory planning, so a single call with a distant
  target would make the arm snap toward it.

  The follower is homed as well, not as a nicety: the loop mirrors leader ->
  follower every tick, so homing only the leader would leave the first tick
  commanding the follower across the whole gap in one unsmoothed move_js.

  Homing is REFUSED if any joint is more than HOME_MAX_DELTA_RAD from target.
  A joint-space ramp is a straight line in joint space, not in the workspace,
  so a long sweep from an arbitrary cold-start pose is not safe to run blind.
  Teleop still runs if homing is refused, from wherever the arms are.

WHY IT IS BUILT THIS WAY (all hardware-measured)

  Firmware profile. These arms report 1.21, which requires NeroFW.V121. Under
  NeroFW.DEFAULT the leader's state is unreadable in gravity-comp mode. The
  profile is resolved at runtime from get_firmware().

  Two joint readers, mutually exclusive:
                            locked (set_follower_mode)   GC on (set_leader_mode)
      get_joint_angles()            live                        FROZEN
      get_leader_joint_angles()     None                        live
  The loop switches readers with the clutch. Never trust the wrong one -- the
  frozen reader returns well-formed stale data with no error.

  Gripper. The two width readers are complementary in exactly the same way:
      get_gripper_status()       live when locked, FROZEN under GC
      get_gripper_ctrl_states()  no frames when locked, live under GC
  Both report the same 0.000-0.10091 m range, so the reader switches with the
  clutch and the follower mirrors at GRIP_GAIN in both states.

  Recording. One HDF5 file per demo, and recording is INDEPENDENT of the
  clutch: r opens an episode, e writes it, d throws it away. Gravity comp is a
  property of the arm, not a file boundary -- tying them together would make it
  impossible to home between demos without cutting a file. Samples are buffered
  in RAM and written once at the end.

  set_normal_mode() is a no-op on firmware >=1.12 and is never called.
  disable() may be ignored in leader mode, so shutdown leaves leader mode first.
"""

import datetime
import os
import select
import sys
import termios
import threading
import time
import tty

# ---- flags ------------------------------------------------------------------
# Parsed here, before numpy/depthai/pyAgxArm are imported, so --help and a
# typo'd flag both work on a machine with none of them installed and neither
# can reach the hardware. The overrides are applied further down, once the
# constants they change exist.
USAGE = """\
nero_teleop_spacebar.py — leader-follower teleop with a spacebar clutch

  usage: ./nero_teleop_spacebar.py [flags]

  --no-home      skip the start-pose move; begin teleop wherever the arms are
  --no-camera    do not open the OAK-D (joints are still recorded)
  --no-record    do not write any HDF5 files (camera is skipped too)
  --dry-home     move to the start pose, then exit without entering teleop
  -h, --help     print this and exit

  Keys once running:
    h      go to the start pose
    r      start a demo        e   end it and write the file
    d      discard the demo in progress
    p      print the leader's pose as a HOME_POSE line
    SPACE  toggle gravity comp by hand      q   quit

  Normal cycle:  h -> r -> demonstrate -> e -> h -> r -> ...

  First run after changing HOME_POSE:
    ./nero_teleop_spacebar.py --dry-home
"""

_KNOWN = {"--no-home", "--no-camera", "--no-record", "--dry-home", "-h", "--help"}
_argv = sys.argv[1:]
_bad = [a for a in _argv if a not in _KNOWN]
if _bad:
    print(f"Unknown flag(s): {' '.join(_bad)}\n{USAGE}")
    sys.exit(2)
if "-h" in _argv or "--help" in _argv:
    print(USAGE)
    sys.exit(0)
DRY_HOME = "--dry-home" in _argv

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import cv2
    import depthai as dai
except ImportError:
    cv2 = None
    dai = None

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

try:
    from pyAgxArm import resolve_firmware_profile
except ImportError:
    resolve_firmware_profile = None

# ----------------------------------------------------------------- tuning ---
LEADER_CHANNEL    = "can1"
FOLLOWER_CHANNEL  = "can0"
FALLBACK_PROFILE  = NeroFW.V121

LOOP_HZ           = 50.0
LOOP_DT           = 1.0 / LOOP_HZ
CONNECT_SETTLE_S  = 2.0
MODE_SETTLE_S     = 0.08          # let feedback swap over after a mode change
MIN_TOGGLE_GAP_S  = 0.30          # ignore repeats from a held spacebar

# Leader gripper: enabled but below stall, so the jaws stay hand-free and
# keep reporting while the arm is locked.
LEADER_GRIP_FORCE   = 0.05
LEADER_GRIP_TARGET  = 0.1
GRIP_REASSERT_S     = 1.0

# ---- start pose -------------------------------------------------------------
# Both arms are driven here once at startup, before the clutch loop begins.
# Angles are radians, j1..j7, same layout get_joint_angles() returns. Press "p"
# during a session to print the leader's current pose in this exact format.
#
# The follower is homed as well, not as a nicety: the loop mirrors leader ->
# follower every tick, so if only the leader moved, the first tick would command
# the follower across the whole gap in a single unsmoothed move_js.
HOME_ON_START     = True
HOME_POSE         = [-0.0832, 0.3611, -0.0326, 0.6512, 0.0360, -0.0283, 0.4728]
HOME_FOLLOWER_TOO = True
HOME_DURATION_S   = 4.0       # ramp time; longer is gentler
HOME_HOLD_S       = 0.5       # keep commanding the target after arrival
HOME_MAX_DELTA_RAD = 1.8      # refuse to auto-home if any joint is further out
HOME_WARN_S       = 3.0       # countdown before anything moves
HOME_WARN_MID_S   = 0.0       # countdown for a mid-session h (you are right there)
POSE_PRINT_KEY    = "p"

# ---- keys -------------------------------------------------------------------
# Gravity comp and recording are INDEPENDENT. GC is a property of the arm; a
# demo is a span of time you choose. Binding them together means every clutch
# change is also a file boundary, which is wrong as soon as you want to home
# between demos.
HOME_KEY          = "h"       # go to HOME_POSE (drops GC and restores it)
REC_START_KEY     = "r"       # begin a demo
REC_END_KEY       = "e"       # end the demo and write it
GC_TOGGLE_KEY     = " "       # manual clutch, kept as an escape hatch
START_WITH_GC     = True      # engage gravity comp once homing is done
GC_WARN_S         = 3.0       # countdown before the leader goes weightless

# Gripper mapping: the follower's jaws move GRIP_GAIN units for every unit the
# leader's move. Gain 1.0 is pass-through; above 1 is coarser (small hand
# movements produce large jaw changes); below 1 is finer, for delicate grasps.
# Zero always maps to zero, so fully-closed leader jaws still fully close the
# follower whatever the gain. The clamp exists only so an out-of-range value is
# never sent on.
GRIP_GAIN         = 1.5
CLAMP_TO_FOLLOWER_RANGE = True
FOLLOWER_MIN_M    = 0.000
FOLLOWER_MAX_M    = 0.1
FOLLOWER_GRIP_FORCE = 1.0
MIRROR_GRIPPER    = True

ERR_LOG_PERIOD_S  = 2.0
SHUTDOWN_WARN_S   = 3.0

# ---- recording --------------------------------------------------------------
# One HDF5 file per episode. An episode is one span of gravity comp: it starts
# when you press SPACE to engage GC and ends when you press SPACE to release it
# (or on exit). Samples are buffered in RAM and written once at episode end --
# writing at 50Hz would put file I/O jitter into the control loop.
RECORD            = True
RECORD_DIR        = "home/champ/AgileX_Nero_setup/episodes"
RECORD_FOLLOWER   = True      # also log the follower's actual state
H5_COMPRESSION    = "gzip"    # None to disable
DISCARD_KEY       = "d"       # drop the episode in progress instead of saving

# ---- camera -----------------------------------------------------------------
# OAK-D runs at 30 FPS against this 50Hz loop, so frames are stored with their
# OWN device timestamps rather than one per control tick. The camera lives on a
# separate thread: JPEG encoding in the control loop would add timing jitter.
#
# Device timestamps are synced to the host steady clock, so joint samples are
# taken with time.monotonic() (NOT perf_counter) to share one clock with them.
CAMERA            = True
CAMERA_DEPTH      = True
CAMERA_FPS        = 30
RGB_SIZE          = (640, 400)
DEPTH_SIZE        = (640, 400)   # depth arrives 1280x800; downscaled on host
JPEG_QUALITY      = 90           # ~50KB/frame at 640x400; raise for fidelity


# ------------------------------------------------------------- command line ---
# Flags parsed at the top of the file; applied here, now that the constants
# they override exist. Everything is a toggle; there are no values to pass.
if "--no-home" in _argv:
    HOME_ON_START = False
if "--no-record" in _argv:
    RECORD = False
    CAMERA = False          # nowhere to put the frames
if "--no-camera" in _argv:
    CAMERA = False
if DRY_HOME:
    CAMERA = False          # nothing to record in a homing-only run
    RECORD = False


# ---------------------------------------------------------------- helpers ---
def _try(label, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(f"[warn] {label}: {exc.__class__.__name__}: {exc}")
        return None


def build(channel, profile):
    return AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=profile,
        interface="socketcan", channel=channel))


def detect_profile(channel):
    """Read the firmware with a throwaway DEFAULT connection, then resolve."""
    arm = build(channel, NeroFW.DEFAULT)
    ver = None
    try:
        arm.connect()
        time.sleep(CONNECT_SETTLE_S)
        fw = arm.get_firmware()
        ver = fw.get("software_version") if isinstance(fw, dict) else str(fw)
    except Exception as exc:
        print(f"[warn] {channel} firmware probe: {exc.__class__.__name__}: {exc}")
    finally:
        _try(f"{channel} probe disconnect", arm.disconnect)

    if ver and resolve_firmware_profile:
        try:
            prof = resolve_firmware_profile(ArmModel.NERO, ver)
            print(f"[init] {channel}: firmware {ver} -> profile {prof!r}")
            return prof
        except Exception as exc:
            print(f"[warn] resolve_firmware_profile: {exc.__class__.__name__}: {exc}")
    print(f"[init] {channel}: firmware {ver} -> falling back to {FALLBACK_PROFILE!r}")
    return FALLBACK_PROFILE


def unwrap(obj):
    return getattr(obj, "msg", None) if obj is not None else None


def read_leader_width(gc_active):
    """Jaw width in metres, or None.

    The two gripper readers are complementary, exactly like the joint readers:
        locked (set_follower_mode) -> get_gripper_status() live, ctrl_states None
        GC on  (set_leader_mode)   -> get_gripper_ctrl_states() live, status frozen
    Both report the same 0.000-0.10091 m range.

    There is deliberately NO fallback to the other reader. The wrong one does
    not fail -- it returns a frozen, well-formed value. Falling back would
    silently mirror a stale width.
    """
    fn = (leader_gripper.get_gripper_ctrl_states if gc_active
          else leader_gripper.get_gripper_status)
    msg = unwrap(fn())
    return None if msg is None else msg.value


def map_width(leader_m):
    """Scale the leader's jaw width by GRIP_GAIN for the follower.

    At GRIP_GAIN = 1.5, one unit of leader travel becomes 1.5 units of follower
    travel, measured from fully closed. Zero maps to zero, so the follower still
    closes completely.

    Clamped to the follower's commandable range so it is never sent a value it
    cannot accept. Set CLAMP_TO_FOLLOWER_RANGE = False for a raw scaled value
    with no clamp at all.

    Note the gain means the follower saturates before the leader does: at 1.5,
    the follower is fully open once the leader passes FOLLOWER_MAX_M / 1.5, and
    leader travel beyond that has no effect.
    """
    w = leader_m * GRIP_GAIN
    if not CLAMP_TO_FOLLOWER_RANGE:
        return w
    return min(FOLLOWER_MAX_M, max(FOLLOWER_MIN_M, w))


def _smoothstep(u):
    """Ease in and out, so the ramp has no velocity step at either end."""
    return u * u * (3.0 - 2.0 * u)


def read_locked_angles(arm, label):
    """Live joint angles. Valid ONLY while the arm is in follower (locked) mode.

    Under gravity comp get_joint_angles() returns frozen, well-formed data with
    no error, so calling this on a leader in GC would silently ramp from a stale
    start point. Home before the clutch loop, never during GC.
    """
    msg = unwrap(_try(f"{label}.get_joint_angles", arm.get_joint_angles))
    return None if msg is None else list(msg)


def ramp_to_pose(arms, target, duration=HOME_DURATION_S):
    """Drive one or more LOCKED arms to `target` with an interpolated move_js ramp.

    move_js is MIT passthrough -- no smoothing, no trajectory planning -- so a
    single call with a distant target makes the arm snap toward it at whatever
    rate the driver can manage. The trajectory is therefore generated here, on
    the host, at the same LOOP_HZ the teleop loop runs at.

    `arms` is a list of (label, arm) pairs. Every arm must already be in
    follower mode and enabled. Returns True only if every arm was driven.
    """
    plan = {}
    for label, arm in arms:
        here = read_locked_angles(arm, label)
        if here is None:
            print(f"[home] {label}: no joint feedback — homing ABORTED, "
                  f"arms left where they are.")
            return False
        n = min(len(here), len(target))
        here, goal = here[:n], list(target[:n])
        delta = max(abs(g - h) for g, h in zip(goal, here))
        if delta > HOME_MAX_DELTA_RAD:
            print(f"[home] {label}: {delta:.2f} rad from the start pose "
                  f"(limit {HOME_MAX_DELTA_RAD:.2f}) — homing ABORTED. Move it "
                  f"closer by hand, or raise HOME_MAX_DELTA_RAD if the sweep "
                  f"is genuinely clear.")
            return False
        plan[label] = (arm, here, goal)
        print(f"[home] {label}: {[round(v, 3) for v in here]} -> "
              f"{[round(v, 3) for v in goal]}  (max {delta:.2f} rad)")

    steps = max(1, int(duration * LOOP_HZ))
    next_tick = time.perf_counter()
    for i in range(1, steps + 1):
        a = _smoothstep(i / steps)
        for label, (arm, here, goal) in plan.items():
            _try(f"{label}.move_js (home)", arm.move_js,
                 [h + a * (g - h) for h, g in zip(here, goal)])
        next_tick += LOOP_DT
        slack = next_tick - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            next_tick = time.perf_counter()

    # move_js trades stiffness for response, so the arm can settle short of the
    # target under its own weight. Keep commanding it until the loop takes over.
    hold_until = time.perf_counter() + HOME_HOLD_S
    while time.perf_counter() < hold_until:
        for label, (arm, _here, goal) in plan.items():
            _try(f"{label}.move_js (hold)", arm.move_js, goal)
        time.sleep(LOOP_DT)

    # Read back. move_js trades stiffness for response, so a loaded joint can
    # settle short of its target; this is the number that tells you whether
    # move_js is good enough for your arms or whether the leader needs
    # controller-side planning (set_speed_percent + move_j) instead.
    ok = True
    for label, (arm, _here, goal) in plan.items():
        landed = read_locked_angles(arm, label)
        if landed is None:
            print(f"[home] {label}: no feedback to verify with")
            ok = False
            continue
        err = [l - g for l, g in zip(landed[:len(goal)], goal)]
        worst = max(range(len(err)), key=lambda i: abs(err[i]))
        if abs(err[worst]) > 0.02:
            print(f"[home] {label}: settled {err[worst]:+.4f} rad off on j{worst+1} "
                  f"— likely gravity sag under move_js")
            ok = False
        else:
            print(f"[home] {label}: on target (worst joint {err[worst]:+.4f} rad)")
    print("[home] At the start pose." if ok
          else "[home] At the start pose, with the offsets noted above.")
    return True


def format_pose(joints):
    return "[" + ", ".join(f"{v:.4f}" for v in joints) + "]"


class RateLimitedLogger:
    def __init__(self, period):
        self.period, self.last = period, {}

    def log(self, key, text):
        now = time.perf_counter()
        if now - self.last.get(key, -1e9) >= self.period:
            self.last[key] = now
            print(text)


class Keyboard:
    """Non-blocking single-key reader. cbreak keeps Ctrl+C working."""

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.saved = None
        self.active = sys.stdin.isatty()

    def __enter__(self):
        if self.active:
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        else:
            print("[warn] stdin is not a terminal — spacebar clutch disabled.")
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def get(self):
        if not self.active:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class Recorder:
    """One HDF5 file per episode. Buffers in RAM, writes on episode end.

    A sample is only stored when the leader's joints AND width are both
    available this tick. A dropped read is skipped, never carried forward --
    repeating the last value would fabricate data, and timestamps are stored
    explicitly so the gap is visible rather than hidden.
    """

    def __init__(self, directory, record_follower, compression):
        self.dir = directory
        self.record_follower = record_follower
        self.compression = compression
        self.enabled = RECORD and h5py is not None
        self.active = False
        self.episode = 0
        self.lock = threading.Lock()      # camera thread also writes here
        self._clear()

        if not self.enabled:
            if RECORD and h5py is None:
                print("[warn] h5py not installed — recording disabled. "
                      "pip install h5py")
            return

        os.makedirs(self.dir, exist_ok=True)
        existing = [f for f in os.listdir(self.dir)
                    if f.startswith("ep_") and f.endswith(".h5")]
        for name in existing:                     # continue the numbering
            try:
                self.episode = max(self.episode, int(name.split("_")[1]))
            except (IndexError, ValueError):
                pass
        print(f"[init] Recording to {os.path.abspath(self.dir)}/ "
              f"(next episode {self.episode + 1:04d}, joints and camera "
              f"written as separate files)")

    def _clear(self):
        self.t, self.lj, self.lw, self.fj, self.fw = [], [], [], [], []
        self.cam_t, self.cam_jpg, self.cam_depth = [], [], []
        self.t0 = None
        self.started_iso = None
        self.dropped = 0

    def start(self):
        if not self.enabled:
            return
        with self.lock:
            self.episode += 1
            self._clear()
            # monotonic, NOT perf_counter: DepthAI device timestamps are synced
            # to the host steady clock, so this puts both streams on one clock.
            self.t0 = time.monotonic()
            self.started_iso = datetime.datetime.now().isoformat(timespec="seconds")
            self.active = True
        print(f"[REC] Episode {self.episode:04d} started")

    def sample(self, leader_joints, leader_width, follower_joints, follower_width):
        if not (self.enabled and self.active):
            return
        if leader_joints is None or leader_width is None:
            self.dropped += 1
            return
        with self.lock:
            if not self.active:
                return
            self.t.append(time.monotonic() - self.t0)
            self.lj.append(list(leader_joints))
            self.lw.append(float(leader_width))
            if self.record_follower:
                self.fj.append(list(follower_joints) if follower_joints is not None
                               else [float("nan")] * len(leader_joints))
                self.fw.append(float(follower_width) if follower_width is not None
                               else float("nan"))

    def camera_sample(self, device_ts, jpeg, depth):
        """Called from the camera thread. device_ts is on the monotonic clock."""
        if not (self.enabled and self.active):
            return
        with self.lock:
            if not self.active or self.t0 is None:
                return
            self.cam_t.append(device_ts - self.t0)
            self.cam_jpg.append(jpeg)
            if CAMERA_DEPTH:
                self.cam_depth.append(depth)

    def stop(self, save=True):
        if not (self.enabled and self.active):
            return None
        with self.lock:
            self.active = False

        if not save:
            n, nc = len(self.t), len(self.cam_t)
            self._clear()
            print(f"[REC] Episode {self.episode:04d} DISCARDED "
                  f"({n} samples, {nc} frames)")
            self.episode -= 1
            return None

        if not self.t:
            print(f"[REC] Episode {self.episode:04d} had no samples — nothing written")
            self.episode -= 1
            return None

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"ep_{self.episode:04d}_{stamp}"
        jpath = os.path.join(self.dir, base + "_joints.h5")
        cpath = os.path.join(self.dir, base + "_camera.h5")
        duration = self.t[-1]

        written = []
        ok_j = self._write_joints(jpath, duration)
        if ok_j:
            written.append(jpath)
        ok_c = self._write_camera(cpath, duration) if self.cam_t else False
        if ok_c:
            written.append(cpath)

        rate = len(self.t) / duration if duration > 0 else 0.0
        cam_rate = len(self.cam_t) / duration if duration > 0 and self.cam_t else 0.0
        mb = sum(os.path.getsize(p) for p in written) / 1e6
        print(f"[REC] Episode {self.episode:04d} saved: {len(self.t)} samples "
              f"@ {rate:.1f}Hz, {len(self.cam_t)} frames @ {cam_rate:.1f}Hz, "
              f"{duration:.1f}s, {self.dropped} dropped, {mb:.1f} MB")
        for p in written:
            print(f"      {p}")
        if self.cam_t and not ok_c:
            print("      [warn] camera file was NOT written")
        self._clear()
        return written or None

    def _common_attrs(self, f, duration):
        """Written into BOTH files so a pair can be matched and cross-checked."""
        f.attrs["episode"] = self.episode
        f.attrs["start_wall_time"] = self.started_iso
        f.attrs["t0_monotonic"] = self.t0
        f.attrs["duration_s"] = duration
        f.attrs["clock"] = ("time.monotonic; timestamps are seconds since "
                            "t0_monotonic; camera stamps are device-synced to "
                            "the same clock")

    def _write_joints(self, path, duration):
        kw = {"compression": self.compression} if self.compression else {}
        try:
            with h5py.File(path, "w") as f:
                f.create_dataset("timestamps", data=np.asarray(self.t), **kw)
                g = f.create_group("leader")
                g.create_dataset("joint_angles", data=np.asarray(self.lj), **kw)
                g.create_dataset("gripper", data=np.asarray(self.lw), **kw)
                if self.record_follower:
                    g2 = f.create_group("follower")
                    g2.create_dataset("joint_angles", data=np.asarray(self.fj), **kw)
                    g2.create_dataset("gripper", data=np.asarray(self.fw), **kw)
                self._common_attrs(f, duration)
                f.attrs["stream"] = "joints"
                f.attrs["n_samples"] = len(self.t)
                f.attrs["dropped_samples"] = self.dropped
                f.attrs["mean_rate_hz"] = (len(self.t) / duration
                                           if duration > 0 else 0.0)
                f.attrs["target_rate_hz"] = LOOP_HZ
                f.attrs["leader_channel"] = LEADER_CHANNEL
                f.attrs["follower_channel"] = FOLLOWER_CHANNEL
                f.attrs["units"] = "joints: rad; gripper: m; timestamps: s"
            return True
        except Exception as exc:
            print(f"[REC] JOINTS WRITE FAILED {path}: "
                  f"{exc.__class__.__name__}: {exc}")
            return False

    def _write_camera(self, path, duration):
        kw = {"compression": self.compression} if self.compression else {}
        try:
            with h5py.File(path, "w") as f:
                f.create_dataset("timestamps", data=np.asarray(self.cam_t), **kw)
                # JPEG frames are variable length, so a vlen uint8 dataset.
                # Decode with cv2.imdecode(arr, cv2.IMREAD_COLOR).
                vlen = h5py.vlen_dtype(np.uint8)
                d = f.create_dataset("color_jpeg", (len(self.cam_jpg),), dtype=vlen)
                for i, buf in enumerate(self.cam_jpg):
                    d[i] = np.frombuffer(buf, dtype=np.uint8)
                if CAMERA_DEPTH and self.cam_depth:
                    f.create_dataset("depth_mm",
                                     data=np.asarray(self.cam_depth,
                                                     dtype=np.uint16), **kw)
                self._common_attrs(f, duration)
                f.attrs["stream"] = "camera"
                f.attrs["n_frames"] = len(self.cam_t)
                f.attrs["mean_rate_hz"] = (len(self.cam_t) / duration
                                           if duration > 0 else 0.0)
                f.attrs["rgb_size"] = list(RGB_SIZE)
                f.attrs["depth_size"] = list(DEPTH_SIZE)
                f.attrs["jpeg_quality"] = JPEG_QUALITY
                f.attrs["target_fps"] = CAMERA_FPS
                f.attrs["depth_units"] = "millimetres, uint16, 0 = invalid"
            return True
        except Exception as exc:
            print(f"[REC] CAMERA WRITE FAILED {path}: "
                  f"{exc.__class__.__name__}: {exc}")
            return False


class CameraSource:
    """OAK-D on its own thread. Pushes frames straight into the recorder.

    Runs at CAMERA_FPS regardless of the control loop rate, so frames carry
    their own device timestamps. JPEG encoding happens here, off the control
    thread. cv2.imencode releases the GIL, so this costs the loop very little.
    """

    def __init__(self, recorder):
        self.recorder = recorder
        self.enabled = CAMERA and dai is not None and cv2 is not None
        self.thread = None
        self.running = False
        self.frames = 0
        self.last_shape = None
        if CAMERA and (dai is None or cv2 is None):
            print("[warn] depthai/opencv not installed — camera disabled. "
                  "pip install depthai opencv-python")

    def start(self):
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(2.0)                      # let the pipeline boot
        print(f"[init] Camera streaming ({self.frames} frames seen so far)")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _run(self):
        try:
            with dai.Pipeline() as pipeline:
                cam = pipeline.create(dai.node.Camera).build(
                    dai.CameraBoardSocket.CAM_A)
                q_rgb = cam.requestOutput(
                    RGB_SIZE, dai.ImgFrame.Type.BGR888i,
                    fps=CAMERA_FPS).createOutputQueue(maxSize=4, blocking=False)

                q_depth = None
                if CAMERA_DEPTH:
                    try:
                        left = pipeline.create(dai.node.Camera).build(
                            dai.CameraBoardSocket.CAM_B)
                        right = pipeline.create(dai.node.Camera).build(
                            dai.CameraBoardSocket.CAM_C)
                        stereo = pipeline.create(dai.node.StereoDepth)
                        left.requestOutput(RGB_SIZE, fps=CAMERA_FPS).link(stereo.left)
                        right.requestOutput(RGB_SIZE, fps=CAMERA_FPS).link(stereo.right)
                        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
                        q_depth = stereo.depth.createOutputQueue(
                            maxSize=4, blocking=False)
                    except Exception as exc:
                        print(f"[warn] depth setup failed, colour only: "
                              f"{exc.__class__.__name__}: {exc}")
                        q_depth = None

                pipeline.start()
                enc = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                latest_depth = None

                while self.running and pipeline.isRunning():
                    if q_depth is not None:
                        d = q_depth.tryGet()
                        if d is not None:
                            dep = d.getFrame()
                            if (dep.shape[1], dep.shape[0]) != DEPTH_SIZE:
                                dep = cv2.resize(dep, DEPTH_SIZE,
                                                 interpolation=cv2.INTER_NEAREST)
                            latest_depth = dep

                    frame = q_rgb.tryGet()
                    if frame is not None:
                        self.frames += 1
                        self.last_shape = (frame.getWidth(), frame.getHeight())
                        if self.recorder.active:
                            ok, buf = cv2.imencode(".jpg", frame.getCvFrame(), enc)
                            if ok:
                                self.recorder.camera_sample(
                                    frame.getTimestamp().total_seconds(),
                                    buf.tobytes(),
                                    latest_depth if latest_depth is not None
                                    else np.zeros((DEPTH_SIZE[1], DEPTH_SIZE[0]),
                                                  dtype=np.uint16))
                    else:
                        time.sleep(0.002)
        except Exception as exc:
            print(f"[warn] camera thread stopped: {exc.__class__.__name__}: {exc}")
            self.enabled = False


# ------------------------------------------------------------------- setup ---
print()
print("=" * 68)
print("  NERO LEADER-FOLLOWER TELEOP")
print("=" * 68)
print(f"  Leader   {LEADER_CHANNEL}      Follower {FOLLOWER_CHANNEL}      Loop {LOOP_HZ:.0f} Hz")
print()
print("  This run will, in order:")
print("    1. probe both arms for firmware and connect")
print("    2. ENABLE both arms          <- they become powered and stiff")
if HOME_ON_START:
    print("    3. MOVE BOTH ARMS to the start pose over "
          f"{HOME_DURATION_S:.0f} s   <- they move on their own")
    print(f"       start pose: {[round(v, 3) for v in HOME_POSE]}")
else:
    print("    3. skip homing (--no-home) — teleop starts from the current pose")
print("    4. hand control to you at the keys listed below")
print()
print("  BEFORE YOU CONTINUE")
print("    - clear the workspace of hands, tools and cables")
print("    - keep the e-stop within reach for the whole run")
if HOME_ON_START:
    print("    - hand-move both arms ROUGHLY to the start pose; homing is")
    print(f"      refused if any joint is more than {HOME_MAX_DELTA_RAD:.1f} rad away")
print("    - run with --help to see the flags, --dry-home to home and exit")
print("=" * 68)
print()

print("Detecting firmware profiles...")
leader_profile = detect_profile(LEADER_CHANNEL)
follower_profile = detect_profile(FOLLOWER_CHANNEL)

leader = build(LEADER_CHANNEL, leader_profile)
follower = build(FOLLOWER_CHANNEL, follower_profile)
leader.connect()
follower.connect()
time.sleep(CONNECT_SETTLE_S)

leader_gripper = leader.init_effector(leader.OPTIONS.EFFECTOR.AGX_GRIPPER)
follower_gripper = follower.init_effector(follower.OPTIONS.EFFECTOR.AGX_GRIPPER)

print(f"Leader={LEADER_CHANNEL}  Follower={FOLLOWER_CHANNEL}")
print("Enabling both arms in 3 s — clear the workspace.")
time.sleep(3)

leader.enable()
follower.enable()
time.sleep(0.5)

follower.set_follower_mode()          # accepts move_js
leader.set_follower_mode()            # starts LOCKED, gravity comp OFF
time.sleep(0.5)

leader_gripper.move_gripper_m(value=LEADER_GRIP_TARGET, force=LEADER_GRIP_FORCE)
print(f"[init] Leader gripper held at force={LEADER_GRIP_FORCE} — jaws hand-free.")

# ---- go to the start pose ---------------------------------------------------
# Both arms are LOCKED here (set_follower_mode above) and enabled, which is the
# only state in which they will hold a commanded pose. Nothing below this point
# assumes homing succeeded -- if it is refused, teleop still runs from wherever
# the arms are.
homed = False
if HOME_ON_START:
    _home_targets = [("leader", leader)]
    if HOME_FOLLOWER_TOO:
        _home_targets.append(("follower", follower))
    else:
        print("[home] HOME_FOLLOWER_TOO is off — the follower will jump to the "
              "leader's pose on the first loop tick.")
    print(f"\n*** MOVING TO START POSE in {HOME_WARN_S:.0f} s — hands clear ***")
    for _i in range(int(HOME_WARN_S), 0, -1):
        print(f"  {_i}...")
        time.sleep(1.0)
    homed = ramp_to_pose(_home_targets, HOME_POSE)

if DRY_HOME:
    if homed:
        print("\n[dry-home] Start pose reached. Shutting down without teleop.")
        print("[dry-home] If the settle numbers above look right, re-run "
              "without the flag.")
    elif HOME_ON_START:
        print("\n[dry-home] HOMING DID NOT RUN — see the reason above. "
              "The arms have not moved.")
    else:
        print("\n[dry-home] --no-home was also given, so there was nothing "
              "to do.")

gc_on = False
logger = RateLimitedLogger(ERR_LOG_PERIOD_S)
last_reassert = 0.0
last_toggle = 0.0
recorder = Recorder(RECORD_DIR, RECORD_FOLLOWER, H5_COMPRESSION)
camera = CameraSource(recorder)
camera.start()

if not DRY_HOME:
    _gc_name = "SPACE" if GC_TOGGLE_KEY == " " else GC_TOGGLE_KEY
    print()
    print("=" * 68)
    print("  READY — DEMO COLLECTION LOOP")
    print("=" * 68)
    print(f"    {HOME_KEY}     go to the start pose (both arms)")
    print(f"    {REC_START_KEY}     start a demo  — opens a new episode")
    print(f"    {REC_END_KEY}     end a demo    — writes the episode to disk")
    print(f"    {DISCARD_KEY}     discard the demo in progress instead of saving")
    print(f"    {POSE_PRINT_KEY}     print the leader's pose as a HOME_POSE line")
    print(f"    {_gc_name} toggle gravity comp by hand (not normally needed)")
    print("    q     quit")
    print()
    print(f"  Normal cycle:  {HOME_KEY} -> {REC_START_KEY} -> demonstrate -> "
          f"{REC_END_KEY} -> {HOME_KEY} -> {REC_START_KEY} -> ...")
    print("  Every demo is written as its own ep_NNNN_<timestamp>_*.h5 pair.")
    print()
    print("  Gravity comp stays ON throughout — it is NOT tied to recording.")
    print(f"  {HOME_KEY} drops it for the move and puts it back on its own.")
    print(f"  Gripper mirrors at {GRIP_GAIN:g}x the whole time, GC or not.")
    if homed:
        print()
        print(f"  Homed to {[round(v, 3) for v in HOME_POSE]}")
        print(f"  Press {POSE_PRINT_KEY} to confirm the arms actually held it.")
    elif HOME_ON_START:
        print()
        print("  NOT HOMED — see the reason above. Move the arms closer by hand")
        print(f"  and press {HOME_KEY}, or demos will start from the wrong pose.")
    if camera.enabled:
        print()
        print(f"  Camera: colour{' + depth' if CAMERA_DEPTH else ''} "
              f"@ {CAMERA_FPS}fps, JPEG q{JPEG_QUALITY}")
    elif CAMERA:
        print()
        print("  Camera: DISABLED")
    print("=" * 68)
print()


def engage_gc():
    """Locked -> weightless. Does NOT touch recording."""
    global gc_on
    leader.set_leader_mode()
    time.sleep(MODE_SETTLE_S)
    gc_on = True
    print("[CLUTCH] Gravity Comp ON  — leader backdrivable, follower tracking")


def release_gc():
    """Weightless -> locked, latched at the pose it is physically in."""
    global gc_on
    leader.set_follower_mode()
    time.sleep(MODE_SETTLE_S)
    here = unwrap(_try("leader.get_joint_angles", leader.get_joint_angles))
    if here is not None:
        _try("leader.move_js (re-latch)", leader.move_js, list(here))
    gc_on = False
    print("[CLUTCH] Gravity Comp OFF — leader locked")


def go_home(warn_s=HOME_WARN_MID_S):
    """Drive both arms to HOME_POSE, then put the clutch back as it was.

    The ramp needs the leader LOCKED. move_js only holds a pose in follower
    mode, and ramp_to_pose reads its start point from get_joint_angles(), which
    is FROZEN under gravity comp -- homing from a stale start point would ramp
    from a pose the arm is not in. So GC is dropped for the duration.

    Restoring the clutch happens in a finally: a refused ramp must not leave you
    holding a locked arm you believe is weightless.
    """
    if recorder.active:
        print(f"[home] a demo is recording — press {REC_END_KEY} to end it "
              f"(or {DISCARD_KEY} to discard) before homing")
        return False

    was_gc = gc_on
    try:
        if gc_on:
            release_gc()
        if warn_s > 0:
            print(f"*** MOVING TO START POSE in {warn_s:.0f} s — hands clear ***")
            for i in range(int(warn_s), 0, -1):
                print(f"  {i}...")
                time.sleep(1.0)
        targets = [("leader", leader)]
        if HOME_FOLLOWER_TOO:
            targets.append(("follower", follower))
        return ramp_to_pose(targets, HOME_POSE)
    finally:
        if was_gc and not gc_on:
            engage_gc()


def start_demo():
    if recorder.active:
        print(f"[rec] already recording — {REC_END_KEY} ends it")
        return
    if not gc_on:
        # Recording a locked leader captures a stationary arm. Refuse rather
        # than auto-engaging: GC makes the arm go weightless, which is not
        # something to spring on whoever is standing next to it.
        print(f"[rec] gravity comp is OFF — hold the leader and press "
              f"{'SPACE' if GC_TOGGLE_KEY == ' ' else GC_TOGGLE_KEY} first")
        return
    recorder.start()


def end_demo():
    if not recorder.active:
        print(f"[rec] not recording — {REC_START_KEY} starts a demo")
        return
    recorder.stop(save=True)


# ---- hand over --------------------------------------------------------------
# GC comes on here, once, and normally stays on for the whole session. The
# operator's hand should already be on the leader: engaging GC makes it
# weightless, and an arm that was holding itself a moment ago now will not.
if START_WITH_GC and not DRY_HOME:
    print()
    print(f"*** ENGAGING GRAVITY COMP in {GC_WARN_S:.0f} s — PUT A HAND ON "
          f"THE LEADER ***")
    for _i in range(int(GC_WARN_S), 0, -1):
        print(f"  {_i}...")
        time.sleep(1.0)
    _try("engage_gc", engage_gc)
    print(f"\nReady. Press {REC_START_KEY} to record your first demo.")


# --------------------------------------------------------------- main loop ---
next_tick = time.perf_counter()
try:
    with Keyboard() as kb:
        while not DRY_HOME:
            now = time.perf_counter()

            # ---- keys --------------------------------------------------------
            key = kb.get()
            k = key.lower() if key is not None and key != " " else key

            if k in ("q",):
                print("\nQuit requested.")
                break

            elif k == HOME_KEY:
                go_home()

            elif k == REC_START_KEY:
                start_demo()

            elif k == REC_END_KEY:
                end_demo()

            elif k == DISCARD_KEY:
                if recorder.active:
                    recorder.stop(save=False)
                else:
                    print("[rec] nothing to discard")

            elif k == POSE_PRINT_KEY:
                # Same reader rule as the tracking block: the other one is frozen.
                src = (leader.get_leader_joint_angles if gc_on
                       else leader.get_joint_angles)
                here = unwrap(_try("pose print", src))
                if here is None:
                    print("[pose] no leader joint frame")
                else:
                    print(f"[pose] HOME_POSE = {format_pose(here)}")

            elif k == GC_TOGGLE_KEY and now - last_toggle >= MIN_TOGGLE_GAP_S:
                last_toggle = now
                try:
                    release_gc() if gc_on else engage_gc()
                except Exception as exc:
                    print(f"[warn] mode switch: {exc.__class__.__name__}: {exc}")

            leader_state = None
            leader_w = None

            # ---- joint tracking: pick the reader that is live in this mode ----
            try:
                if gc_on:
                    state = unwrap(leader.get_leader_joint_angles())
                else:
                    state = unwrap(leader.get_joint_angles())
                if state is not None:
                    leader_state = list(state)
                    follower.move_js(leader_state)
                else:
                    logger.log("track", "[warn] no leader joint frame this tick")
            except Exception as exc:
                logger.log("move", f"[warn] tracking: {exc.__class__.__name__}: {exc}")

            # ---- gripper: mirrored in BOTH clutch states ----------------------
            # Reader switches with the clutch, same as the joint readers:
            # locked -> get_gripper_status(), GC on -> get_gripper_ctrl_states().
            if now - last_reassert >= GRIP_REASSERT_S:
                last_reassert = now
                _try("leader grip re-assert", leader_gripper.move_gripper_m,
                     value=LEADER_GRIP_TARGET, force=LEADER_GRIP_FORCE)
            try:
                leader_w = read_leader_width(gc_on)
                if leader_w is None:
                    logger.log("gripnone", "[warn] no gripper frame this tick")
                elif MIRROR_GRIPPER:
                    follower_gripper.move_gripper_m(
                        value=map_width(leader_w), force=FOLLOWER_GRIP_FORCE)
            except Exception as exc:
                logger.log("grip", f"[warn] gripper: {exc.__class__.__name__}: {exc}")

            # ---- record ------------------------------------------------------
            if recorder.active:
                fj, fw = None, None
                if RECORD_FOLLOWER:
                    fj = unwrap(_try("follower.get_joint_angles",
                                     follower.get_joint_angles))
                    fwm = unwrap(_try("follower gripper",
                                      follower_gripper.get_gripper_status))
                    fw = None if fwm is None else fwm.value
                recorder.sample(leader_state, leader_w, fj, fw)

            # ---- fixed-rate scheduling ---------------------------------------
            next_tick += LOOP_DT
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.perf_counter()

except KeyboardInterrupt:
    print("\nStopping teleoperation...")

finally:
    # Flush an episode that was still running when you quit, then stop the camera.
    _try("recorder.stop", recorder.stop, True)
    _try("camera.stop", camera.stop)

    # Leave leader mode BEFORE disabling — disable() may be ignored otherwise,
    # and the arm must be latched before torque is cut or it collapses.
    _try("leader.set_follower_mode", leader.set_follower_mode)
    time.sleep(MODE_SETTLE_S)
    here = unwrap(_try("leader.get_joint_angles", leader.get_joint_angles))
    if here is not None:
        _try("leader.move_js (hold)", leader.move_js, list(here))

    print(f"\n*** SUPPORT BOTH ARMS — motors cut in {SHUTDOWN_WARN_S:.0f} s ***")
    for i in range(int(SHUTDOWN_WARN_S), 0, -1):
        print(f"  {i}...")
        time.sleep(1.0)

    _try("leader_gripper.disable_gripper", leader_gripper.disable_gripper)
    _try("follower_gripper.disable_gripper", follower_gripper.disable_gripper)
    _try("leader.disable", leader.disable)
    _try("follower.disable", follower.disable)
    _try("leader.disconnect", leader.disconnect)
    _try("follower.disconnect", follower.disconnect)
    print("Safely disconnected.")
