# AgileX Nero — Leader-Follower Teleoperation & Behaviour Cloning

Hardware-in-the-loop teleoperation for two 7-DoF AgileX Nero arms, with
episode recording, an OAK-D camera, and a behaviour-cloning pipeline.

A human backdrives the **leader** arm under gravity compensation; the
**follower** mirrors its joints and gripper at 50 Hz over SocketCAN.
Demonstrations are written to HDF5 and used to train a policy that is then
deployed on the follower.

---

## Hardware

| | |
|---|---|
| Arms | 2 × AgileX Nero, 7-DoF, firmware 1.21 |
| Grippers | 2 × AgxGripper, feedback CAN ID `0x51A` |
| Camera | Luxonis OAK-D (colour + stereo depth) |
| Interface | 2 × USB-to-CAN, SocketCAN at 1 Mbit |
| Host | Ubuntu 22.04, Python 3.10 |

**Channel assignment:** `can1` = leader, `can0` = follower.
Names come from USB enumeration order and can swap between machines —
confirm with `identify_channels_live.py` before the first run.

---

## Install

```bash
git clone git@github.com:YOURNAME/AgileX_Nero_setup.git
cd AgileX_Nero_setup
chmod +x setup_ubuntu2204.sh can_up.sh
./setup_ubuntu2204.sh
```

Installs APT packages, CAN kernel modules, DepthAI dependencies, udev rules,
a `teleop_env` virtualenv, and pyAgxArm from GitHub.

Verify the SDK is new enough for firmware 1.21:

```bash
source teleop_env/bin/activate
python -c "from pyAgxArm import NeroFW; print(NeroFW.V121)"
```

An `AttributeError` here means the pyAgxArm checkout predates 2026-09, and the
leader's state will be unreadable under gravity compensation.

Bring the CAN buses up after every reboot or replug:

```bash
./can_up.sh                     # both interfaces at 1 Mbit
./can_up.sh --install-service   # or once, to do it at boot
```

---

## Collecting demonstrations

```bash
./nero_teleop_spacebar.py
```

Both arms move to `HOME_POSE`, then gravity compensation engages — **keep a
hand on the leader** through the countdown.

| key | action |
|---|---|
| `h` | go to the start pose (both arms) |
| `r` | start a demo |
| `e` | end the demo and write it |
| `d` | discard the demo in progress |
| `p` | print the leader's pose as a `HOME_POSE` line |
| `SPACE` | toggle gravity compensation by hand |
| `q` | quit |

Normal cycle: `h` → `r` → demonstrate → `e` → `h` → `r` → …

Flags: `--no-home`, `--no-camera`, `--no-record`, `--dry-home`, `--help`.

Each demo writes two files:

```
episodes/ep_0001_20260914_143022_joints.h5     50 Hz joints + gripper
episodes/ep_0001_20260914_143022_camera.h5     30 Hz JPEG + depth
```

Both carry the same `t0_monotonic`, so their timestamps are directly
comparable and can be paired offline.

---

## Inspecting and replaying

```bash
python3 view_episode.py episodes/ep_0001_*_camera.h5 --depth
python3 replay_episode.py episodes/ep_0001_*_joints.h5 --dry-run
python3 replay_episode.py episodes/ep_0001_*_joints.h5 --speed 0.5
```

Replay drives the follower with the recorded leader command stream, homing
first and counting down before it moves. **Stop the teleop script first** —
two processes commanding one arm fight over every frame.

Replay is the cheapest end-to-end test available: if it reproduces the demo,
the timestamps, action stream and command path are all sound, and any later
failure belongs to the policy.

---

## Training a policy

Current pipeline is **proprioception-only** vanilla behaviour cloning:
follower state in, leader action out, MSE loss, single step.

```bash
python3 bc_dataset.py episodes/            # inventory + normalisation stats
python3 bc_train.py   episodes/ --epochs 300
python3 bc_deploy.py  checkpoints/best.pt --dry-run
python3 bc_deploy.py  checkpoints/best.pt
```

Training prints an **identity baseline** — the loss you would get by
predicting `action = qpos`. The follower tracks the leader closely by design,
so a network that learns the identity function scores very well and does
nothing useful. Beat that number, or the policy has not learned the task.

For a vision policy, pair the two streams onto one timebase first:

```bash
python3 pair_episodes.py episodes/ --rate 30
```

---

## Files

| file | purpose |
|---|---|
| `setup_ubuntu2204.sh` | one-shot host setup |
| `can_up.sh` | bring both CAN interfaces up at 1 Mbit |
| `nero_teleop_spacebar.py` | teleoperation + episode recording |
| `identify_channels_live.py` | which arm is on which CAN channel |
| `oakd_check.py` | camera smoke test |
| `view_episode.py` | play back a recorded episode |
| `replay_episode.py` | replay a demo on the follower |
| `pair_episodes.py` | align camera and joint streams offline |
| `bc_dataset.py` | dataset + normalisation stats |
| `bc_train.py` | policy and training loop |
| `bc_deploy.py` | run a trained policy on the follower |

---

## Notes from the bring-up

Things that cost real debugging time and are easy to hit again:

- **Firmware profile.** Firmware 1.21 needs `NeroFW.V121`. Under
  `NeroFW.DEFAULT` the leader's joint feedback silently freezes in gravity-comp
  mode — well-formed stale data, no error.
- **Readers switch with the clutch.** `get_joint_angles()` is live when locked
  and frozen under GC; `get_leader_joint_angles()` is the reverse. The gripper
  readers (`get_gripper_status()` / `get_gripper_ctrl_states()`) behave the same
  way. Using the wrong one returns stale values rather than failing.
- **`move_js` has no trajectory planning.** A single call with a distant target
  makes the arm snap toward it. Long moves are ramped on the host.
- **Gripper CAN ID.** The AgxGripper ships at `0x517`/`0x518` and must be
  changed to `0x51A` to integrate with Nero.
- **Gripper gain.** `GRIP_GAIN = 1.5` scales leader jaw travel to the follower.
  The follower saturates once the leader passes ~0.067 m. Episodes recorded
  under different gains are not comparable — keep them in separate folders.

---

## Safety

- Clear the workspace before enabling. Both arms are powered and stiff.
- Keep the e-stop within reach for every run.
- Motors are cut 3 seconds after exit — **support both arms** during the
  countdown or they fall.
- A learned policy will command confidently when the scene drifts outside its
  training distribution. `bc_deploy.py` clamps per-tick joint deltas, checks the
  start pose, limits runtime and binds `x` to an emergency stop. Those limit
  speed; they do not make a bad policy safe.
