#!/usr/bin/env python3
"""
Which physical arm is on can0, and which is on can1?

Watches BOTH channels simultaneously and reports which one is moving.
No poses to set, no numbers to eyeball, no motors enabled.

RUN IT, then grab ONE arm by the wrist and rotate the last joint back and
forth. The channel that lights up is that arm's channel.

Joint 7 (wrist roll) is the one to move: it is light, it cannot drop, and it
needs no arm support. Do not lift the whole arm — the motors are off.
"""

import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

CHANNELS = ("can0", "can1")
WATCH_S = 25.0
SETTLE_S = 2.0
POLL_DT = 0.05
MOVE_THRESHOLD_RAD = 0.05      # per-sample delta that counts as deliberate motion


def make(ch):
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT,
        interface="socketcan", channel=ch))
    arm.connect()
    return arm


def joints(arm):
    st = arm.get_joint_angles()
    msg = getattr(st, "msg", None) if st is not None else None
    return list(msg) if msg is not None else None


arms = {}
try:
    for ch in CHANNELS:
        arms[ch] = make(ch)
    time.sleep(SETTLE_S)

    base = {}
    for ch, arm in arms.items():
        j = joints(arm)
        if j is None:
            print(f"[fatal] {ch}: no joint frame")
            raise SystemExit(1)
        base[ch] = j
    print(f"Baseline captured ({len(base[CHANNELS[0]])} joints per arm).")

    print(f"\n>>> Now rotate ONE arm's WRIST (last joint) back and forth "
          f"for {WATCH_S:.0f}s.\n")

    peak = {ch: 0.0 for ch in CHANNELS}
    travel = {ch: 0.0 for ch in CHANNELS}
    prev = {ch: list(base[ch]) for ch in CHANNELS}

    end = time.perf_counter() + WATCH_S
    last_print = 0.0
    while time.perf_counter() < end:
        for ch, arm in arms.items():
            j = joints(arm)
            if j is None:
                continue
            peak[ch] = max(peak[ch], max(abs(a - b) for a, b in zip(j, base[ch])))
            step = max(abs(a - b) for a, b in zip(j, prev[ch]))
            if step > 0.002:                       # ignore encoder noise
                travel[ch] += step
            prev[ch] = j

        now = time.perf_counter()
        if now - last_print >= 0.5:
            last_print = now
            line = "  ".join(
                f"{ch}: peak={peak[ch]:5.3f} travel={travel[ch]:6.3f} "
                f"{'<== MOVING' if peak[ch] > MOVE_THRESHOLD_RAD else '          '}"
                for ch in CHANNELS)
            print(f"  {line}", end="\r", flush=True)
        time.sleep(POLL_DT)

    print("\n\n" + "=" * 58)
    print("VERDICT")
    print("=" * 58)
    for ch in CHANNELS:
        print(f"  {ch}: peak deflection {peak[ch]:.3f} rad, total travel {travel[ch]:.3f} rad")

    ranked = sorted(CHANNELS, key=lambda c: peak[c], reverse=True)
    hot, cold = ranked[0], ranked[1]
    if peak[hot] < MOVE_THRESHOLD_RAD:
        print("\n  Neither arm moved. Rerun and rotate the wrist further.")
    elif peak[hot] < 3 * max(peak[cold], 1e-6):
        print(f"\n  AMBIGUOUS — {hot} moved most but {cold} moved comparably.")
        print("  You may have bumped both, or the bus mapping is not stable.")
    else:
        print(f"\n  The arm you moved is on {hot}.")
        print(f"  The other arm is on {cold}.")
        print(f"\n  In nero_teleop_clutch_final.py set:")
        print(f"    LEADER_CHANNEL   = \"{hot}\"    (if you moved your leader)")
        print(f"    FOLLOWER_CHANNEL = \"{cold}\"")

finally:
    for arm in arms.values():
        try:
            arm.disconnect()
        except Exception:
            pass
    print("\nDisconnected.")
