#!/usr/bin/env python3
"""
OAK-D check. Opens colour (and optionally depth), reports the true frame rate
and device timestamps, saves one frame of each.

Written against DepthAI v3 (Camera node, requestOutput, createOutputQueue).
v2 tutorials will not run on this API.

Deliberately uses NON-BLOCKING reads. In the teleop loop a blocking q.get()
would stall the 50Hz control loop whenever a frame is late, so the pattern to
learn here is tryGet().
"""

import time

import cv2
import depthai as dai

RGB_SIZE = (640, 400)
WANT_DEPTH = True
RUN_S = 15.0
SHOW = True          # False if you are on a headless/remote terminal

def describe(info):
    """DeviceInfo field names differ between depthai versions — report what exists."""
    bits = []
    for attr in ("deviceId", "mxid", "name", "state", "protocol", "platform"):
        if hasattr(info, attr):
            val = getattr(info, attr)
            if callable(val):
                try:
                    val = val()
                except Exception:
                    continue
            bits.append(f"{attr}={val}")
    for meth in ("getDeviceId", "getMxId", "getName"):
        if hasattr(info, meth):
            try:
                bits.append(f"{meth}()={getattr(info, meth)()}")
            except Exception:
                pass
    return "  ".join(bits) if bits else repr(info)


print(f"depthai {dai.__version__}")
devices = dai.Device.getAllAvailableDevices()
if not devices:
    raise SystemExit("No OAK device found. Check the USB3 cable and udev rule.")
for d in devices:
    print(f"  found: {describe(d)}")
    print(f"         fields: {[a for a in dir(d) if not a.startswith('_')]}")

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = cam.requestOutput(RGB_SIZE, dai.ImgFrame.Type.BGR888i)
    q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)

    q_depth = None
    if WANT_DEPTH:
        try:
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            left.requestOutput(RGB_SIZE).link(stereo.left)
            right.requestOutput(RGB_SIZE).link(stereo.right)
            stereo.setRectifyEdgeFillColor(0)
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
            q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        except Exception as exc:
            print(f"[warn] depth setup failed, colour only: "
                  f"{exc.__class__.__name__}: {exc}")
            q_depth = None

    pipeline.start()
    print(f"\nStreaming for {RUN_S:.0f}s. Press q in the window to stop early.\n")

    n_rgb = n_depth = 0
    first_ts = last_ts = None
    saved_rgb = saved_depth = False
    t0 = time.monotonic()

    while pipeline.isRunning() and time.monotonic() - t0 < RUN_S:
        frame = q_rgb.tryGet()
        if frame is not None:
            n_rgb += 1
            # Device timestamps are synced to the host steady clock, i.e. they
            # are directly comparable with time.monotonic() -- which is what
            # makes joint/camera alignment possible later.
            ts = frame.getTimestamp().total_seconds()
            if first_ts is None:
                first_ts = ts
                print(f"  first colour frame: {frame.getWidth()}x{frame.getHeight()} "
                      f"seq={frame.getSequenceNum()} ts={ts:.4f}s "
                      f"(monotonic now {time.monotonic():.4f}s)")
            last_ts = ts

            img = frame.getCvFrame()
            if not saved_rgb:
                cv2.imwrite("oakd_rgb.png", img)
                saved_rgb = True
            if SHOW:
                cv2.imshow("rgb", img)

        if q_depth is not None:
            d = q_depth.tryGet()
            if d is not None:
                n_depth += 1
                dep = d.getFrame()          # uint16, millimetres
                if not saved_depth:
                    print(f"  first depth frame: {dep.shape} dtype={dep.dtype} "
                          f"range={dep.min()}-{dep.max()} mm")
                    vis = cv2.applyColorMap(
                        cv2.convertScaleAbs(dep, alpha=0.03), cv2.COLORMAP_JET)
                    cv2.imwrite("oakd_depth.png", vis)
                    saved_depth = True
                if SHOW:
                    cv2.imshow("depth", cv2.applyColorMap(
                        cv2.convertScaleAbs(dep, alpha=0.03), cv2.COLORMAP_JET))

        if SHOW and cv2.waitKey(1) == ord("q"):
            break
        time.sleep(0.001)

    elapsed = time.monotonic() - t0
    print(f"\n  colour: {n_rgb} frames in {elapsed:.1f}s = {n_rgb / elapsed:.1f} FPS")
    if q_depth is not None:
        print(f"  depth : {n_depth} frames in {elapsed:.1f}s = "
              f"{n_depth / elapsed:.1f} FPS")
    if first_ts is not None and n_rgb > 1:
        print(f"  device timestamp span: {last_ts - first_ts:.2f}s "
              f"({(n_rgb - 1) / (last_ts - first_ts):.1f} FPS by device clock)")
    print(f"  saved: oakd_rgb.png{' , oakd_depth.png' if saved_depth else ''}")

if SHOW:
    cv2.destroyAllWindows()
print("Done.")
