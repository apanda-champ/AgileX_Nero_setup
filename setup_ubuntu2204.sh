#!/usr/bin/env bash
# Nero teleop rig — fresh Ubuntu 22.04 setup.
#
#   chmod +x setup_ubuntu2204.sh
#   ./setup_ubuntu2204.sh
#
# Does NOT install pyAgxArm (vendor SDK, supplied by AgileX) — see step 5.
# Re-runnable: every step is idempotent.

set -euo pipefail

echo "=== 1. system packages ==="
sudo apt update
sudo apt install -y \
    build-essential git curl wget \
    python3 python3-venv python3-pip python3-dev \
    can-utils iproute2 \
    libusb-1.0-0-dev udev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libxcb-xinerama0 libxkbcommon-x11-0

# libgl1/libglib2.0-0 and the libxcb/libxkb pair are what cv2.imshow needs.
# Without them OpenCV imports fine and then fails only when you open a window.

echo
echo "=== 2. DepthAI system dependencies ==="
wget -qO- https://docs.luxonis.com/install_dependencies.sh | bash

echo
echo "=== 3. udev rules ==="
# OAK-D (Movidius Myriad X, vendor 03e7) reachable without root
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/80-movidius.rules >/dev/null
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "  udev rules reloaded"

echo
echo "=== 3a. serial port access ==="
# slcan-style USB-CAN adapters appear as /dev/ttyACM* or /dev/ttyUSB* and need
# dialout membership. gs_usb adapters do not, but joining costs nothing.
if id -nG "$USER" | grep -qw dialout; then
    echo "  $USER already in dialout"
else
    sudo usermod -aG dialout "$USER"
    echo "  added $USER to dialout — LOG OUT AND BACK IN for this to take effect"
fi

echo
echo "=== 3b. CAN kernel modules ==="
# gs_usb covers CANable/candleLight adapters; slcan covers serial-style ones.
# Ubuntu 22.04's 5.15 kernel ships both, they just need loading.
for m in can can_raw gs_usb; do
    sudo modprobe "$m" 2>/dev/null && echo "  loaded $m" || echo "  [warn] $m not available"
done
printf 'can\ncan_raw\ngs_usb\n' | sudo tee /etc/modules-load.d/can.conf >/dev/null
echo "  set to load at boot via /etc/modules-load.d/can.conf"

echo
echo "=== 4. python environment ==="
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "  [warn] already inside ${VIRTUAL_ENV} — deactivate first, or the"
    echo "         new venv inherits confusing state. Continuing anyway."
fi
python3 -m venv teleop_env
# shellcheck disable=SC1091
source teleop_env/bin/activate
pip install --upgrade pip wheel
pip install \
    "python-can>3.3.4" \
    "numpy>=1.24" \
    "h5py>=3.8" \
    "depthai>=3.0" \
    "opencv-python>=4.8" \
    "torch>=2.0"

# pyAgxArm is on GitHub, not PyPI. It needs a post-2026-09 revision for the
# v121 driver (firmware >= 1.21), so install from main rather than a pin.
pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"

echo
echo "=== 5. pyAgxArm check ==="
# Installed from GitHub in step 4. The V121 attribute is the thing that
# matters: firmware 1.21 is unreadable under gravity comp without it.
python3 - <<'PY2'
try:
    import pyAgxArm
    from pyAgxArm import NeroFW
    print(f"  pyAgxArm at {pyAgxArm.__file__}")
    print(f"  NeroFW.V121 = {NeroFW.V121!r}  <- required for firmware 1.21")
except ImportError as e:
    print(f"  NOT INSTALLED: {e}")
    print("  pip install \"git+https://github.com/agilexrobotics/pyAgxArm.git\"")
except AttributeError:
    print("  INSTALLED BUT TOO OLD — no NeroFW.V121.")
    print("  pip install --upgrade --force-reinstall \\")
    print("      \"git+https://github.com/agilexrobotics/pyAgxArm.git\"")
PY2

echo
echo "=== 6. CAN interfaces ==="
if [[ -x ./can_up.sh ]]; then
    ./can_up.sh || echo "  (adapters not plugged in yet — rerun ./can_up.sh later)"
else
    echo "  can_up.sh not found or not executable — chmod +x can_up.sh"
fi
cat <<'NOTE'
  Plug in both USB-to-CAN adapters, then bring each up at 1 Mbit:

      sudo ip link set can0 down 2>/dev/null
      sudo ip link set can0 type can bitrate 1000000
      sudo ip link set can0 up

      sudo ip link set can1 down 2>/dev/null
      sudo ip link set can1 type can bitrate 1000000
      sudo ip link set can1 up

  Check:  ip -details link show can0
          candump can0          # frames should stream once an arm is powered

  Or just run ./can_up.sh, which does both interfaces and reports their state.
  That script also has an --install-service flag to bring them up at boot.
NOTE

echo
echo "=== 6b. .gitignore ==="
if [[ -e .gitignore ]]; then
    echo "  .gitignore already present — left alone"
else
    cat > .gitignore <<'GI'
# virtualenv — rebuildable from requirements.txt, and its paths are absolute
teleop_env/
venv/
.venv/

# recorded data — large binaries, keep them out of git
episodes/
aligned/
checkpoints/
*.h5
*.mp4
*.pt

# probe artefacts
oakd_rgb.png
oakd_depth.png

# python
__pycache__/
*.py[cod]
*.egg-info/

# editors
.vscode/
.idea/
GI
    echo "  wrote .gitignore"
fi

echo
echo "=== 7. verify ==="
python3 - <<'PY'
mods = ["numpy", "h5py", "cv2", "torch"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  {m:8} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:8} MISSING: {e}")
try:
    import depthai as dai
    print(f"  depthai  {dai.__version__}")
    devs = dai.Device.getAllAvailableDevices()
    print(f"  OAK devices found: {len(devs)}")
    for d in devs:
        print(f"    {getattr(d, 'deviceId', '?')}  {d.state}  {d.protocol}")
except Exception as e:
    print(f"  depthai  problem: {e}")
try:
    import pyAgxArm
    print("  pyAgxArm ok")
except Exception:
    print("  pyAgxArm NOT INSTALLED — see step 5")
PY

echo
echo "Done. Activate with:  source teleop_env/bin/activate"
