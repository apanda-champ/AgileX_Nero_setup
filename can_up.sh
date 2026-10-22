#!/usr/bin/env bash
# Bring both USB-to-CAN interfaces up at 1 Mbit.
#
#   ./can_up.sh                     bring can0 and can1 up now
#   ./can_up.sh --install-service   also do it automatically at boot
#   ./can_up.sh --down              take them down
#
# Run after every reboot or replug, unless the service is installed.
#
# BITRATE MUST BE 1000000. The Nero arms and the AgileX gripper both run at
# 1 Mbit; a mismatched bitrate shows up as silence or error frames, not as a
# clear failure.

set -uo pipefail

BITRATE=1000000
IFACES=(can0 can1)

usage() { sed -n '2,12p' "$0"; exit 0; }

case "${1:-}" in
    -h|--help) usage ;;
esac

if [[ "${1:-}" == "--down" ]]; then
    for i in "${IFACES[@]}"; do
        sudo ip link set "$i" down 2>/dev/null && echo "  $i down" \
            || echo "  $i not present"
    done
    exit 0
fi

echo "=== bringing CAN interfaces up at ${BITRATE} bit/s ==="
missing=0
for i in "${IFACES[@]}"; do
    if ! ip link show "$i" >/dev/null 2>&1; then
        echo "  $i  NOT PRESENT — adapter unplugged, or gs_usb/slcan not loaded"
        missing=1
        continue
    fi
    sudo ip link set "$i" down 2>/dev/null || true
    if sudo ip link set "$i" type can bitrate "$BITRATE" && sudo ip link set "$i" up; then
        state=$(ip -details link show "$i" | grep -o 'state [A-Z]*' | head -1)
        echo "  $i  up at ${BITRATE}  ($state)"
    else
        echo "  $i  FAILED to configure"
        missing=1
    fi
done

echo
echo "=== which adapter is which ==="
echo "  can1 should be the LEADER, can0 the FOLLOWER."
echo "  Names come from USB enumeration order, so they can swap if you replug"
echo "  in a different order. If teleop behaves oddly, confirm with:"
echo "      python3 identify_channels_live.py"
echo
echo "  Traffic check (an arm must be powered):"
for i in "${IFACES[@]}"; do
    echo "      candump $i"
done

if [[ "${1:-}" == "--install-service" ]]; then
    echo
    echo "=== installing can-up.service ==="
    sudo tee /etc/systemd/system/can-up.service >/dev/null <<EOF
[Unit]
Description=Bring up SocketCAN interfaces at ${BITRATE} bit/s
After=network.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'for i in ${IFACES[*]}; do \
  ip link set \$i down 2>/dev/null || true; \
  ip link set \$i type can bitrate ${BITRATE} 2>/dev/null || true; \
  ip link set \$i up 2>/dev/null || true; done'

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable can-up.service
    echo "  enabled. The interfaces come up at boot from now on."
    echo "  Note: it silently skips an adapter that is not plugged in."
fi

exit $missing
