#!/bin/bash
# Split the B60 into VFs, sharing resources evenly (the xe driver's default split).
# Run as root:  sudo ./split_gpu.sh [number_of_vfs]     (default 7)
set -euo pipefail

PF=0000:09:00.0
NUM=${1:-7}
SYS=/sys/bus/pci/devices/$PF

[ "$(id -u)" -eq 0 ] || { echo "run as root"; exit 1; }
TOTAL=$(cat "$SYS/sriov_totalvfs")
[ "$TOTAL" -gt 0 ] || { echo "PF offers no VFs; boot 6.17 with xe.max_vfs=7"; exit 1; }
[ "$NUM" -ge 1 ] && [ "$NUM" -le "$TOTAL" ] || { echo "number of VFs must be 1-$TOTAL"; exit 1; }

# Writing a new nonzero count while VFs exist fails, so clear them first.
echo 0 > "$SYS/sriov_numvfs"
echo "$NUM" > "$SYS/sriov_numvfs"

dmesg | grep -iE "xe.*(sriov|vf)" | tail -"$((NUM + 10))"
lspci -nn | grep e211
echo "VFs enabled: $(cat "$SYS/sriov_numvfs")"
