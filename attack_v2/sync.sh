#!/bin/bash
set -e

SCALAR="$1"
ITS="$2"

if [ -z "$SCALAR" ] || [ -z "$ITS" ]; then
    echo "Usage: ./sync.sh <scalar-hex> <ITS>"
    exit 1
fi

echo "[INFO] Running scalar=$SCALAR ITS=$ITS"

pkill -u "$USER" ecc 2>/dev/null || true
pkill -u "$USER" attacker 2>/dev/null || true
rm -f pipe.fifo

mkfifo pipe.fifo
make -s

# Start ECC victim on core 0
taskset -c 0 ./ecc M "$ITS" "$SCALAR" &

sleep 0.05

# Run attacker+spy
taskset -c 24 ./attacker

rm -f pipe.fifo

# (NOTE) raw_timings is NOT deleted here.
# Python deletes or recreates it once per full run.
