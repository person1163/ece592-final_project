#!/bin/bash
set -e

MODE="$1"     # 0 = memcpy-path, 1 = rename-heavy
ITS="$2"      # ignored by victim, but kept if you want to expand later

if [ -z "$MODE" ]; then
    echo "Usage: ./sync.sh <0|1>"
    exit 1
fi

echo "[INFO] Running victim MODE=$MODE"

# Kill old instances
pkill -u "$USER" victim 2>/dev/null || true
pkill -u "$USER" attacker 2>/dev/null || true
rm -f pipe.fifo

mkfifo pipe.fifo
make -s

# Start victim on core 0
taskset -c 0 ./victim "$MODE" &

sleep 0.05

# Run attacker+spy
taskset -c 24 ./attacker

rm -f pipe.fifo
