#!/bin/bash
set -x   # <--- ADD THIS

echo "PWD = $(pwd)"
echo "FILES:"
ls -l

# Kill only YOUR processes, not the entire cluster's
pkill -u $USER ecc 2>/dev/null
pkill -u $USER attacker 2>/dev/null

rm -f pipe.fifo
mkfifo pipe.fifo

make -s

# (1) Start victim blocked on fread()
taskset -c 0 ./ecc M 1000 $1 &

sleep 0.05

# (2) Run perf + attacker (attacker wakes victim *during* its startup)
taskset -c 24 perf stat -e cycles,instructions,uops_issued.any,idq_uops_not_delivered.core ./attacker

rm -f pipe.fifo
pkill attacker

# ./sync.sh FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
# ./sync.sh 00000000000000000000000000000000