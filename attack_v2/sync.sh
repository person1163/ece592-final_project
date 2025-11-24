#!/bin/bash

make -s

pkill ecc
pkill attacker
rm -f pipe.fifo

mkfifo pipe.fifo

# (1) Start victim blocked on fread()
taskset -c 0 ./ecc M 1000 $1 &

sleep 0.05

# (2) Run perf + attacker (attacker wakes victim *during* its startup)
taskset -c 24 perf stat -e \
    move_elimination.int_eliminated \
    move_elimination.int_not_eliminated \
    cycles \
    instructions \
    uops_issued.any \
    idq_uops_not_delivered.core \
    ./attacker

rm -f pipe.fifo
pkill attacker

# ./sync.sh FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
# ./sync.sh 00000000000000000000000000000000