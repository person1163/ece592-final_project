#!/bin/bash

make clean
make

pkill attacker
pkill ecc
rm -f pipe.fifo

# 1. Create pipe for synchronization
mkfifo pipe.fifo

# 2. Start ECC victim (blocked on pipe)
taskset -c 0 ./ecc M 1000 $1 &   # $1 = scalar you pass

sleep 0.1   # Let victim reach fread() and block

# 3. Run attacker with perf AND wake the victim (pipe sync)
taskset -c 24 perf stat -e \
    move_elimination.int_eliminated,\
    move_elimination.int_not_eliminated,\
    cycles,instructions,\
    uops_issued.any,\
    idq_uops_not_delivered.core \
    ./attacker

# 4. Cleanup
rm -f pipe.fifo
pkill attacker

#./sync.sh FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
#./sync_me.sh 000000000000000000000000000000000001

