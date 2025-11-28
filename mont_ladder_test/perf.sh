#!/bin/bash

SPY_CPU=24
VICTIM_CPU=0

# 1. Start attacker under perf with a fixed window (e.g., 0.1 seconds)
(
    perf stat --timeout 10 \
        -e move_elimination.int_eliminated,move_elimination.int_not_eliminated \
        taskset -c 24 ./attack

) &
APID=$!


# 3. Start victim
taskset -c $VICTIM_CPU ./mont_ladder 0 &
VPID=$!

# 4. Wait for perf to finish (timeout kills attacker)
wait $APID

# 5. Kill victim afterward
kill -9 $VPID 2>/dev/null
