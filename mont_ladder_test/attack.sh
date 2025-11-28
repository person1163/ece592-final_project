#!/bin/bash

SPY_CPU=24
VICTIM_CPU=0

mkfifo pipe.fifo 2>/dev/null

taskset -c $SPY_CPU ./spy &
SPYPID=$!

sleep 0.02

taskset -c $VICTIM_CPU ./mont_ladder 1

kill -9 $SPYPID
