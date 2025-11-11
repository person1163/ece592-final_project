#!/bin/bash
OUTFILE="move_elim_trials.csv"
TRIALS=1000
HIGH="./high_moves"
LOW="./low_moves"

# CSV header
echo "trial,type,int_eliminated,int_not_eliminated,instructions,cycles,time_sec" > $OUTFILE

for i in $(seq 1 $TRIALS); do
  for t in high low; do
    if [ "$t" = "high" ]; then
      EXEC=$HIGH
    else
      EXEC=$LOW
    fi

    perf_output=$(perf stat -x, -e move_elimination.int_eliminated,move_elimination.int_not_eliminated,instructions,cycles $EXEC 2>&1 >/dev/null)
    int_elim=$(echo "$perf_output" | awk -F, '/move_elimination.int_eliminated/ {print $1}')
    int_not_elim=$(echo "$perf_output" | awk -F, '/move_elimination.int_not_eliminated/ {print $1}')
    instr=$(echo "$perf_output" | awk -F, '/instructions/ {print $1}')
    cycles=$(echo "$perf_output" | awk -F, '/cycles/ {print $1}')
    time=$(echo "$perf_output" | awk '/seconds time elapsed/ {print $1}')

    echo "$i,$t,$int_elim,$int_not_elim,$instr,$cycles,$time" >> $OUTFILE
  done
done
