#!/bin/bash
set -e

REPEATS=1       # how many repeated perf runs per P
TOTAL_ITERS=5     # how many full sweeps of P=0..64
OUT="reg_pressure_raw.csv"

# Write header once
if [[ ! -f "$OUT" ]]; then
    echo "iter,P,repeat,cycles,instr,elim,notelim,time" > "$OUT"
fi

for iter in $(seq 1 $TOTAL_ITERS); do
    echo "=== ITER $iter ==="

    for P in $(seq 0 64); do
        echo "ITER=$iter P=$P"

        for r in $(seq 1 $REPEATS); do

            PERF_OUT=$(taskset -c 0 perf stat \
              -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
              ./reg_pressure $P 2>&1 || true)

            # Extract raw values safely
            C=$(echo "$PERF_OUT" | awk '/cycles/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            I=$(echo "$PERF_OUT" | awk '/instructions/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            E=$(echo "$PERF_OUT" | awk '/int_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            N=$(echo "$PERF_OUT" | awk '/int_not_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            T=$(echo "$PERF_OUT" | awk '/seconds time elapsed/ {print $1}')

            # Default to 0 if parsing failed
            C=${C:-0}
            I=${I:-0}
            E=${E:-0}
            N=${N:-0}
            T=${T:-0}

            # Append raw data
            echo "$iter,$P,$r,$C,$I,$E,$N,$T" >> "$OUT"
        done
    done
done
