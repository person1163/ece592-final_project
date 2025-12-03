#!/bin/bash
set -e

RUNS=1000
CSV=results.csv

# Binaries you already built
BIN_O0=./move_elim_demo_O0
BIN_O3=./move_elim_demo_O3

echo "opt_level,run,mode,time_elapsed,user_time,sys_time" > "$CSV"

run_one () {
    opt="$1"      # O0 or O3
    run="$2"
    mode="$3"
    bin="$4"

    # perf writes to stderr → capture it
    out=$(perf stat -e cycles,instructions "$bin" "$mode" 2>&1)

    # Parse values (works on all standard perf versions)
    t_elapsed=$(echo "$out" | grep "seconds time elapsed" | awk '{print $1}')
    t_user=$(echo "$out" | grep "seconds user" | awk '{print $1}')
    t_sys=$(echo "$out" | grep "seconds sys" | awk '{print $1}')

    echo "$opt,$run,$mode,$t_elapsed,$t_user,$t_sys" >> "$CSV"
}

# -----------------------------
# Run O0 binary
# -----------------------------
for mode in 0 1; do
    for ((r=1; r<=RUNS; r++)); do
        echo "[RUN] O0 mode=$mode run=$r"
        run_one "O0" "$r" "$mode" "$BIN_O0"
    done
done

# -----------------------------
# Run O3 binary
# -----------------------------
for mode in 0 1; do
    for ((r=1; r<=RUNS; r++)); do
        echo "[RUN] O3 mode=$mode run=$r"
        run_one "O3" "$r" "$mode" "$BIN_O3"
    done
done

echo "Saved results → $CSV"
