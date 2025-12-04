#!/bin/bash
set -e

REPEATS=1      # perf repetitions per M
TOTAL_ITERS=5   # number of full sweeps of M=1..200
OUT="rob_pressure_results_5iters.csv"

# Write header ONLY if file does not exist
if [[ ! -f "$OUT" ]]; then
    echo "iter,M,cycles_mean,cycles_stdev,instr_mean,instr_stdev,elim_mean,elim_stdev,notelim_mean,notelim_stdev,time_mean,time_stdev" > "$OUT"
fi

# -------- safe stats function --------
stats() {
    arr=("$@")
    n=${#arr[@]}

    if [[ $n -eq 0 ]]; then
        echo "0 0"
        return
    fi

    sum=0
    for v in "${arr[@]}"; do 
        sum=$(echo "$sum + $v" | bc)
    done
    mean=$(echo "scale=12; $sum / $n" | bc)

    var=0
    for v in "${arr[@]}"; do
        var=$(echo "$var + ($v - $mean)^2" | bc)
    done
    stdev=$(echo "scale=12; sqrt($var/$n)" | bc)

    echo "$mean $stdev"
}

# -------- outer loop (multiple sweeps) --------
for iter in $(seq 1 $TOTAL_ITERS); do
    echo "=== ITER $iter ==="

    for M in $(seq 1 200); do
        echo "iter=$iter M=$M"

        cycles=()
        instr=()
        elim=()
        notelim=()
        times=()

        for r in $(seq 1 $REPEATS); do
            PERF_OUT=$(taskset -c 0 perf stat \
                -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
                ./ROB_move_elim_test $M 2>&1)

            C=$(echo "$PERF_OUT" | awk '/cycles/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            I=$(echo "$PERF_OUT" | awk '/instructions/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            E=$(echo "$PERF_OUT" | awk '/move_elimination.int_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            N=$(echo "$PERF_OUT" | awk '/move_elimination.int_not_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            T=$(echo "$PERF_OUT" | awk '/seconds time elapsed/ {print $1}')

            C=${C:-0}
            I=${I:-0}
            E=${E:-0}
            N=${N:-0}
            T=${T:-0}

            cycles+=("$C")
            instr+=("$I")
            elim+=("$E")
            notelim+=("$N")
            times+=("$T")
        done

        read cmean cstd < <(stats "${cycles[@]}")
        read imean istd < <(stats "${instr[@]}")
        read emean estd < <(stats "${elim[@]}")
        read nmean nstd < <(stats "${notelim[@]}")
        read tmean tstd < <(stats "${times[@]}")

        echo "$iter,$M,$cmean,$cstd,$imean,$istd,$emean,$estd,$nmean,$nstd,$tmean,$tstd" >> "$OUT"
    done
done
