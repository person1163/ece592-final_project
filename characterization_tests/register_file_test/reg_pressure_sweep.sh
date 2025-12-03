#!/bin/bash
set -e

REPEATS=10
OUT="reg_pressure_results.csv"
TOTAL_ITERS=5   # number of full sweeps you want

# write header only once
if [[ ! -f "$OUT" ]]; then
    echo "iter,P,cycles_mean,cycles_stdev,instr_mean,instr_stdev,elim_mean,elim_stdev,notelim_mean,notelim_stdev,time_mean,time_stdev" > "$OUT"
fi

stats() {
    arr=("$@")
    n=${#arr[@]}
    if [[ $n -eq 0 ]]; then echo "0 0"; return; fi
    sum=0
    for v in "${arr[@]}"; do sum=$(echo "$sum + $v" | bc); done
    mean=$(echo "scale=12; $sum/$n" | bc)
    var=0
    for v in "${arr[@]}"; do var=$(echo "$var + ($v - $mean)^2" | bc); done
    stdev=$(echo "scale=12; sqrt($var/$n)" | bc)
    echo "$mean $stdev"
}

# -------------------------
# Outer loop for full sweeps
# -------------------------
for iter in $(seq 1 $TOTAL_ITERS); do
    echo "=== ITER $iter ==="

    for P in $(seq 0 64); do
        echo "iter=$iter P=$P"

        cycles=()
        instrs=()
        elim=()
        notelim=()
        times=()

        for r in $(seq 1 $REPEATS); do

            PERF_OUT=$(taskset -c 1 perf stat \
                -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
                ./reg_pressure $P 2>&1 || true)

            C=$(echo "$PERF_OUT" | awk '/cycles/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            I=$(echo "$PERF_OUT" | awk '/instructions/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            E=$(echo "$PERF_OUT" | awk '/int_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            N=$(echo "$PERF_OUT" | awk '/int_not_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
            T=$(echo "$PERF_OUT" | awk '/seconds time elapsed/ {print $1}')

            C=${C:-0}
            I=${I:-0}
            E=${E:-0}
            N=${N:-0}
            T=${T:-0}

            cycles+=("$C")
            instrs+=("$I")
            elim+=("$E")
            notelim+=("$N")
            times+=("$T")
        done

        read cmean cstd < <(stats "${cycles[@]}")
        read imean istd < <(stats "${instrs[@]}")
        read emean estd < <(stats "${elim[@]}")
        read nmean nstd < <(stats "${notelim[@]}")
        read tmean tstd < <(stats "${times[@]}")

        # append all data
        echo "$iter,$P,$cmean,$cstd,$imean,$istd,$emean,$estd,$nmean,$nstd,$tmean,$tstd" >> "$OUT"

    done
done
