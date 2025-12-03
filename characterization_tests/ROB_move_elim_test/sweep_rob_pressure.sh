#!/bin/bash
set -e

REPEATS=10
OUT="rob_pressure_results.csv"

echo "M,cycles_mean,cycles_stdev,instr_mean,instr_stdev,elim_mean,elim_stdev,notelim_mean,notelim_stdev,time_mean,time_stdev" > "$OUT"

# safe stats function
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
    var=$(echo "scale=12; $var / $n" | bc)
    stdev=$(echo "scale=12; sqrt($var)" | bc)

    echo "$mean $stdev"
}

for M in $(seq 1 200); do
    echo "M=$M"

    cycles=()
    instr=()
    elim=()
    notelim=()
    times=()

    for r in $(seq 1 $REPEATS); do
        
        PERF_OUT=$(taskset -c 0 perf stat \
            -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
            ./ROB_move_elim_test $M 2>&1)

        # Robust extraction for your cluster perf format
        C=$(echo "$PERF_OUT" | awk '/cycles/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
        I=$(echo "$PERF_OUT" | awk '/instructions/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
        E=$(echo "$PERF_OUT" | awk '/move_elimination.int_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
        N=$(echo "$PERF_OUT" | awk '/move_elimination.int_not_eliminated/ && $1 ~ /^[0-9,]+$/ {gsub(",","",$1); print $1}')
        T=$(echo "$PERF_OUT" | awk '/seconds time elapsed/ {print $1}')

        # Replace empty values with zero
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

    # compute stats
    read cmean cstd < <(stats "${cycles[@]}")
    read imean istd < <(stats "${instr[@]}")
    read emean estd < <(stats "${elim[@]}")
    read nmean nstd < <(stats "${notelim[@]}")
    read tmean tstd < <(stats "${times[@]}")

    echo "$M,$cmean,$cstd,$imean,$istd,$emean,$estd,$nmean,$nstd,$tmean,$tstd" >> "$OUT"
done
