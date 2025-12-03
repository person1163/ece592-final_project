#!/bin/bash

REPEATS=2
OUT="move_saturation_results.csv"

# Header
echo "K,cycles_mean,cycles_std,instr_mean,instr_std,elim_mean,elim_std,notelim_mean,notelim_std" > "$OUT"

# ---------- stats ----------
stats() {
    arr=("$@")
    n=${#arr[@]}

    if [[ $n -eq 0 ]]; then
        echo "0 0"
        return
    fi

    # filter to numeric entries only
    local cleaned=()
    for v in "${arr[@]}"; do
        if [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            cleaned+=("$v")
        fi
    done

    n=${#cleaned[@]}
    if [[ $n -eq 0 ]]; then
        echo "0 0"
        return
    fi

    sum=0
    for v in "${cleaned[@]}"; do
        sum=$(echo "$sum + $v" | bc)
    done
    mean=$(echo "scale=12; $sum / $n" | bc)

    var=0
    for v in "${cleaned[@]}"; do
        var=$(echo "$var + ($v - $mean)^2" | bc)
    done
    std=$(echo "scale=12; sqrt($var / $n)" | bc)

    echo "$mean $std"
}

# ---------- sweep ----------
for K in $(seq 1 10); do
    echo "K=$K"

    cycles=()
    instrs=()
    elims=()
    notelims=()

    for r in $(seq 1 $REPEATS); do
        PERF_OUT=$(taskset -c 1 perf stat \
            -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
            ./move_saturation "$K" 2>&1) || true

        # Match ONLY the right lines (your sample output format)
        C=$(echo "$PERF_OUT" | awk '/cycles:u/ {gsub(",","",$1); print $1; exit}')
        I=$(echo "$PERF_OUT" | awk '/instructions:u/ {gsub(",","",$1); print $1; exit}')
        E=$(echo "$PERF_OUT" | awk '/move_elimination.int_eliminated/ {gsub(",","",$1); print $1; exit}')
        N=$(echo "$PERF_OUT" | awk '/move_elimination.int_not_eliminated/ {gsub(",","",$1); print $1; exit}')

        # default to 0 if missing
        C=${C:-0}
        I=${I:-0}
        E=${E:-0}
        N=${N:-0}

        cycles+=("$C")
        instrs+=("$I")
        elims+=("$E")
        notelims+=("$N")
    done

    read cmean cstd < <(stats "${cycles[@]}")
    read imean istd < <(stats "${instrs[@]}")
    read emean estd < <(stats "${elims[@]}")
    read nmean nstd < <(stats "${notelims[@]}")

    echo "$K,$cmean,$cstd,$imean,$istd,$emean,$estd,$nmean,$nstd" >> "$OUT"
done
