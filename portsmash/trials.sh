#!/bin/bash

RUNS=1000
CURVE="secp521r1"
BASEOUT="secp521r1_all_1000"
mkdir -p "$BASEOUT"

CSV="$BASEOUT/results_summary.csv"

# Column header
echo "run,baseline_elim,baseline_notelim,openssl_elim,openssl_notelim,add_elim,add_notelim,double_elim,double_notelim,mul_elim,mul_notelim,ad_elim,ad_notelim" > "$CSV"

# Build binaries once
make clean
make

###############################################
# Microarchitectural flush function
###############################################
flush_cpu_state() {
    # Drop L1/L2/L3 cache residency
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    # Branch predictor flushing (poor man’s version)
    for j in {1..200000}; do :; done

    # DVFS/turbo hysteresis cool-down
    sleep 0.05
}

for i in $(seq 1 $RUNS); do

    RUNID=$(printf "%04d" $i)
    OUTDIR="$BASEOUT/run_$RUNID"
    mkdir -p "$OUTDIR"

    echo "==== BASELINE RUN $RUNID ===="
    ./rename_exp.sh baseline "$RUNID" "$OUTDIR"

    # Baseline counters
    BELIM=$(grep -m1 "move_elimination.int_eliminated" "$OUTDIR/perf_baseline.txt" | awk '{print $1}' | tr -d ',')
    BNOT=$(grep -m1 "move_elimination.int_not_eliminated" "$OUTDIR/perf_baseline.txt" | awk '{print $1}' | tr -d ',')

    # Storage for each pressure mode
    declare -A E
    declare -A NE

    for mode in openssl add double mul ad; do

        VMODE="pressure_$mode"
        echo "==== PRESSURE MODE: $mode RUN $RUNID ===="

        ./rename_exp.sh "$VMODE" "$RUNID" "$OUTDIR"

        # Parse perf results
        E[$mode]=$(grep -m1 "move_elimination.int_eliminated" "$OUTDIR/perf_pressure.txt" | awk '{print $1}' | tr -d ',')
        NE[$mode]=$(grep -m1 "move_elimination.int_not_eliminated" "$OUTDIR/perf_pressure.txt" | awk '{print $1}' | tr -d ',')

        ###############################################
        # CRITICAL: Flush CPU state BEFORE next mode
        ###############################################
        flush_cpu_state
    done

    # Output one row with all modes
    echo "$RUNID,$BELIM,$BNOT,${E[openssl]},${NE[openssl]},${E[add]},${NE[add]},${E[double]},${NE[double]},${E[mul]},${NE[mul]},${E[ad]},${NE[ad]}" >> "$CSV"

done

###############################################
# Append column averages
###############################################
awk -F',' '
NR==1 { next }
{
    be+=$2; bn+=$3;
    oe+=$4; on+=$5;
    ae+=$6; an+=$7;
    de+=$8; dn+=$9;
    me+=$10; mn+=$11;
    ade+=$12; adn+=$13;
    n++
}
END {
    printf "AVG,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f\n",
        be/n, bn/n,
        oe/n, on/n,
        ae/n, an/n,
        de/n, dn/n,
        me/n, mn/n,
        ade/n, adn/n
}' "$CSV" >> "$CSV"

echo "=============================================="
echo " Experiment complete"
echo " CSV written to $CSV"
echo "=============================================="
