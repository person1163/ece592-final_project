#!/bin/bash

###############################################
# CONFIGURABLE TRIALS SCRIPT
#
# Usage examples:
#   ./run_experiment.sh --mode openssl --runs 5000
#   ./run_experiment.sh --mode add --runs 2000
#   ./run_experiment.sh --mode all --runs 3000
###############################################

# Default parameters
MODE="openssl"
RUNS=1000
CURVE="brainpoolP512r1"

###############################################
# Parse command-line arguments
###############################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --runs) RUNS="$2"; shift 2 ;;
        --curve) CURVE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

###############################################
# Output directory
###############################################
BASEOUT="results_${CURVE}_${MODE}_${RUNS}"
mkdir -p "$BASEOUT"

CSV="$BASEOUT/results_summary.csv"
echo "run,baseline_elim,baseline_notelim,pressure_elim,pressure_notelim,mode" > "$CSV"

###############################################
# Rebuild once
###############################################
make clean
make

echo "=============================================="
echo " Running experiment:"
echo "   Mode:  $MODE"
echo "   Curve: $CURVE"
echo "   Runs:  $RUNS"
echo "   Output: $BASEOUT"
echo "=============================================="

###############################################
# Determine mode for rename_exp.sh
###############################################
case "$MODE" in
    openssl)   VMODE="pressure_openssl" ;;
    add)       VMODE="pressure_add" ;;
    double)    VMODE="pressure_double" ;;
    mul)       VMODE="pressure_mul" ;;
    ad)        VMODE="pressure_ad" ;;
    all)       VMODE="all" ;;
    *) echo "Invalid mode: $MODE"; exit 1 ;;
esac

###############################################
# Microarchitectural flush helpers
###############################################
flush_cpu_state() {
    # Drop page cache: L1/L2/L3 impacts reduced
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    # Branch predictor & pipeline "scrubbing"
    for j in {1..200000}; do :; done

    # DVFS / pipeline cool-down
    sleep 0.05
}

###############################################
# Main loop
###############################################
for i in $(seq 1 $RUNS); do
    RUNID=$(printf "%04d" $i)
    OUTDIR="$BASEOUT/run_$RUNID"
    mkdir -p "$OUTDIR"

    ###############################################
    # BASELINE
    ###############################################
    echo "==== BASELINE RUN $RUNID ===="
    ./rename_exp.sh baseline "$RUNID" "$OUTDIR"

    BELIM=$(grep -m1 "move_elimination.int_eliminated"     "$OUTDIR/perf_baseline.txt" | awk '{print $1}' | tr -d ',')
    BNOT=$(grep -m1 "move_elimination.int_not_eliminated"  "$OUTDIR/perf_baseline.txt" | awk '{print $1}' | tr -d ',')

    ###############################################
    # MODE = ALL: run every pressure operation
    ###############################################
    if [[ "$MODE" == "all" ]]; then
        for M in pressure_openssl pressure_add pressure_double pressure_mul pressure_ad; do

            echo "==== MODE $M RUN $RUNID ===="

            ./rename_exp.sh "$M" "$RUNID" "$OUTDIR"

            PELIM=$(grep -m1 "move_elimination.int_eliminated"     "$OUTDIR/perf_pressure.txt" | awk '{print $1}' | tr -d ',')
            PNOT=$(grep -m1 "move_elimination.int_not_eliminated"  "$OUTDIR/perf_pressure.txt" | awk '{print $1}' | tr -d ',')

            echo "$RUNID,$BELIM,$BNOT,$PELIM,$PNOT,$M" >> "$CSV"

            # *** IMPORTANT ***
            # Flush CPU state BETWEEN different victim types
            flush_cpu_state

        done
        continue
    fi

    ###############################################
    # SINGLE OPERATION MODE
    ###############################################
    echo "==== PRESSURE RUN $RUNID ===="
    ./rename_exp.sh "$VMODE" "$RUNID" "$OUTDIR"

    PELIM=$(grep -m1 "move_elimination.int_eliminated"     "$OUTDIR/perf_pressure.txt" | awk '{print $1}' | tr -d ',')
    PNOT=$(grep -m1 "move_elimination.int_not_eliminated"  "$OUTDIR/perf_pressure.txt" | awk '{print $1}' | tr -d ',')

    echo "$RUNID,$BELIM,$BNOT,$PELIM,$PNOT,$MODE" >> "$CSV"

done

###############################################
# Add averages
###############################################
awk -F',' '
NR==1 { next } # skip header
{
    be+=$2; bn+=$3; pe+=$4; pn+=$5; n++
}
END {
    printf "AVG,%.0f,%.0f,%.0f,%.0f,---\n", be/n, bn/n, pe/n, pn/n
}' "$CSV" >> "$CSV"

echo "=============================================="
echo " Experiment complete"
echo " CSV written to $CSV"
echo "=============================================="
