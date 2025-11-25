#!/bin/bash

# Usage:
#   ./rename_exp.sh <baseline|pressure_openssl|pressure_add|pressure_double|pressure_mul|pressure_ad> <runid> <outdir>

MODE="$1"
RUNID="$2"
OUTDIR="$3"

if [ -z "$MODE" ] || [ -z "$RUNID" ] || [ -z "$OUTDIR" ]; then
    echo "Usage: ./rename_exp.sh <baseline|pressure_openssl|pressure_add|pressure_double|pressure_mul|pressure_ad> <runid> <outdir>"
    exit 1
fi

mkdir -p "$OUTDIR"

# Victim binaries
OPENSSL=/usr/local/ssl/bin/openssl
ECC=./ecc

KEY_CURVE=brainpoolP512r1
KEY_FILE=$KEY_CURVE.pem

# Clean stale processes/fifo
pkill spy 2>/dev/null
pkill openssl 2>/dev/null
rm -f pipe.fifo

# Create key once if missing for OpenSSL signing
if [[ $MODE == pressure_openssl ]]; then
    if [ ! -f $KEY_FILE ]; then
        $OPENSSL ecparam -genkey -name $KEY_CURVE -out $KEY_FILE
        $OPENSSL ec -in $KEY_FILE -pubout >> $KEY_FILE
    fi
fi

mkfifo pipe.fifo

#############################################
# VICTIM (only in pressure modes)
#############################################

case "$MODE" in

    pressure_openssl)
        taskset -c 24 $OPENSSL dgst -sha512 -sign $KEY_FILE -out data.sig pipe.fifo &
        ;;

    pressure_add)
        taskset -c 24 $ECC A 2000 DEAD &   # iterations + nonce
        ;;

    pressure_double)
        taskset -c 24 $ECC D 2000 DEAD &
        ;;

    pressure_mul)
        taskset -c 24 $ECC M 2000 DEAD &
        ;;

    pressure_ad)
        taskset -c 24 $ECC AD 2000 DEAD &
        ;;

    baseline)
        # no victim
        ;;
esac

sleep 0.1

#############################################
# SPY (perf measurement)
#############################################

if [[ $MODE == baseline ]]; then
    PERF_OUT="$OUTDIR/perf_baseline.txt"
else
    PERF_OUT="$OUTDIR/perf_pressure.txt"
fi

taskset -c 0 perf stat \
    -o "$PERF_OUT" \
    -e move_elimination.int_eliminated,move_elimination.int_not_eliminated \
    ./spy

#############################################
# Save timing trace
#############################################

if [[ $MODE == baseline ]]; then
    cp timings.bin "$OUTDIR/timings_baseline.bin"
else
    cp timings.bin "$OUTDIR/timings_pressure_${MODE}.bin"
fi

wait

dd if=/dev/zero of=data.bin bs=1 count=1K

rm -f pipe.fifo
pkill spy
pkill openssl
