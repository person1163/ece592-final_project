#!/bin/bash
set -e

make

echo "[*] Running victim(0) on CPU0 and attacker on CPU24..."

# Run attacker *under perf* on CPU24
perf stat -o perf_0.txt \
    -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
    taskset -c 24 ./attacker &
ATTACKER=$!

sleep 0.02

# Run victim normally on CPU0 (short-lived)
taskset -c 0 ./victim 0

wait $ATTACKER
mv timings.bin timings_0.bin


echo "[*] Running victim(1) on CPU0 and attacker on CPU24..."

perf stat -o perf_1.txt \
    -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
    taskset -c 24 ./attacker &
ATTACKER=$!

sleep 0.02

taskset -c 0 ./victim 1

wait $ATTACKER
mv timings.bin timings_1.bin


echo "Done."
echo "Perf attacker results in perf_0.txt and perf_1.txt."


# perf stat -e cycles,instructions,move_elimination.int_eliminated,move_elimination.int_not_eliminated \
#     taskset -c 0 ./victim 0 2> perf_0.txt
 