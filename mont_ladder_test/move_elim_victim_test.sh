#!/bin/bash

OUT=results.csv
echo "run,bit0_elim,bit0_notelim,bit1_elim,bit1_notelim" > $OUT

runs=10

for i in $(seq 1 $runs); do
    #
    # bit = 0
    #
    perf stat -e move_elimination.int_eliminated,move_elimination.int_not_eliminated \
        ./ecc 0 2> perf0.tmp

    b0_elim=$(grep move_elimination.int_eliminated     perf0.tmp | awk '{print $1}')
    b0_not=$(grep move_elimination.int_not_eliminated  perf0.tmp | awk '{print $1}')

    #
    # bit = 1
    #
    perf stat -e move_elimination.int_eliminated,move_elimination.int_not_eliminated \
        ./ecc 1 2> perf1.tmp

    b1_elim=$(grep move_elimination.int_eliminated     perf1.tmp | awk '{print $1}')
    b1_not=$(grep move_elimination.int_not_eliminated  perf1.tmp | awk '{print $1}')

    #
    # write row
    #
    echo "$i,$b0_elim,$b0_not,$b1_elim,$b1_not" >> $OUT
done

rm perf0.tmp perf1.tmp
