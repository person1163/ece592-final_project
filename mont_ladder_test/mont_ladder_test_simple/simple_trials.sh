gcc -O3 -march=native -o spy spy.c
taskset -c 24 ./spy &
SPYPID=$!

for i in $(seq 1 50000); do
    taskset -c 0 ./victim 0
done

kill $SPYPID
