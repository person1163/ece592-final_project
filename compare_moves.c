#include <stdio.h>
#include <stdint.h>
#include <string.h>

uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

void high_move_elimination() {
    register int a = 0, b = 1, c = 2, d = 3;
    for (uint64_t i = 0; i < 100000000; i++) {
        a = b; b = c; c = d; d = a;
        a = a ^ a; b = b ^ b; c = c ^ c; d = d ^ d;
    }
    volatile int sink = a + b + c + d;
}

void low_move_elimination() {
    int a = 1, b = 2, c = 3, d = 4;
    for (uint64_t i = 0; i < 100000000; i++) {
        a = b + i;
        b = c + a;
        c = d + b;
        d = a + c;
    }
    volatile int sink = a + b + c + d;
}

int main(int argc, char **argv) {
    uint64_t start, end;
    if (argc < 2) {
        printf("Usage: %s [high|low]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "high") == 0) {
        start = rdtsc();
        high_move_elimination();
        end = rdtsc();
        printf("High move elimination cycles: %lu\n", end - start);
    } else if (strcmp(argv[1], "low") == 0) {
        start = rdtsc();
        low_move_elimination();
        end = rdtsc();
        printf("Low move elimination cycles: %lu\n", end - start);
    } else {
        printf("Invalid argument: use 'high' or 'low'\n");
        return 1;
    }
    return 0;
}
