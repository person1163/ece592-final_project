#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>

uint64_t rdtsc() {
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

void high_move_elimination() {
    register int a = 0, b = 1, c = 2, d = 3;
    for (uint64_t i = 0; i < 100000000; i++) {
        a = b;        // move eliminated
        b = c;        // move eliminated
        c = d;        // move eliminated
        d = a;        // move eliminated
        a = a ^ a;    // zero idiom eliminated
        b = b ^ b;
        c = c ^ c;
        d = d ^ d;
    }
    volatile int sink = a + b + c + d;
}

void low_move_elimination() {
    int a = 1, b = 2, c = 3, d = 4;
    for (uint64_t i = 0; i < 100000000; i++) {
        a = b + i;    // real dependency
        b = c + a;
        c = d + b;
        d = a + c;
    }
    volatile int sink = a + b + c + d;
}

int main() {
    uint64_t start, end;

    start = rdtsc();
    high_move_elimination();
    end = rdtsc();
    printf("High move elimination cycles: %lu\n", end - start);

    start = rdtsc();
    low_move_elimination();
    end = rdtsc();
    printf("Low move elimination cycles: %lu\n", end - start);

    return 0;
}
