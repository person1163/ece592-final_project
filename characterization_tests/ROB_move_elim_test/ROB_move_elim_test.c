#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define ITERS 200000000UL

__attribute__((noinline))
void rob_pressure_test(int M) {
    register uint64_t a = 1;
    register uint64_t b = 2;

    for (uint64_t i = 0; i < ITERS; i++) {

        asm volatile("lfence" ::: "memory");

        for (int k = 0; k < M; k++) {
            asm volatile("add %0, %0" : "+r"(b));
        }

        asm volatile("lfence" ::: "memory");

        asm volatile("mov %1, %0" : "+r"(a) : "r"(b) : "memory");

        asm volatile("lfence" ::: "memory");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <M>\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    rob_pressure_test(M);
    return 0;
}
