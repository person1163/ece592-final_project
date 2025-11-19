
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <x86intrin.h>

#include "attacker.h"  // only for SPY_PIPE and ZERO_COUNT

int main() {

    // --- 1. Wake up victim (pipe sync) ---
    uint8_t *zeroes = calloc(ZERO_COUNT, 1);
    assert(zeroes != NULL);

    FILE *pipe = fopen(SPY_PIPE, "wb+");
    assert(pipe != NULL);

    size_t ret = fwrite(zeroes, 1, ZERO_COUNT, pipe);
    assert(ret == ZERO_COUNT);

    fclose(pipe);
    free(zeroes);

    // --- 2. MOV-only attacker loop (your side channel hammer) ---
    register uint64_t a = 1, b = 2;

    for (uint64_t i = 0; i < 2000000000ULL; i++) {
        __asm__ volatile(
            "mov %1, %0\n\t"
            : "+r"(a)
            : "r"(b)
        );
    }

    return 0;
}
