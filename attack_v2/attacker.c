#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <x86intrin.h>

#include "attacker.h"

int main() {
    // (1) Wake victim via pipe
    uint8_t *zeroes = calloc(ZERO_COUNT, 1);
    assert(zeroes);

    FILE *pipe = fopen(SPY_PIPE, "wb");
    assert(pipe);

    size_t written = fwrite(zeroes, 1, ZERO_COUNT, pipe);
    assert(written == ZERO_COUNT);

    fclose(pipe);
    free(zeroes);

    // (2) MOV-only attacker loop
    register uint64_t a = 1, b = 2;

    for (uint64_t i = 0; i < 2000000000ULL; i++) {
        __asm__ volatile(
            "mov %1, %0"
            : "+r"(a)
            : "r"(b)
        );
    }

    return 0;
}
