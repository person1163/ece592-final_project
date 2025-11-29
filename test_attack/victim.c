#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "attacker.h"   // SPY_PIPE, ZERO_COUNT

// #define SMALL_ITERS 2000000000L
// #define LARGE_ITERS  200000000L
#define SMALL_ITERS 200000L        // 200k
#define LARGE_ITERS  20000L        // 20k

// ------------------------------------------------------------
// rename-heavy path: forces move-elimination / renamer pressure
// ------------------------------------------------------------
__attribute__((noinline))
static void rename_heavy_path(void) {
    register uint64_t a = 1, b = 2, c = 3, d = 4;

    for (long i = 0; i < SMALL_ITERS; i++) {
        __asm__ volatile(
            ".rept 64\n\t"
            "mov %0, %1\n\t"
            "mov %1, %2\n\t"
            "mov %2, %3\n\t"
            "mov %3, %0\n\t"
            "xor %0, %0\n\t"
            "xor %1, %1\n\t"
            "xor %2, %2\n\t"
            "xor %3, %3\n\t"
            ".endr\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
            :
            : "memory"
        );
    }
}

// ------------------------------------------------------------
// rename-light path: large memcpy loop (LSU/bandwidth bound)
// ------------------------------------------------------------
__attribute__((noinline))
static void bulk_memcpy_path(void) {
    static uint8_t src[4096] = {0};
    static uint8_t dst[4096];

    for (long i = 0; i < LARGE_ITERS; i++) {
        memcpy(dst, src, sizeof(src));   // minimal rename, heavy load/store
    }
}

// ------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <0|1>\n", argv[0]);
        return 1;
    }

    int mode = atoi(argv[1]);   // 0 = memcpy-path, 1 = rename-heavy

    // ---- wait for attacker (synchronization) ----
    uint8_t *zeroes = calloc(ZERO_COUNT, 1);
    assert(zeroes != NULL);

    FILE *pipe = fopen(SPY_PIPE, "rb");
    assert(pipe != NULL);

    size_t ret = fread(zeroes, 1, ZERO_COUNT, pipe);
    assert(ret == ZERO_COUNT);

    fclose(pipe);
    free(zeroes);

    // ---- choose path ----
    if (mode == 1) {
        rename_heavy_path();
    } else {
        bulk_memcpy_path();
    }

    return 0;
}
