#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ITERS 5000000L

// ------------------------------------------------------------
// MANY register-to-register MOVs, all trivially eliminatable
// ------------------------------------------------------------
__attribute__((noinline))
void move_elim_candidate(void) {
    register uint64_t a = 1, b = 2, c = 3, d = 4;

    for (long i = 0; i < ITERS; i++) {
        __asm__ volatile(
            ".rept 64\n\t"
            "mov %0, %1\n\t"
            "mov %1, %2\n\t"
            "mov %2, %3\n\t"
            "mov %3, %0\n\t"
            ".endr\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
        );
    }
}

// ------------------------------------------------------------
// FORCED non-eliminatable MOVs (dependency chain & overlap)
// ------------------------------------------------------------
__attribute__((noinline))
void move_non_elim(void) {
    register uint64_t a = 1;

    for (long i = 0; i < ITERS; i++) {
        __asm__ volatile(
            ".rept 64\n\t"
            // forced dependency chain: each mov writes a different reg
            "mov %0, %0\n\t"
            "mov %0, %0\n\t"
            "mov %0, %0\n\t"
            "mov %0, %0\n\t"
            ".endr\n\t"
            : "+r"(a)
        );
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <0|1>\n", argv[0]);
        return 1;
    }

    int mode = atoi(argv[1]);

    if (mode == 0)
        move_elim_candidate();
    else
        move_non_elim();

    return 0;
}
