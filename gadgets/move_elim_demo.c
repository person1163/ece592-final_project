#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ITERS 200000000UL   // increase if needed

// -------------------------------------------------------------------
// Gadget A: eliminatable MOV (mov b → a)
// Renamer optimizes it away → no backend uops
// -------------------------------------------------------------------
__attribute__((noinline))
void gadget_elim(void) {
    register uint64_t a = 1;
    register uint64_t b = 2;

    for (uint64_t i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mov %1, %0\n\t"      // eliminatable
            : "+r"(a), "+r"(b)
            :
            : "memory"
        );
    }

    asm volatile("" :: "r"(a), "r"(b));
}

// -------------------------------------------------------------------
// Gadget B: non-eliminatable MOV (mov a → a)
// Self-move is architecturally required → must execute
// -------------------------------------------------------------------
__attribute__((noinline))
void gadget_no_elim(void) {
    register uint64_t a = 1;

    for (uint64_t i = 0; i < ITERS; i++) {
        __asm__ volatile(
            "mov %0, %0\n\t"      // forced non-eliminatable
            : "+r"(a)
            :
            : "memory"
        );
    }

    asm volatile("" :: "r"(a));
}

// -------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <0|1>\n", argv[0]);
        printf("  0 = eliminatable MOV\n");
        printf("  1 = non-eliminatable MOV\n");
        return 1;
    }

    int mode = atoi(argv[1]);

    if (mode == 0) {
        gadget_elim();
    } else {
        gadget_no_elim();
    }

    return 0;
}
