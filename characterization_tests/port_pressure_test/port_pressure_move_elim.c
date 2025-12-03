// port_pressure_move_elim_fenced.c
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define ITERS 20000000UL   // adjust if needed

// ----------------------------------------
// ALU pressure + eliminatable MOV (ultra-fenced)
// ----------------------------------------
__attribute__((noinline))
void port_pressure_alu(int n_alu) {
    register uint64_t a = 1, b = 2;
    register uint64_t c = 3, d = 4;

    for (uint64_t i = 0; i < ITERS; i++) {
        // ALU ops to create port 0/1 pressure
        for (int j = 0; j < n_alu; j++) {
            asm volatile("lfence" ::: "memory");
            asm volatile(
                "add %1, %0\n\t"
                : "+r"(a)
                : "r"(b)
            );
            asm volatile("lfence" ::: "memory");

            asm volatile("lfence" ::: "memory");
            asm volatile(
                "add %1, %0\n\t"
                : "+r"(c)
                : "r"(d)
            );
            asm volatile("lfence" ::: "memory");
        }

        // eliminatable mov (same-reg renaming pattern), also fenced
        asm volatile("lfence" ::: "memory");
        asm volatile(
            "mov %1, %0\n\t"
            : "+r"(a)
            : "r"(b)
        );
        asm volatile("lfence" ::: "memory");
    }

    asm volatile("" :: "r"(a), "r"(b), "r"(c), "r"(d));
}

// ----------------------------------------
// LOAD pressure + eliminatable MOV (ultra-fenced)
// ----------------------------------------
__attribute__((noinline))
void port_pressure_load(int n_load) {
    static uint64_t buf[4096]; // small hot array
    register uint64_t a = 1, b = 2;
    uint64_t idx = 0;

    for (uint64_t i = 0; i < ITERS; i++) {
        for (int j = 0; j < n_load; j++) {
            // simple pointer chase / streaming pattern
            asm volatile("lfence" ::: "memory");
            asm volatile(
                "mov (%[base], %[off], 8), %[dst]\n\t"
                : [dst] "+r"(a)
                : [base] "r"(buf), [off] "r"(idx)
                : "memory"
            );
            asm volatile("lfence" ::: "memory");

            idx = (idx + 1) & (4096 - 1);
        }

        asm volatile("lfence" ::: "memory");
        asm volatile(
            "mov %1, %0\n\t"
            : "+r"(a)
            : "r"(b)
        );
        asm volatile("lfence" ::: "memory");
    }

    asm volatile("" :: "r"(a), "r"(b));
}

// ----------------------------------------
// STORE pressure + eliminatable MOV (ultra-fenced)
// ----------------------------------------
__attribute__((noinline))
void port_pressure_store(int n_store) {
    static uint64_t buf[4096];
    register uint64_t a = 1, b = 2;
    uint64_t idx = 0;

    for (uint64_t i = 0; i < ITERS; i++) {
        for (int j = 0; j < n_store; j++) {
            asm volatile("lfence" ::: "memory");
            asm volatile(
                "mov %[src], (%[base], %[off], 8)\n\t"
                :
                : [src] "r"(a), [base] "r"(buf), [off] "r"(idx)
                : "memory"
            );
            asm volatile("lfence" ::: "memory");

            idx = (idx + 1) & (4096 - 1);
        }

        asm volatile("lfence" ::: "memory");
        asm volatile(
            "mov %1, %0\n\t"
            : "+r"(a)
            : "r"(b)
        );
        asm volatile("lfence" ::: "memory");
    }

    asm volatile("" :: "r"(a), "r"(b));
}

// ----------------------------------------
// main: mode + N
// mode: 0 = ALU, 1 = LOAD, 2 = STORE
// ----------------------------------------
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <mode:0|1|2> <N>\n", argv[0]);
        fprintf(stderr, "  mode 0: ALU pressure\n");
        fprintf(stderr, "  mode 1: LOAD pressure\n");
        fprintf(stderr, "  mode 2: STORE pressure\n");
        return 1;
    }

    int mode = atoi(argv[1]);
    int N    = atoi(argv[2]);
    if (N < 0) N = 0;

    switch (mode) {
    case 0:
        port_pressure_alu(N);
        break;
    case 1:
        port_pressure_load(N);
        break;
    case 2:
        port_pressure_store(N);
        break;
    default:
        fprintf(stderr, "bad mode %d\n", mode);
        return 1;
    }

    return 0;
}
