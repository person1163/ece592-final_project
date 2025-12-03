#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define ITERS 200000000UL

// Generate register pressure by touching many registers
#define TOUCH(r) asm volatile("add %0,%0" : "+r"(r))

__attribute__((noinline))
void reg_pressure_test(int P) {

    // Highly live registers
    register uint64_t r8v=1, r9v=2, r10v=3, r11v=4;
    register uint64_t r12v=5, r13v=6, r14v=7, r15v=8;

    // The MOV under test (should eliminate)
    register uint64_t a = 1, b = 2;

    for (uint64_t i = 0; i < ITERS; i++) {

        asm volatile("lfence" ::: "memory");

        // --- register pressure block ---
        // P can be 0–8
        switch (P) {
            case 8:
                TOUCH(r15v);
            case 7:
                TOUCH(r14v);
            case 6:
                TOUCH(r13v);
            case 5:
                TOUCH(r12v);
            case 4:
                TOUCH(r11v);
            case 3:
                TOUCH(r10v);
            case 2:
                TOUCH(r9v);
            case 1:
                TOUCH(r8v);
            default:
                break;
        }

        asm volatile("lfence" ::: "memory");

        // --- The move under test ---
        asm volatile("mov %1, %0" : "+r"(a) : "r"(b) : "memory");

        asm volatile("lfence" ::: "memory");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <P_registers_to_pressure>\n", argv[0]);
        return 1;
    }
    int P = atoi(argv[1]);
    reg_pressure_test(P);
    return 0;
}
