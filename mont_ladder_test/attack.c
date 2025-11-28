#include <stdint.h>
#include <x86intrin.h>

int main(void) {
    register uint64_t a = 1, b = 2, c = 3, d = 4;
    uint64_t t0, t1, dt;

    for (;;) {

        t0 = __rdtscp(&d);

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
        );

        t1 = __rdtscp(&d);
        dt = t1 - t0;

        /* store or print dt here */
        (void)dt;
    }

    return 0;
}
