// attacker.c high moves
#include <stdint.h>

int main() {
    volatile int a = 1, b = 2, c = 3, d = 4;

    for (;;) {
        a = b;
        b = c;
        c = d;
        d = a;

        a = a ^ a;
        b = b ^ b;
        c = c ^ c;
        d = d ^ d;
    }

    return 0;
}

// #include <stdint.h>

// int main(void) {
//     // Keep values in registers as much as possible
//     register int a = 1;
//     register int b = 2;
//     register int c = 3;
//     register int d = 4;

//     for (;;) {
//         __asm__ volatile(
//             // Chain of register-to-register moves (move-elimination candidates)
//             "mov %1, %0\n\t"
//             "mov %2, %1\n\t"
//             "mov %3, %2\n\t"
//             "mov %0, %3\n\t"

//             // Zero-idioms (XOR reg, reg) — also handled by renamer
//             "xor %0, %0\n\t"
//             "xor %1, %1\n\t"
//             "xor %2, %2\n\t"
//             "xor %3, %3\n\t"
//             : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
//         );
//     }

//     return 0;
// }

