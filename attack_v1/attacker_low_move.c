// attacker_low_move.c
#include <stdint.h>

int main() {
    volatile int a = 1, b = 2, c = 3, d = 4;

    for (;;) {      // infinite loop
        a = a + b;
        b = b + c;
        c = c + d;
        d = d + a;
    }

    return 0;
}

//
// #include <stdint.h>

// int main(void) {
//     register int a = 1;
//     register int b = 2;
//     register int c = 3;
//     register int d = 4;

//     for (;;) {
//         __asm__ volatile(
//             // True data dependencies with ADDs (backend ALU pressure)
//             "add %1, %0\n\t"
//             "add %2, %1\n\t"
//             "add %3, %2\n\t"
//             "add %0, %3\n\t"
//             : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
//         );
//     }

//     return 0;
// }
