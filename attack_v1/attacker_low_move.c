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
