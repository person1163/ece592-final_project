// attacker.c
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
