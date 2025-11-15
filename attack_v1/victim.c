// victim.c
#include <stdint.h>

int main() {
    volatile int x = 1;
    volatile int y = 2;
    volatile int z = 3;

    for (uint64_t i = 0; i < 2000000000ULL; i++) {

        // true dependencies: each depends on the previous result
        x = x + y;
        y = y + z;
        z = z + x;
    }

    return x + y + z;
}
