#include <string.h>
#include <stdint.h>

int main() {
    uint8_t a[16], b[16];
    for (;;) {
        memcmp(a, b, 16);
        memcmp(a, b, 16);
        memcmp(a, b, 16);
        memcmp(a, b, 16);
    }
}
