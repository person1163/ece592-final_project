#include <string.h>
#include <stdint.h>

int main() {
    static uint8_t src[16], dst[16];
    volatile uint8_t *ps = src;
    volatile uint8_t *pd = dst;

    for (long i = 0; i < 2000000000L; i++) {
        memcpy((void*)pd, (void*)ps, 16);
        memcpy((void*)pd, (void*)ps, 16);
        memcpy((void*)pd, (void*)ps, 16);
        memcpy((void*)pd, (void*)ps, 16);
    }

    return 0;
}
