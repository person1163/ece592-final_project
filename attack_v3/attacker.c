#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <x86intrin.h>

#define SPY_NUM_TIMINGS (1 << 16)

static inline uint64_t rdtsc64(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

static void rensmash(uint64_t *buf)
{
    for (int i = 0; i < SPY_NUM_TIMINGS; i++) {
        uint64_t a = (uint64_t)(uintptr_t)buf;
        uint64_t b = a + 1;
        uint64_t c = a + 2;
        uint64_t d = a + 3;

        uint64_t start = rdtsc64();

        for (int j = 0; j < 256; j++) {
            uint64_t t = a;
            a = b;
            b = c;
            c = d;
            d = t;
        }

        uint64_t end = rdtsc64();

        uint64_t packed =
            ((end  & 0xffffffffULL) << 32) |
             (start & 0xffffffffULL);

        buf[i] = packed;
    }
}

int main(void)
{
    uint64_t *buf = malloc(SPY_NUM_TIMINGS * sizeof(uint64_t));
    assert(buf != NULL);

    rensmash(buf);

    FILE *fp = fopen("timings.bin", "wb");
    assert(fp != NULL);

    size_t n = fwrite(buf, sizeof(uint64_t), SPY_NUM_TIMINGS, fp);
    assert(n == SPY_NUM_TIMINGS);

    fclose(fp);
    free(buf);
    return 0;
}
