#include <stdint.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define SPY_NUM_TIMINGS (1 << 16)

static inline uint64_t rdtsc64(void) {
    unsigned aux;
    return __rdtscp(&aux);
}

static void rensmash(uint64_t *buf)
{
    for (int i = 0; i < SPY_NUM_TIMINGS; i++) {

        uint64_t start = rdtsc64();

        // Hammer rename logic HARD
     __asm__ volatile(
    ".rept 512\n\t"
    "mov %%r8, %%r9\n\t"
    "mov %%r9, %%r10\n\t"
    "mov %%r10, %%r11\n\t"
    "mov %%r11, %%r8\n\t"
    ".endr\n\t"
    :
    :
    : "r8","r9","r10","r11","memory"
);

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

    fwrite(buf, sizeof(uint64_t), SPY_NUM_TIMINGS, fp);

    fclose(fp);
    free(buf);
    return 0;
}
