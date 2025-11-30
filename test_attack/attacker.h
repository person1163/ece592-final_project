#ifndef ATTACKER_H
#define ATTACKER_H

#define SPY_NUM_TIMINGS (1<<16)
#define ZERO_COUNT      (1<<10)
#define SPY_PIPE        "pipe.fifo"

#ifndef __ASSEMBLER__
#include <stdint.h>
void rensmash(uint64_t *buffer);
#endif

#endif