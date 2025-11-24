// attacker.h – shared constants for attacker and ECC victim
#ifndef ATTACKER_H
#define ATTACKER_H

// Named pipe used to synchronize victim and attacker
#define SPY_PIPE "pipe.fifo"

// Number of bytes transferred for the sync
#define ZERO_COUNT 1024   // matches original (1<<10)

#endif
