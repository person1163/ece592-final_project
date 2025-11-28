#include <stdint.h>
#include <string.h>

// ------- PATH A: real-world small-copy fallback (rename-heavy) -------
__attribute__((noinline))
void small_copy_path(uint8_t *dst, const uint8_t *src)
{
    // small memcpy fallback, like glibc does for tiny sizes
    // compiler emits: mov rX -> rY, mov rZ -> rW, etc.
    for (int i = 0; i < 200000000; i++) {
        memcpy(dst, src, 16);   // yields register shuffle path
    }
}

// ------- PATH B: real-world compute-heavy loop (ALU-heavy) -------
__attribute__((noinline))
void compute_path(uint64_t *x)
{
    uint64_t v = *x;
    for (int i = 0; i < 200000000; i++) {
        v = v * 1664525 + 1013904223; // LCG
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
    }
    *x = v;
}

// ------------------ MAIN -------------------
int main(int argc, char **argv)
{
    static uint8_t src[16] = {0};
    static uint8_t dst[16];
    static uint64_t x = 1;

    if (argc > 1 && argv[1][0] == 'A') {
        small_copy_path(dst, src);   // PATH A (rename heavy)
    } else {
        compute_path(&x);            // PATH B (ALU heavy)
    }

    return 0;
}
