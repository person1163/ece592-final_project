#include <stdint.h>
#include <string.h>

#define SMALL_ITERS 2000000000L
#define LARGE_ITERS   200000000L

__attribute__((noinline))
void small_path()
{
    uint8_t src[16] = {0};
    uint8_t dst[16];

    for (long i = 0; i < SMALL_ITERS; i++) {
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
    }
}

__attribute__((noinline))
void large_path()
{
    // Make them static to ensure they live in .bss and are aligned
    static uint8_t src[4096];
    static uint8_t dst[4096];

    // Always copy within the 4096-byte buffer
    for (long i = 0; i < LARGE_ITERS; i++) {
        memcpy(dst, src, 256);
    }
}

int main(int argc, char **argv)
{
    if (argc > 1 && argv[1][0] == 'S')
        small_path();    // rename-heavy glibc small memcpy
    else
        large_path();    // rename-light glibc bulk memcpy

    return 0;
}
