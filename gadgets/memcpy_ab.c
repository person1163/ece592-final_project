#include <stdint.h>
#include <string.h>

__attribute__((noinline))
void small_memcpy_path(uint8_t *dst, const uint8_t *src)
{
    // This triggers the glibc small-copy fallback, which is register-heavy.
    for (long i = 0; i < 2000000000L; i++) {
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
    }
}

__attribute__((noinline))
void large_memcpy_path(uint8_t *dst, const uint8_t *src)
{
    // This triggers the bulk-copy path, which is load/store + SIMD heavy.
    // Very little move elimination occurs.
    for (long i = 0; i < 200000000L; i++) {
        memcpy(dst, src, 256);
    }
}

int main(int argc, char **argv)
{
    static uint8_t src[256] = {0};
    static uint8_t dst[256] = {0};

    if (argc > 1 && argv[1][0] == 'S') {
        small_memcpy_path(dst, src);     // PATH A: rename-heavy
    } else {
        large_memcpy_path(dst, src);     // PATH B: rename-light
    }

    return 0;
}
