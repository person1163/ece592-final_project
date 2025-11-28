#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define SMALL_ITERS 2000000000L  // adjust if runtime too long
#define LARGE_ITERS  200000000L

__attribute__((noinline))
void small_match_path(void) {
    // Rename-heavy: many small copies (like tiny match copy)
    static uint8_t src[16] = {0};
    static uint8_t dst[16];

    for (long i = 0; i < SMALL_ITERS; i++) {
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
        memcpy(dst, src, 16);
    }
}

__attribute__((noinline))
void large_match_path(void) {
    // Rename-light: bulk copies (like wildCopy / large match)
    static uint8_t src[256] = {0};
    static uint8_t dst[256];

    for (long i = 0; i < LARGE_ITERS; i++) {
        memcpy(dst, src, 256);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <compressed_file>\n", argv[0]);
        return 1;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int c = fgetc(f);
    fclose(f);
    if (c == EOF) {
        fprintf(stderr, "empty file\n");
        return 1;
    }

    uint8_t token = (uint8_t)c;

    // "Compressed format":
    //   token LSB = 1 → model small match path (rename-heavy)
    //   token LSB = 0 → model large match path (rename-light)
    if (token & 1) {
        small_match_path();
    } else {
        large_match_path();
    }

    return 0;
}

