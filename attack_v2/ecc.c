#include <openssl/obj_mac.h>
#include <openssl/ec.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "attacker.h"   // for SPY_PIPE, ZERO_COUNT

int main(int argc, char * argv[]) {

    if (argc < 4 || (strcmp(argv[1], "A") && strcmp(argv[1], "D") &&
                     strcmp(argv[1], "AD") && strcmp(argv[1], "M"))) {
        printf("usage: %s <A|D|M> <iterations> <nonce> /* (A)dd (D)ouble (M)ultiply */\n",
               argv[0]);
        return 0;
    }

    int its = atoi(argv[2]);  // number of iterations

    BN_CTX *ctx = BN_CTX_new();
    assert(ctx != NULL);
    BN_CTX_start(ctx);

    EC_GROUP *group = EC_GROUP_new_by_curve_name(NID_secp384r1);
    assert(group != NULL);

    BIGNUM *k, *order, *x, *y;
    k     = BN_CTX_get(ctx);
    order = BN_CTX_get(ctx);
    x     = BN_CTX_get(ctx);
    y     = BN_CTX_get(ctx);
    assert(y != NULL);

    EC_POINT *Q = EC_POINT_new(group);
    EC_POINT *P = EC_POINT_new(group);
    assert(Q != NULL);
    assert(P != NULL);

    EC_GROUP_get_order(group, order, ctx);

    // scalar k from hex argument
    BN_hex2bn(&k, argv[3]);
    if (BN_is_zero(k))
        BN_rand_range(k, order);

    // P = k*G, Q = P
    EC_POINT_mul(group, P, k, NULL, NULL, ctx);
    EC_POINT_make_affine(group, P, ctx);
    EC_POINT_copy(Q, P);

    // ---- pipe sync: wait for attacker to write ----
    size_t  ret;
    uint8_t *zeroes = (uint8_t *)calloc(ZERO_COUNT, sizeof(uint8_t));
    assert(zeroes != NULL);

    FILE *pipe = fopen(SPY_PIPE, "rb");
    assert(pipe != NULL);

    ret = fread(zeroes, sizeof(uint8_t), ZERO_COUNT, pipe);
    assert(ret == ZERO_COUNT);

    fclose(pipe);
    free(zeroes);

    // ---- ECC work (this is the "secret-dependent" victim) ----
    if (!strcmp(argv[1], "AD")) {
        for (; its; its--) {
            EC_POINT_add(group, Q, Q, P, ctx);
        }
        for (; its; its--) {
            EC_POINT_dbl(group, Q, Q, ctx);
        }
    } else if (!strcmp(argv[1], "A")) {
        for (; its; its--) {
            EC_POINT_add(group, Q, Q, P, ctx);
        }
    } else if (!strcmp(argv[1], "D")) {
        for (; its; its--) {
            EC_POINT_dbl(group, Q, Q, ctx);
        }
    } else if (!strcmp(argv[1], "M")) {
        for (; its; its--) {
            EC_POINT_mul(group, Q, k, NULL, NULL, ctx);
        }
    } else {
        assert(0);
    }

    EC_POINT_get_affine_coordinates_GFp(group, Q, x, y, ctx);
    char *s0 = BN_bn2hex(k);
    char *s1 = BN_bn2hex(x);
    char *s2 = BN_bn2hex(y);

    printf(" k: 0x%s\n", s0);
    printf("Px: 0x%s\n", s1);
    printf("Py: 0x%s\n", s2);

    BN_CTX_end(ctx);
    EC_POINT_free(P);
    EC_POINT_free(Q);
    EC_GROUP_free(group);
    BN_CTX_free(ctx);

    free(s0);
    free(s1);
    free(s2);

    return 0;
}
