#include <openssl/ec.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define REPEAT 1  /* increase to amplify the signal */

int main(int argc, char **argv) {
    if (argc != 2) return 1;
    unsigned ext_bit = atoi(argv[1]);   /* 0 or 1 */

    BN_CTX   *ctx   = BN_CTX_new();
    EC_GROUP *group = EC_GROUP_new_by_curve_name(NID_secp384r1);
    assert(ctx && group);

    EC_POINT *P  = EC_POINT_new(group);
    EC_POINT *R0 = EC_POINT_new(group);
    EC_POINT *R1 = EC_POINT_new(group);
    assert(P && R0 && R1);

    EC_POINT_copy(P, EC_GROUP_get0_generator(group));
    EC_POINT_set_to_infinity(group, R0);
    EC_POINT_copy(R1, P);

    int degree = EC_GROUP_get_degree(group);   /* ~384 for secp384r1 */
    BIGNUM *k = BN_new();
    assert(k);

    BN_zero(k);
    if (ext_bit) {
        /* scalar with only MSB set */
        BN_set_bit(k, degree - 1);
    }

    /* precompute scalar bits (MSB -> LSB) */
    unsigned char *bits = calloc(degree, 1);
    assert(bits);
    for (int i = 0; i < degree; i++) {
        bits[i] = BN_is_bit_set(k, i) ? 1 : 0;
    }

    for (int rep = 0; rep < REPEAT; rep++) {
        /* reset ladder state each repetition */
        EC_POINT_set_to_infinity(group, R0);
        EC_POINT_copy(R1, P);

        for (int i = degree - 1; i >= 0; i--) {
            unsigned b = bits[i];

            if (b == 0) {
                /* one Montgomery-style ladder step, bit = 0 */
                EC_POINT_dbl(group, R1, R1, ctx);          /* R1 = 2R1 */
                EC_POINT_add(group, R0, R0, R1, ctx);      /* R0 = R0 + R1 */
            } else {
                /* bit = 1 */
                EC_POINT_dbl(group, R0, R0, ctx);          /* R0 = 2R0 */
                EC_POINT_add(group, R1, R0, R1, ctx);      /* R1 = R0 + R1 */
            }
        }
    }

    free(bits);
    BN_free(k);
    EC_POINT_free(R0);
    EC_POINT_free(R1);
    EC_POINT_free(P);
    EC_GROUP_free(group);
    BN_CTX_free(ctx);

    return 0;
}
