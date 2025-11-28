#include <openssl/obj_mac.h>
#include <openssl/ec.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "spy.h"   /* needed for ZERO_COUNT and SPY_PIPE */

int main(int argc, char **argv) {
    unsigned bit = atoi(argv[1]);     /* 0 or 1 */

    /* ---------- PortSmash sync (required) ---------- */
    size_t ret;
    uint8_t *zeroes = (uint8_t *)calloc(ZERO_COUNT, sizeof(uint8_t));
    assert(zeroes != NULL);
    FILE *pipe;
    pipe = fopen(SPY_PIPE, "rb");
    assert(pipe != NULL);
    ret = fread(zeroes, sizeof(uint8_t), ZERO_COUNT, pipe);
    assert(ret == ZERO_COUNT);
    fclose(pipe);
    free(zeroes);
    /* ------------------------------------------------ */

    BN_CTX *ctx = BN_CTX_new();
    EC_GROUP *group = EC_GROUP_new_by_curve_name(NID_secp384r1);

    EC_POINT *P  = EC_POINT_new(group);
    EC_POINT *R0 = EC_POINT_new(group);
    EC_POINT *R1 = EC_POINT_new(group);

    EC_POINT_copy(P,  EC_GROUP_get0_generator(group));
    EC_POINT_set_to_infinity(group, R0);
    EC_POINT_copy(R1, P);

    if (bit == 0) {
        EC_POINT_dbl(group, R0, R0, ctx);      /* DOUBLE */
        EC_POINT_add(group, R1, R0, R1, ctx);  /* ADD    */
    } else {
        EC_POINT_add(group, R0, R0, R1, ctx);  /* ADD    */
        EC_POINT_dbl(group, R1, R1, ctx);      /* DOUBLE */
    }

    return 0;
}
