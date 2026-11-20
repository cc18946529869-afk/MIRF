/* Exact grid accumulations for interval certification; bounds/overflow are
   checked by the caller. This file does not evaluate or optimize a model. */
#include <stdint.h>
#include <stddef.h>

void grid_logits(const int8_t *matrix, const int64_t *lower,
                 const int64_t *upper, int n, int length, int q, int site,
                 int64_t *out_lower, int64_t *out_upper) {
    size_t extent = (size_t)(length - 1) * q * q;
    for (int row = 0; row < n; row++) {
        int64_t *lo = out_lower + (size_t)row * q;
        int64_t *hi = out_upper + (size_t)row * q;
        const int8_t *input = matrix + (size_t)row * length;
        for (int a = 0; a < q; a++) {
            lo[a] = lower[extent + a];
            hi[a] = upper[extent + a];
        }
        int partner = 0;
        for (int col = 0; col < length; col++) {
            if (col == site) continue;
            size_t offset = ((size_t)partner * q + (size_t)input[col]) * q;
            for (int a = 0; a < q; a++) {
                lo[a] += lower[offset + a];
                hi[a] += upper[offset + a];
            }
            partner++;
        }
    }
}

void grid_gradients(const int8_t *matrix, const int64_t *res_lower,
                    const int64_t *res_upper, int n, int length, int q, int site,
                    int64_t *lower, int64_t *upper) {
    size_t extent = (size_t)(length - 1) * q * q;
    for (int row = 0; row < n; row++) {
        const int8_t *input = matrix + (size_t)row * length;
        const int64_t *lo = res_lower + (size_t)row * q;
        const int64_t *hi = res_upper + (size_t)row * q;
        for (int a = 0; a < q; a++) {
            lower[extent + a] += lo[a];
            upper[extent + a] += hi[a];
        }
        int partner = 0;
        for (int col = 0; col < length; col++) {
            if (col == site) continue;
            size_t offset = ((size_t)partner * q + (size_t)input[col]) * q;
            for (int a = 0; a < q; a++) {
                lower[offset + a] += lo[a];
                upper[offset + a] += hi[a];
            }
            partner++;
        }
    }
}
