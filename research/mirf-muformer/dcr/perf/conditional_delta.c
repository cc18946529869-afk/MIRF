#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Same row-order line-search arithmetic as conditional.c; the cache
   already describes the base point, so coupling layout is irrelevant. */

/* ΔF_NLL = Σ_b w_b [log Σ_a p_ba exp(α v_ba) - α v_b,y_b], summed in row order. */

int site_delta_nll(
    const double *probability,
    const double *direction_logit,
    const int8_t *observed,
    const double *weights,
    double alpha,
    int B,
    int q,
    double *delta_nll
) {
    double sum = 0.0;
    double compensation = 0.0;
    for (int row = 0; row < B; row++) {
        const double *prob = probability + (size_t)row * (size_t)q;
        const double *velocity = direction_logit + (size_t)row * (size_t)q;
        const int state = (int)observed[row];
        if (state < 0 || state >= q) {
            return -1;
        }
        double max_shift = alpha * velocity[0];
        for (int a = 1; a < q; a++) {
            const double shift = alpha * velocity[a];
            if (shift > max_shift) {
                max_shift = shift;
            }
        }
        if (!isfinite(max_shift)) {
            return -1;
        }
        double mass = 0.0;
        for (int a = 0; a < q; a++) {
            mass += prob[a] * exp(alpha * velocity[a] - max_shift);
        }
        if (!isfinite(mass) || mass <= 0.0) {
            return -1;
        }
        /* Cancel the two large shifts before adding the small logarithm. */
        const double observed_shift = alpha * velocity[state];
        const double centered_shift = max_shift - observed_shift;
        const double term = weights[row] * (centered_shift + log(mass));
        if (!isfinite(term)) {
            return -1;
        }
        const double updated = sum + term;
        if (fabs(sum) >= fabs(term)) {
            compensation += (sum - updated) + term;
        } else {
            compensation += (term - updated) + sum;
        }
        sum = updated;
    }
    *delta_nll = sum + compensation;
    return 0;
}
