#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* One conditional Potts site.

   theta is h (q) followed by J in C order (predicted state, partner, partner state).
   The penalty is weight_total * (lambda_h * ||h||^2 + lambda_j_prime * ||J||^2).
   Its gradient contributes 2 * weight_total * lambda * parameter.
*/

int site_objective(
    const int8_t *msa,
    const double *weights,
    const double *theta,
    int B,
    int L,
    int q,
    int site,
    double weight_total,
    double lambda_h,
    double lambda_j_prime,
    double *nll_out,
    double *penalty_out,
    double *grad
) {
    const int K = L - 1;
    const int j_size = q * K * q;
    memset(grad, 0, sizeof(double) * (size_t)(q + j_size));
    const double *h = theta;
    const double *J = theta + q;
    double nll = 0.0;
    int failed = 0;

#pragma omp parallel reduction(+ : nll)
    {
        double *gh = (double *)calloc((size_t)q, sizeof(double));
        double *gJ = (double *)calloc((size_t)j_size, sizeof(double));
        double *logits = (double *)malloc(sizeof(double) * (size_t)q);
        double *expa = (double *)malloc(sizeof(double) * (size_t)q);
        if (gh == NULL || gJ == NULL || logits == NULL || expa == NULL) {
#pragma omp atomic write
            failed = 1;
        }
#pragma omp for schedule(static)
        for (int row = 0; row < B; row++) {
            if (failed) {
                continue;
            }
            const int8_t *sequence = msa + (size_t)row * (size_t)L;
            for (int a = 0; a < q; a++) {
                logits[a] = h[a];
            }
            int partner = 0;
            for (int column = 0; column < L; column++) {
                if (column == site) {
                    continue;
                }
                const int state = (int)sequence[column];
                for (int a = 0; a < q; a++) {
                    logits[a] += J[((size_t)a * (size_t)K + (size_t)partner) * (size_t)q + (size_t)state];
                }
                partner++;
            }
            double max_logit = logits[0];
            for (int a = 1; a < q; a++) {
                if (logits[a] > max_logit) {
                    max_logit = logits[a];
                }
            }
            double sum_exp = 0.0;
            for (int a = 0; a < q; a++) {
                expa[a] = exp(logits[a] - max_logit);
                sum_exp += expa[a];
            }
            const int observed = (int)sequence[site];
            const double log_z = max_logit + log(sum_exp);
            const double weight = weights[row];
            nll -= weight * (logits[observed] - log_z);
            for (int a = 0; a < q; a++) {
                const double probability = expa[a] / sum_exp;
                double signed_weight = weight * probability;
                if (a == observed) {
                    signed_weight -= weight;
                }
                gh[a] += signed_weight;
                expa[a] = signed_weight;
            }
            partner = 0;
            for (int column = 0; column < L; column++) {
                if (column == site) {
                    continue;
                }
                const int state = (int)sequence[column];
                for (int a = 0; a < q; a++) {
                    gJ[((size_t)a * (size_t)K + (size_t)partner) * (size_t)q + (size_t)state] += expa[a];
                }
                partner++;
            }
        }
#pragma omp critical
        {
            if (gh != NULL && gJ != NULL) {
                for (int a = 0; a < q; a++) {
                    grad[a] += gh[a];
                }
                for (int index = 0; index < j_size; index++) {
                    grad[q + index] += gJ[index];
                }
            }
        }
        free(gh);
        free(gJ);
        free(logits);
        free(expa);
    }
    if (failed) {
        return -1;
    }

    double sum_h = 0.0;
    for (int a = 0; a < q; a++) {
        sum_h += h[a] * h[a];
    }
    double sum_j = 0.0;
    for (int index = 0; index < j_size; index++) {
        sum_j += J[index] * J[index];
    }
    *nll_out = nll;
    *penalty_out = weight_total * (lambda_h * sum_h + lambda_j_prime * sum_j);
    const double scale_h = 2.0 * weight_total * lambda_h;
    const double scale_j = 2.0 * weight_total * lambda_j_prime;
    for (int a = 0; a < q; a++) {
        grad[a] += scale_h * h[a];
    }
    for (int index = 0; index < j_size; index++) {
        grad[q + index] += scale_j * J[index];
    }
    return 0;
}

/* Logit range and softmax diagnostics. The conditional probabilities use the
   same stable log-sum-exp shift as the objective. Minima and maxima do not
   depend on summation order. */

int site_logit_stats(
    const int8_t *msa,
    const double *theta,
    int B,
    int L,
    int q,
    int site,
    double *logit_min,
    double *logit_max,
    double *max_row_range,
    double *max_prob_sum_error,
    double *min_probability,
    int *nonfinite_rows
) {
    const int K = L - 1;
    const double *h = theta;
    const double *J = theta + q;
    *logit_min = INFINITY;
    *logit_max = -INFINITY;
    *max_row_range = 0.0;
    *max_prob_sum_error = 0.0;
    *min_probability = 1.0;
    *nonfinite_rows = 0;

#pragma omp parallel
    {
        double local_min = INFINITY;
        double local_max = -INFINITY;
        double local_range = 0.0;
        double local_error = 0.0;
        double local_min_p = 1.0;
        int local_bad = 0;
        double *logits = (double *)malloc(sizeof(double) * (size_t)q);
        double *expa = (double *)malloc(sizeof(double) * (size_t)q);
#pragma omp for schedule(static)
        for (int row = 0; row < B; row++) {
            if (logits == NULL || expa == NULL) {
                local_bad += 1;
                continue;
            }
            const int8_t *sequence = msa + (size_t)row * (size_t)L;
            int finite = 1;
            for (int a = 0; a < q; a++) {
                logits[a] = h[a];
            }
            int partner = 0;
            for (int column = 0; column < L; column++) {
                if (column == site) {
                    continue;
                }
                const int state = (int)sequence[column];
                for (int a = 0; a < q; a++) {
                    logits[a] += J[((size_t)a * (size_t)K + (size_t)partner) * (size_t)q + (size_t)state];
                }
                partner++;
            }
            double row_min = logits[0];
            double row_max = logits[0];
            for (int a = 1; a < q; a++) {
                if (!isfinite(logits[a])) {
                    finite = 0;
                }
                if (logits[a] < row_min) {
                    row_min = logits[a];
                }
                if (logits[a] > row_max) {
                    row_max = logits[a];
                }
            }
            if (!finite || !isfinite(row_min) || !isfinite(row_max)) {
                local_bad += 1;
                continue;
            }
            if (row_min < local_min) {
                local_min = row_min;
            }
            if (row_max > local_max) {
                local_max = row_max;
            }
            const double range = row_max - row_min;
            if (range > local_range) {
                local_range = range;
            }
            double sum_exp = 0.0;
            for (int a = 0; a < q; a++) {
                expa[a] = exp(logits[a] - row_max);
                sum_exp += expa[a];
            }
            if (!isfinite(sum_exp) || sum_exp == 0.0) {
                local_bad += 1;
                continue;
            }
            double prob_sum = 0.0;
            for (int a = 0; a < q; a++) {
                const double probability = expa[a] / sum_exp;
                prob_sum += probability;
                if (probability < local_min_p) {
                    local_min_p = probability;
                }
            }
            const double error = fabs(prob_sum - 1.0);
            if (error > local_error) {
                local_error = error;
            }
        }
#pragma omp critical
        {
            if (local_min < *logit_min) {
                *logit_min = local_min;
            }
            if (local_max > *logit_max) {
                *logit_max = local_max;
            }
            if (local_range > *max_row_range) {
                *max_row_range = local_range;
            }
            if (local_error > *max_prob_sum_error) {
                *max_prob_sum_error = local_error;
            }
            if (local_min_p < *min_probability) {
                *min_probability = local_min_p;
            }
            *nonfinite_rows += local_bad;
        }
        free(logits);
        free(expa);
    }
    return 0;
}

/* Probabilities at theta and the logit change of one unit step along direction.
   Both arrays are row-major, length B * q. This does not change the objective. */

int site_direction_cache(
    const int8_t *msa,
    const double *theta,
    const double *direction,
    int B,
    int L,
    int q,
    int site,
    double *probability,
    double *direction_logit
) {
    const int K = L - 1;
    const double *h = theta;
    const double *J = theta + q;
    const double *dh = direction;
    const double *dJ = direction + q;

    for (int row = 0; row < B; row++) {
        const int8_t *sequence = msa + (size_t)row * (size_t)L;
        double *logits = probability + (size_t)row * (size_t)q;
        double *velocity = direction_logit + (size_t)row * (size_t)q;
        for (int a = 0; a < q; a++) {
            logits[a] = h[a];
            velocity[a] = dh[a];
        }
        int partner = 0;
        for (int column = 0; column < L; column++) {
            if (column == site) {
                continue;
            }
            const int state = (int)sequence[column];
            for (int a = 0; a < q; a++) {
                const size_t index = ((size_t)a * (size_t)K + (size_t)partner) * (size_t)q + (size_t)state;
                logits[a] += J[index];
                velocity[a] += dJ[index];
            }
            partner++;
        }
        double max_logit = logits[0];
        int finite = 1;
        for (int a = 0; a < q; a++) {
            if (!isfinite(logits[a]) || !isfinite(velocity[a])) {
                finite = 0;
            }
            if (logits[a] > max_logit) {
                max_logit = logits[a];
            }
        }
        if (!finite) {
            return -1;
        }
        double sum_exp = 0.0;
        for (int a = 0; a < q; a++) {
            logits[a] = exp(logits[a] - max_logit);
            sum_exp += logits[a];
        }
        if (!isfinite(sum_exp) || sum_exp == 0.0) {
            return -1;
        }
        for (int a = 0; a < q; a++) {
            logits[a] /= sum_exp;
        }
    }
    return 0;
}

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
        const double term = weights[row] * (max_shift + log(mass) - alpha * velocity[state]);
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
