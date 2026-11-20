#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* Same external layout as conditional.c: h, then J[a, partner, state].
   The working copy is J[partner, state, a]. Gradients are written back in
   the external order. No fast-math, and no change to the penalty. */

static void pack_transposed(const double *src, double *dst, int K, int q) {
    for (int a = 0; a < q; a++) {
        for (int partner = 0; partner < K; partner++) {
            for (int state = 0; state < q; state++) {
                const size_t from = ((size_t)a * (size_t)K + (size_t)partner) * (size_t)q + (size_t)state;
                const size_t to = ((size_t)partner * (size_t)q + (size_t)state) * (size_t)q + (size_t)a;
                dst[to] = src[from];
            }
        }
    }
}

static void add_unpacked(const double *src, double *dst, int K, int q) {
    for (int a = 0; a < q; a++) {
        for (int partner = 0; partner < K; partner++) {
            for (int state = 0; state < q; state++) {
                const size_t from = ((size_t)partner * (size_t)q + (size_t)state) * (size_t)q + (size_t)a;
                const size_t to = ((size_t)a * (size_t)K + (size_t)partner) * (size_t)q + (size_t)state;
                dst[to] += src[from];
            }
        }
    }
}

#ifdef MIRF_B_FIT_SINGLE_THREAD
#include <fenv.h>
#include <float.h>

static _Thread_local int fit_team_size;

int site_objective_team_size(void) {
    return fit_team_size;
}

int site_fit_fp_ready(void) {
    volatile double smallest_normal = DBL_MIN;
    volatile double smallest_subnormal = DBL_TRUE_MIN;
    return fegetround() == FE_TONEAREST
        && smallest_normal * 0.5 > 0.0
        && smallest_subnormal + smallest_subnormal > 0.0;
}
#endif

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
    double *Jt = (double *)malloc(sizeof(double) * (size_t)j_size);
    if (Jt == NULL) {
        return -1;
    }
    pack_transposed(J, Jt, K, q);
    double nll = 0.0;
    int failed = 0;

#ifdef MIRF_B_FIT_SINGLE_THREAD
    fit_team_size = 0;
    if (!site_fit_fp_ready()) {
        free(Jt);
        return -2;
    }
#endif
#pragma omp parallel reduction(+ : nll)
    {
#ifdef MIRF_B_FIT_SINGLE_THREAD
        const int team_size = omp_get_num_threads();
        if (omp_get_thread_num() == 0) {
            fit_team_size = team_size;
        }
#endif
        double *gh = (double *)calloc((size_t)q, sizeof(double));
        double *gJt = (double *)calloc((size_t)j_size, sizeof(double));
        double *logits = (double *)malloc(sizeof(double) * (size_t)q);
        double *expa = (double *)malloc(sizeof(double) * (size_t)q);
        if (gh == NULL || gJt == NULL || logits == NULL || expa == NULL) {
#pragma omp atomic write
            failed = 1;
        }
#pragma omp for schedule(static)
        for (int row = 0; row < B; row++) {
#ifdef MIRF_B_FIT_SINGLE_THREAD
            if (team_size != 1) {
                continue;
            }
#endif
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
                const double *block = Jt + ((size_t)partner * (size_t)q + (size_t)state) * (size_t)q;
                for (int a = 0; a < q; a++) {
                    logits[a] += block[a];
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
                double *block = gJt + ((size_t)partner * (size_t)q + (size_t)state) * (size_t)q;
                for (int a = 0; a < q; a++) {
                    block[a] += expa[a];
                }
                partner++;
            }
        }
#pragma omp critical
        {
            if (gh != NULL && gJt != NULL) {
                for (int a = 0; a < q; a++) {
                    grad[a] += gh[a];
                }
                add_unpacked(gJt, grad + q, K, q);
            }
        }
        free(gh);
        free(gJt);
        free(logits);
        free(expa);
    }
    free(Jt);
#ifdef MIRF_B_FIT_SINGLE_THREAD
    if (fit_team_size != 1) {
        return -2;
    }
#endif
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
    const int j_size = q * K * q;
    const double *h = theta;
    const double *J = theta + q;
    const double *dh = direction;
    const double *dJ = direction + q;
    double *Jt = (double *)malloc(sizeof(double) * (size_t)j_size);
    double *dJt = (double *)malloc(sizeof(double) * (size_t)j_size);
    if (Jt == NULL || dJt == NULL) {
        free(Jt);
        free(dJt);
        return -1;
    }
    pack_transposed(J, Jt, K, q);
    pack_transposed(dJ, dJt, K, q);

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
            const size_t base = ((size_t)partner * (size_t)q + (size_t)state) * (size_t)q;
            const double *jblock = Jt + base;
            const double *dblock = dJt + base;
            for (int a = 0; a < q; a++) {
                logits[a] += jblock[a];
                velocity[a] += dblock[a];
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
            free(Jt);
            free(dJt);
            return -1;
        }
        double sum_exp = 0.0;
        for (int a = 0; a < q; a++) {
            logits[a] = exp(logits[a] - max_logit);
            sum_exp += logits[a];
        }
        if (!isfinite(sum_exp) || sum_exp == 0.0) {
            free(Jt);
            free(dJt);
            return -1;
        }
        for (int a = 0; a < q; a++) {
            logits[a] /= sum_exp;
        }
    }
    free(Jt);
    free(dJt);
    return 0;
}
