/* Fixed-topology site-profile likelihood and analytic effective-length scores. */
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { Q = 21, OK = 0, IMPOSSIBLE = 1, RANGE_ERROR = 2, NO_MEMORY = 3 };

static int normalize(double x[Q], double *log_scale) {
    double maximum = 0;
    for (int a = 0; a < Q; ++a) if (x[a] > maximum) maximum = x[a];
    if (maximum == 0) return IMPOSSIBLE;
    if (log_scale) *log_scale += log(maximum);
    for (int a = 0; a < Q; ++a) {
        double value = x[a] / maximum;
        if (value < DBL_MIN && x[a] > 0) return RANGE_ERROR;
        x[a] = value;
    }
    return OK;
}

static void ones(double x[Q]) {
    for (int a = 0; a < Q; ++a) x[a] = 1;
}

static int multiply_normalize(double out[Q], const double left[Q],
                              const double right[Q], double *log_scale) {
    for (int a = 0; a < Q; ++a) {
        double value = left[a] * right[a];
        if (value < DBL_MIN && left[a] > 0 && right[a] > 0) return RANGE_ERROR;
        out[a] = value;
    }
    return normalize(out, log_scale);
}

static int dot_nonnegative(const double left[Q], const double right[Q], double *result) {
    double total = 0;
    for (int a = 0; a < Q; ++a) {
        double term = left[a] * right[a];
        if (term < DBL_MIN && left[a] > 0 && right[a] > 0) return RANGE_ERROR;
        total += term;
    }
    *result = total;
    return OK;
}

static int message(double out[Q], const double *inside, const double *pi,
                   double stay, double redraw) {
    double mean;
    int status = dot_nonnegative(pi, inside, &mean);
    if (status != OK) return status;
    double common = redraw * mean;
    for (int a = 0; a < Q; ++a) {
        double same = stay * inside[a];
        if (common < DBL_MIN && redraw > 0 && mean > 0 && same < DBL_MIN) return RANGE_ERROR;
        if (same < DBL_MIN && stay > 0 && inside[a] > 0 && common < DBL_MIN) return RANGE_ERROR;
        out[a] = same + common;
    }
    return OK;
}

static int upward(const uint8_t *state, const double *pi, int32_t n, int32_t leaves,
                  const int32_t *offsets, const int32_t *children, const double *stay,
                  const double *redraw, double *inside, double *messages,
                  double *loglik) {
    double log_scale = 0;
    for (int32_t node = 0; node < n; ++node) {
        double *partial = inside + (size_t)node * Q;
        if (node < leaves) {
            memset(partial, 0, Q * sizeof(*partial));
            partial[state[node]] = 1;
        } else {
            ones(partial);
            for (int32_t k = offsets[node]; k < offsets[node + 1]; ++k) {
                const double *child_message = messages + (size_t)children[k] * Q;
                int status = multiply_normalize(partial, partial, child_message, &log_scale);
                if (status != OK) return status;
            }
        }
        if (node != n - 1) {
            int status = message(messages + (size_t)node * Q, partial, pi, stay[node], redraw[node]);
            if (status != OK) return status;
        }
    }
    double root_mass;
    int status = dot_nonnegative(pi, inside + (size_t)(n - 1) * Q, &root_mass);
    if (status != OK) return status;
    if (root_mass == 0) return IMPOSSIBLE;
    *loglik = log_scale + log(root_mass);
    return isfinite(*loglik) ? OK : RANGE_ERROR;
}

static int edge_score(const double *excluding, const double *incoming, const double *child,
                      const double *pi, double stay, double redraw,
                      double *outside_child, double *gradient) {
    double child_mean, denominator;
    int status = dot_nonnegative(child, pi, &child_mean);
    if (status != OK) return status;
    status = dot_nonnegative(excluding, incoming, &denominator);
    if (status != OK) return status;
    if (denominator <= 0) return RANGE_ERROR; /* The up-pass already proved positive mass. */
    double outside_sum = 0, grad = 0;
    for (int a = 0; a < Q; ++a) {
        outside_sum += excluding[a];
        grad += excluding[a] * (child_mean - child[a]);
    }
    double term = (stay * grad) / denominator;
    if (!isfinite(term) || !isfinite(*gradient + term)) return RANGE_ERROR;
    *gradient += term;
    for (int a = 0; a < Q; ++a) {
        double same = stay * excluding[a];
        double changed = redraw * pi[a] * outside_sum;
        if (same == 0 && changed == 0 &&
            ((stay > 0 && excluding[a] > 0) || (redraw > 0 && pi[a] > 0))) return RANGE_ERROR;
        outside_child[a] = same + changed;
    }
    return normalize(outside_child, NULL);
}

static int downward(const double *pi, int32_t n, int32_t leaves, const int32_t *offsets,
                    const int32_t *children, const double *stay, const double *redraw,
                    const double *inside, const double *messages, double *outside,
                    double *prefix, double *gradient) {
    memcpy(outside + (size_t)(n - 1) * Q, pi, Q * sizeof(*pi));
    for (int32_t node = n - 1; node >= leaves; --node) {
        int32_t first = offsets[node], count = offsets[node + 1] - first;
        memcpy(prefix, outside + (size_t)node * Q, Q * sizeof(*prefix));
        for (int32_t k = 0; k < count - 1; ++k) {
            const double *msg = messages + (size_t)children[first + k] * Q;
            int status = multiply_normalize(prefix + (size_t)(k + 1) * Q,
                                            prefix + (size_t)k * Q, msg, NULL);
            if (status != OK) return status == IMPOSSIBLE ? RANGE_ERROR : status;
        }
        double suffix[Q];
        ones(suffix);
        for (int32_t k = count - 1; k >= 0; --k) {
            int32_t child = children[first + k];
            const double *msg = messages + (size_t)child * Q;
            double excluding[Q];
            int status = multiply_normalize(excluding, prefix + (size_t)k * Q, suffix, NULL);
            if (status != OK) return status == IMPOSSIBLE ? RANGE_ERROR : status;
            status = edge_score(excluding, msg, inside + (size_t)child * Q, pi,
                                stay[child], redraw[child], outside + (size_t)child * Q,
                                gradient + child);
            if (status != OK) return status == IMPOSSIBLE ? RANGE_ERROR : status;
            if (k > 0) {
                status = multiply_normalize(suffix, suffix, msg, NULL);
                if (status != OK) return status == IMPOSSIBLE ? RANGE_ERROR : status;
            }
        }
    }
    return OK;
}

/* Log-domain replay handles deep products and finite lengths whose
 * retention is smaller than the binary64 normal range. */
static double log_add(double left, double right) {
    if (isnan(left) || isnan(right)) return NAN;
    if (left == -INFINITY) return right;
    if (right == -INFINITY) return left;
    double maximum = fmax(left, right);
    return maximum + log1p(exp(-fabs(left - right)));
}

static double log_dot(const double left[Q], const double right[Q]) {
    double result = -INFINITY;
    for (int a = 0; a < Q; ++a) {
        double term = left[a] + right[a];
        if (term == -INFINITY && isfinite(left[a]) && isfinite(right[a])) return NAN;
        result = log_add(result, term);
    }
    return result;
}

static int log_normalize(double x[Q], double *scale) {
    double maximum = -INFINITY;
    for (int a = 0; a < Q; ++a) {
        if (isnan(x[a]) || x[a] == INFINITY) return RANGE_ERROR;
        if (x[a] > maximum) maximum = x[a];
    }
    if (maximum == -INFINITY) return IMPOSSIBLE;
    if (scale) *scale += maximum;
    for (int a = 0; a < Q; ++a) {
        double centered = x[a] - maximum;
        if (!isfinite(centered) && x[a] != -INFINITY) return RANGE_ERROR;
        x[a] = centered;
    }
    return OK;
}

static int log_multiply_normalize(double out[Q], const double left[Q],
                                  const double right[Q], double *scale) {
    for (int a = 0; a < Q; ++a) {
        double term = left[a] + right[a];
        if (term == -INFINITY && isfinite(left[a]) && isfinite(right[a])) return RANGE_ERROR;
        out[a] = term;
    }
    return log_normalize(out, scale);
}

static int log_upward(const uint8_t *states, const double *pi, int32_t n, int32_t leaves,
                      const int32_t *offsets, const int32_t *children,
                      const double *length, const double *log_redraw,
                      double *inside, double *messages, double *loglik) {
    double log_scale = 0;
    for (int32_t node = 0; node < n; ++node) {
        double *partial = inside + (size_t)node * Q;
        if (node < leaves) {
            for (int a = 0; a < Q; ++a) partial[a] = -INFINITY;
            partial[states[node]] = 0;
        } else {
            memset(partial, 0, Q * sizeof(*partial));
            for (int32_t k = offsets[node]; k < offsets[node + 1]; ++k) {
                const double *msg = messages + (size_t)children[k] * Q;
                int status = log_multiply_normalize(partial, partial, msg, &log_scale);
                if (status != OK) return status;
            }
        }
        if (node != n - 1) {
            double mean = log_dot(pi, partial);
            double common = log_redraw[node] + mean;
            if (common == -INFINITY && isfinite(log_redraw[node]) && isfinite(mean)) return RANGE_ERROR;
            double *msg = messages + (size_t)node * Q;
            for (int a = 0; a < Q; ++a) {
                double same = partial[a] - length[node];
                if (same == -INFINITY && isfinite(partial[a])) return RANGE_ERROR;
                msg[a] = log_add(same, common);
            }
        }
    }
    double root = log_dot(pi, inside + (size_t)(n - 1) * Q);
    if (root == -INFINITY) return IMPOSSIBLE;
    *loglik = log_scale + root;
    return isfinite(*loglik) ? OK : RANGE_ERROR;
}

static int log_edge_score(const double *excluding, const double *incoming, const double *child,
                          const double *pi, double length, double log_redraw,
                          double *outside_child, double *gradient) {
    double outside_sum = -INFINITY;
    for (int a = 0; a < Q; ++a) outside_sum = log_add(outside_sum, excluding[a]);
    double mean = log_dot(pi, child);
    double first = outside_sum + mean;
    double second = log_dot(excluding, child);
    double denom = log_dot(excluding, incoming);
    if (!isfinite(denom) || isnan(first) || isnan(second)) return RANGE_ERROR;
    if (first == second) {
        *gradient = 0;
    } else {
        double difference = -expm1(-fabs(first - second));
        double magnitude = exp((fmax(first, second) - denom) - length + log(difference));
        if (!isfinite(magnitude)) return RANGE_ERROR;
        *gradient = first > second ? magnitude : -magnitude;
    }
    for (int a = 0; a < Q; ++a) {
        double same = excluding[a] - length;
        double changed = log_redraw + pi[a] + outside_sum;
        if (same == -INFINITY && isfinite(excluding[a])) return RANGE_ERROR;
        if (changed == -INFINITY && isfinite(log_redraw) && isfinite(pi[a])) return RANGE_ERROR;
        outside_child[a] = log_add(same, changed);
    }
    return log_normalize(outside_child, NULL);
}

static int log_downward(const double *pi, int32_t n, int32_t leaves, const int32_t *offsets,
                        const int32_t *children, const double *length, const double *log_redraw,
                        const double *inside, const double *messages, double *outside,
                        double *prefix, double *gradient) {
    memcpy(outside + (size_t)(n - 1) * Q, pi, Q * sizeof(*pi));
    for (int32_t node = n - 1; node >= leaves; --node) {
        int32_t first = offsets[node], count = offsets[node + 1] - first;
        memcpy(prefix, outside + (size_t)node * Q, Q * sizeof(*prefix));
        for (int32_t k = 0; k < count - 1; ++k) {
            const double *msg = messages + (size_t)children[first + k] * Q;
            if (log_multiply_normalize(prefix + (size_t)(k + 1) * Q,
                                      prefix + (size_t)k * Q, msg, NULL) != OK) return RANGE_ERROR;
        }
        double suffix[Q] = {0};
        for (int32_t k = count - 1; k >= 0; --k) {
            int32_t child = children[first + k];
            const double *msg = messages + (size_t)child * Q;
            double excluding[Q];
            if (log_multiply_normalize(excluding, prefix + (size_t)k * Q, suffix, NULL) != OK) return RANGE_ERROR;
            if (log_edge_score(excluding, msg, inside + (size_t)child * Q, pi,
                               length[child], log_redraw[child],
                               outside + (size_t)child * Q, gradient + child) != OK) return RANGE_ERROR;
            if (k > 0 && log_multiply_normalize(suffix, suffix, msg, NULL) != OK) return RANGE_ERROR;
        }
    }
    return OK;
}

/* Inputs are validated by the caller: internal parent ids exceed child ids,
 * leaves correspond to rows, and edge slots are indexed by child id. */
enum { GRADIENT_BLOCK_SIZE = 16 };

int f81_gradient_block_size(void) {
    return GRADIENT_BLOCK_SIZE;
}

void f81_sites_fixed_blocks_diagonal(const uint8_t *states, const double *profiles, int32_t nsites,
               int32_t n, int32_t leaves, const int32_t *offsets,
               const int32_t *children, const double *stay, const double *redraw,
               const double *length, const double *log_redraw, int32_t force_log,
               int32_t max_degree, int32_t threads, int32_t need_gradient,
               double *gradient, double *site_loglik, int32_t *statuses, uint8_t *log_used, double *diagonal) {
    size_t bytes = (size_t)n * Q * sizeof(double);
    #pragma omp parallel num_threads(threads)
    {
        double *inside = malloc(bytes);
        double *messages = malloc(bytes);
        double *outside = need_gradient ? malloc(bytes) : NULL;
        double *prefix = need_gradient ? malloc((size_t)max_degree * Q * sizeof(double)) : NULL;
        size_t gradient_bytes = (size_t)(n - 1) * sizeof(double);
        double *contribution = need_gradient ? malloc(gradient_bytes ? gradient_bytes : 1) : NULL;
        int allocated = inside && messages && (!need_gradient || (outside && prefix && contribution));
        #pragma omp for schedule(static)
        for (int32_t block = 0; block < (nsites - 1) / GRADIENT_BLOCK_SIZE + 1; ++block) {
            int32_t begin = block * GRADIENT_BLOCK_SIZE;
            int32_t end = nsites - begin < GRADIENT_BLOCK_SIZE ? nsites : begin + GRADIENT_BLOCK_SIZE;
            for (int32_t site = begin; site < end; ++site) {
                if (!allocated) {
                    statuses[site] = NO_MEMORY;
                    continue;
                }
                const double *pi = profiles + (size_t)site * Q;
                const uint8_t *state = states + (size_t)site * leaves;
                int status = RANGE_ERROR;
                if (!force_log) {
                    status = upward(state, pi, n, leaves, offsets, children, stay, redraw,
                                    inside, messages, site_loglik + site);
                    if (status == OK && need_gradient) {
                        memset(contribution, 0, gradient_bytes);
                        status = downward(pi, n, leaves, offsets, children, stay, redraw,
                                          inside, messages, outside, prefix, contribution);
                    }
                }
                if (status == RANGE_ERROR) {
                    double log_pi[Q];
                    for (int a = 0; a < Q; ++a) log_pi[a] = pi[a] > 0 ? log(pi[a]) : -INFINITY;
                    status = log_upward(state, log_pi, n, leaves, offsets, children, length, log_redraw,
                                        inside, messages, site_loglik + site);
                    if (status == OK && need_gradient) {
                        status = log_downward(log_pi, n, leaves, offsets, children, length, log_redraw,
                                              inside, messages, outside, prefix, contribution);
                    }
                    log_used[site] = 1;
                }
                if (status == OK && need_gradient) {
                    double *total = gradient + (size_t)block * (n - 1);
                    for (int32_t edge = 0; edge < n - 1; ++edge) total[edge] += contribution[edge];
                    if (diagonal) {
                        double *hessian = diagonal + (size_t)block * (n - 1);
                        for (int32_t edge = 0; edge < n - 1; ++edge) {
                            /* Native contribution is the log-likelihood score.
                             * Return the diagonal for negative log likelihood. */
                            double score = contribution[edge];
                            hessian[edge] += score * (score + 1);
                        }
                    }
                }
                statuses[site] = status;
            }
        }
        free(contribution);
        free(prefix);
        free(outside);
        free(messages);
        free(inside);
    }
}

/* Keep the gradient-only ABI used by archived runners. */
void f81_sites_fixed_blocks(const uint8_t *states, const double *profiles, int32_t nsites,
               int32_t n, int32_t leaves, const int32_t *offsets,
               const int32_t *children, const double *stay, const double *redraw,
               const double *length, const double *log_redraw, int32_t force_log,
               int32_t max_degree, int32_t threads, int32_t need_gradient,
               double *gradient, double *site_loglik, int32_t *statuses, uint8_t *log_used) {
    f81_sites_fixed_blocks_diagonal(states, profiles, nsites, n, leaves, offsets,
                                   children, stay, redraw, length, log_redraw,
                                   force_log, max_degree, threads, need_gradient,
                                   gradient, site_loglik, statuses, log_used, NULL);
}
