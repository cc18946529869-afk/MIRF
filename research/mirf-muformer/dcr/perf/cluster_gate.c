/* Exact two-threshold connected components on fixed categorical rows. */
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>

enum { WIDTH = 263, MAIN_LIMIT = 52, ALT_LIMIT = 26 };
static const uint32_t ALT_BIT = UINT32_C(1) << 31;

static int mismatches(const uint8_t *left, const uint8_t *right) {
    int count = 0;
    for (int k = 0; k < 256; k += 32) {
        __m256i lhs = _mm256_loadu_si256((const __m256i *)(left + k));
        __m256i rhs = _mm256_loadu_si256((const __m256i *)(right + k));
        unsigned same = (unsigned)_mm256_movemask_epi8(_mm256_cmpeq_epi8(lhs, rhs));
        count += 32 - __builtin_popcount(same);
        if (count > MAIN_LIMIT) return count;
    }
    for (int k = 256; k < WIDTH; ++k) count += left[k] != right[k];
    return count;
}

static uint32_t root(uint32_t *parents, uint32_t row) {
    while (parents[row] != row) {
        parents[row] = parents[parents[row]];
        row = parents[row];
    }
    return row;
}

static void connect(uint32_t *parents, uint32_t left, uint32_t right) {
    left = root(parents, left);
    right = root(parents, right);
    if (left < right) parents[right] = left;
    if (left > right) parents[left] = right;
}

/* The caller provides checked per-row neighbor counts (including self).
 * Slots store upward edges only, so the known non-self degree is an upper bound.
 * A scan must reproduce the entire reference degree vector before use. */
int cluster_scan(const uint8_t *rows, uint32_t n, const int64_t *offset,
                 const double *weights, uint32_t *parents_main, uint32_t *parents_alt,
                 double *edge_mass_main, double *edge_mass_alt,
                 int64_t *pairs_main, int64_t *pairs_alt, int threads) {
    size_t slots = offset[n] > 0 ? (size_t)offset[n] : 1;
    uint32_t *edges = malloc(slots * sizeof(*edges));
    int64_t *counts = calloc(n, sizeof(*counts));
    int64_t *degree = calloc(n, sizeof(*degree));
    int overflow = 0;
    if (!edges || !counts || !degree) {
        free(edges);
        free(counts);
        free(degree);
        return 1;
    }
    #pragma omp parallel for num_threads(threads) schedule(guided, 32) reduction(|:overflow)
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t *left = rows + (size_t)i * WIDTH;
        int64_t used = offset[i];
        for (uint32_t j = i + 1; j < n; ++j) {
            const uint8_t *right = rows + (size_t)j * WIDTH;
            int distance = mismatches(left, right);
            if (distance <= MAIN_LIMIT) {
                if (used == offset[i + 1]) {
                    overflow = 1;
                    break;
                }
                edges[used++] = j | (distance <= ALT_LIMIT ? ALT_BIT : 0);
            }
        }
        counts[i] = used;
    }
    if (overflow) goto cleanup;

    for (uint32_t i = 0; i < n; ++i) {
        parents_main[i] = i;
        parents_alt[i] = i;
    }
    for (uint32_t i = 0; i < n; ++i) {
        for (int64_t k = offset[i]; k < counts[i]; ++k) {
            uint32_t j = edges[k] & ~ALT_BIT;
            connect(parents_main, i, j);
            ++degree[i];
            ++degree[j];
            ++*pairs_main;
            if (edges[k] & ALT_BIT) {
                connect(parents_alt, i, j);
                ++*pairs_alt;
            }
        }
    }
    for (uint32_t i = 0; i < n; ++i) {
        if (degree[i] != offset[i + 1] - offset[i]) overflow = 1;
        parents_main[i] = root(parents_main, i);
        parents_alt[i] = root(parents_alt, i);
    }
    if (overflow) goto cleanup;

    for (uint32_t i = 0; i < n; ++i) {
        for (int64_t k = offset[i]; k < counts[i]; ++k) {
            uint32_t j = edges[k] & ~ALT_BIT;
            double mass = weights[i] * weights[j];
            edge_mass_main[parents_main[i]] += mass;
            if (edges[k] & ALT_BIT) edge_mass_alt[parents_alt[i]] += mass;
        }
    }
cleanup:
    free(edges);
    free(counts);
    free(degree);
    return overflow ? 2 : 0;
}
