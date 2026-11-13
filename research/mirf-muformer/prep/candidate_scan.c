/* Exact candidate scan for the 53-block neighbor test.
 *
 * Each unordered pair is visited once, from the smaller row index, after the
 * block buckets are merged. The distance scan stops at 53 mismatches. These
 * routines do not change which pairs are neighbors.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

void unique_candidate_degree(
    int32_t n_rows,
    int32_t n_blocks,
    const int32_t *starts,
    const int32_t *ends,
    const int32_t *members,
    int64_t *degree,
    int64_t *unique_pairs
) {
    int32_t *visited = calloc((size_t)n_rows, sizeof(int32_t));
    int64_t pairs = 0;
    int32_t row;

    if (visited == NULL) {
        *unique_pairs = -1;
        return;
    }
    for (row = 0; row < n_rows; row++) {
        int32_t stamp = row + 1;
        int32_t block;
        for (block = 0; block < n_blocks; block++) {
            size_t base = (size_t)block * (size_t)n_rows;
            int32_t start = starts[base + (size_t)row];
            int32_t end = ends[base + (size_t)row];
            const int32_t *group = members + base;
            int32_t cursor;
            for (cursor = start; cursor < end; cursor++) {
                int32_t other = group[cursor];
                if (other <= row || visited[other] == stamp) {
                    continue;
                }
                visited[other] = stamp;
                pairs += 1;
                degree[row] += 1;
                degree[other] += 1;
            }
        }
    }
    free(visited);
    *unique_pairs = pairs;
}

void early_abort_sample(
    const uint8_t *rows,
    int32_t n_rows,
    int32_t n_columns,
    int32_t n_blocks,
    const int32_t *starts,
    const int32_t *ends,
    const int32_t *members,
    const int32_t *column_block,
    const int32_t *column_order,
    const int32_t *sample_rows,
    int32_t n_sample,
    int32_t higher_only,
    int32_t mismatch_limit,
    int64_t *reject_hist,
    int64_t *accept_hist,
    int64_t *n_reject,
    int64_t *n_accept,
    int64_t *columns_compared,
    int64_t *neighbor_counts
) {
    int32_t *visited = calloc((size_t)n_rows, sizeof(int32_t));
    int32_t sample;
    int64_t rejected = 0;
    int64_t accepted = 0;
    int64_t compared_total = 0;

    if (visited == NULL) {
        *n_reject = -1;
        *n_accept = -1;
        *columns_compared = -1;
        return;
    }
    for (sample = 0; sample < n_sample; sample++) {
        int32_t row = sample_rows[sample];
        int32_t stamp = sample + 1;
        int32_t block;
        const uint8_t *left = rows + (size_t)row * (size_t)n_columns;
        for (block = 0; block < n_blocks; block++) {
            size_t base = (size_t)block * (size_t)n_rows;
            int32_t start = starts[base + (size_t)row];
            int32_t end = ends[base + (size_t)row];
            const int32_t *group = members + base;
            int32_t cursor;
            for (cursor = start; cursor < end; cursor++) {
                int32_t other = group[cursor];
                const uint8_t *right;
                int32_t mismatches = 0;
                int32_t compared = 0;
                int32_t position;
                if (other == row || (higher_only && other < row) || visited[other] == stamp) {
                    continue;
                }
                visited[other] = stamp;
                right = rows + (size_t)other * (size_t)n_columns;
                for (position = 0; position < n_columns; position++) {
                    int32_t column = column_order[position];
                    if (column_block[column] == block) {
                        continue;
                    }
                    compared += 1;
                    if (left[column] != right[column]) {
                        mismatches += 1;
                        if (mismatches == mismatch_limit) {
                            break;
                        }
                    }
                }
                compared_total += compared;
                if (mismatches == mismatch_limit) {
                    reject_hist[compared] += 1;
                    rejected += 1;
                } else {
                    accept_hist[compared] += 1;
                    accepted += 1;
                    if (neighbor_counts != NULL) {
                        neighbor_counts[row] += 1;
                        neighbor_counts[other] += 1;
                    }
                }
            }
        }
    }
    free(visited);
    *n_reject = rejected;
    *n_accept = accepted;
    *columns_compared = compared_total;
}

static int popcount64(uint64_t value) {
    return __builtin_popcountll(value);
}

void hamming_bitplane_pairs(
    const uint64_t *planes,
    int32_t n_rows,
    const int32_t *left_rows,
    const int32_t *right_rows,
    int32_t n_pairs,
    int32_t early_limit,
    uint16_t *distances,
    uint16_t *words_used
) {
    int32_t pair;
    (void)n_rows;
    for (pair = 0; pair < n_pairs; pair++) {
        size_t left = (size_t)left_rows[pair] * 25u;
        size_t right = (size_t)right_rows[pair] * 25u;
        int distance = 0;
        int used = 0;
        int word;
        for (word = 0; word < 5; word++) {
            uint64_t delta = 0;
            int bit;
            for (bit = 0; bit < 5; bit++) {
                size_t offset = (size_t)bit * 5u + (size_t)word;
                delta |= planes[left + offset] ^ planes[right + offset];
            }
            distance += popcount64(delta);
            used += 1;
            if (early_limit > 0 && distance >= early_limit) {
                break;
            }
        }
        distances[pair] = (uint16_t)distance;
        words_used[pair] = (uint16_t)used;
    }
}
