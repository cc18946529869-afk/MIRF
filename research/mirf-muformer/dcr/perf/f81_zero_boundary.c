/* Structural feasibility in branch-length coordinates of a fixed tree. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    int32_t root, size, label;
} Component;

typedef struct {
    double step;
    uint64_t numerator, denominator;
    int exponent;
    int32_t child;
} Hit;

static Component *components(const int32_t *leaf_labels, int32_t leaves, int32_t edges) {
    Component *groups = malloc(((size_t)edges + 1) * sizeof(*groups));
    if (!groups) return NULL;
    for (int32_t i = 0; i <= edges; ++i) {
        groups[i] = (Component){i, 1, i < leaves ? leaf_labels[i] : -1};
    }
    return groups;
}

static int32_t find(Component *groups, int32_t i) {
    while (groups[i].root != i) {
        groups[i].root = groups[groups[i].root].root;
        i = groups[i].root;
    }
    return i;
}

static int unite(Component *groups, int32_t a, int32_t b) {
    a = find(groups, a);
    b = find(groups, b);
    if (a == b) return 1;
    if (groups[a].label >= 0 && groups[b].label >= 0 && groups[a].label != groups[b].label) return 0;
    if (groups[a].size < groups[b].size) {
        int32_t temp = a;
        a = b;
        b = temp;
    }
    groups[b].root = a;
    groups[a].size += groups[b].size;
    if (groups[a].label < 0) groups[a].label = groups[b].label;
    return 1;
}

int f81_zero_feasible(const int32_t *parent, const int32_t *label, int32_t leaves,
                      int32_t edges, const double *length) {
    Component *groups = components(label, leaves, edges);
    if (!groups) return -1;
    int feasible = 1;
    for (int32_t child = 0; child < edges; ++child) {
        if (length[child] == 0 && !unite(groups, child, parent[child])) {
            feasible = 0;
            break;
        }
    }
    free(groups);
    return feasible;
}

static Hit make_hit(double x, double speed, int32_t child) {
    int xe, de;
    double xf = frexp(x, &xe), df = frexp(speed, &de);
    /* frexp normalizes subnormals too. Both scaled mantissas are exact
     * 53-bit integers, so cross products fit in an unsigned 128-bit word. */
    return (Hit){x / speed, (uint64_t)ldexp(xf, 53), (uint64_t)ldexp(df, 53), xe - de, child};
}

static int compare_hit(const void *a, const void *b) {
    const Hit *left = a;
    const Hit *right = b;
    int shift = left->exponent - right->exponent;
    /* Each normalized mantissa ratio is between 1/2 and 2. */
    if (shift >= 2) return 1;
    if (shift <= -2) return -1;
    __uint128_t left_cross = (__uint128_t)left->numerator * right->denominator;
    __uint128_t right_cross = (__uint128_t)right->numerator * left->denominator;
    if (shift > 0) left_cross <<= 1;
    if (shift < 0) right_cross <<= 1;
    if (left_cross != right_cross) return left_cross < right_cross ? -1 : 1;
    return (left->child > right->child) - (left->child < right->child);
}

/* Compute the first infeasible point on the projected ray for alpha > 0.
 * Ties may be processed in any order: the conflict is attributed to the
 * common hit time, and component contraction cannot undo a conflict. */
int f81_first_conflict(const int32_t *parent, const int32_t *label, int32_t leaves,
                       int32_t edges, const double *point, const double *direction,
                       double *limit) {
    Component *groups = components(label, leaves, edges);
    Hit *hits = malloc((size_t)edges * sizeof(*hits));
    if (!groups || !hits) {
        free(groups);
        free(hits);
        return -1;
    }
    *limit = INFINITY;
    size_t count = 0;
    for (int32_t child = 0; child < edges; ++child) {
        if (point[child] == 0 && direction[child] <= 0) {
            if (!unite(groups, child, parent[child])) {
                *limit = 0;
                goto done;
            }
        } else if (point[child] > 0 && direction[child] < 0) {
            hits[count++] = make_hit(point[child], -direction[child], child);
        }
    }
    qsort(hits, count, sizeof(*hits), compare_hit);
    for (size_t j = 0; j < count; ++j) {
        int32_t child = hits[j].child;
        if (!unite(groups, child, parent[child])) {
            *limit = hits[j].step;
            break;
        }
    }
done:
    free(groups);
    free(hits);
    return 0;
}
