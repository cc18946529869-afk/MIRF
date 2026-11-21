// Independent one-edge derivative by forward-mode differentiation of rooted pruning.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <boost/multiprecision/cpp_bin_float.hpp>

namespace {
constexpr int states = 21;
constexpr int result_capacity = 128;

template <unsigned Bits>
int differentiate(const uint8_t *observations, const double *profiles, int sites,
                  int leaves, int nodes, const int32_t *parents, const double *lengths,
                  int target, char *result) {
    using Float = boost::multiprecision::number<
        boost::multiprecision::cpp_bin_float<Bits, boost::multiprecision::digit_base_2>,
        boost::multiprecision::et_off>;
    using Partial = std::array<Float, states>;
    const int root = nodes - 1;
    std::vector<Float> keep(root), refresh(root), scores(sites);
    std::vector<int> invalid(sites);
    for (int i = 0; i < root; ++i) {
        const Float exponent = -Float(lengths[i]);
        keep[i] = exp(exponent);
        refresh[i] = -expm1(exponent);
    }

    #pragma omp parallel
    {
        std::vector<Partial> value(nodes), derivative(nodes);
        std::vector<uint8_t> active(nodes);
        #pragma omp for schedule(static)
        for (int site = 0; site < sites; ++site) {
            Partial pi;
            std::fill(active.begin(), active.end(), 0);
            for (int i = 0; i < states; ++i) pi[i] = profiles[static_cast<size_t>(site) * states + i];
            for (int leaf = 0; leaf < leaves; ++leaf) {
                value[leaf].fill(Float(0));
                value[leaf][observations[static_cast<size_t>(site) * leaves + leaf]] = 1;
            }
            for (int node = leaves; node < nodes; ++node) value[node].fill(Float(1));
            for (int child = 0; child < root; ++child) {
                const int parent = parents[child];
                Float mean = 0, mean_derivative = 0;
                for (int a = 0; a < states; ++a) {
                    mean += pi[a] * value[child][a];
                    if (active[child]) mean_derivative += pi[a] * derivative[child][a];
                }
                const Float common = refresh[child] * mean;
                const Float common_derivative = refresh[child] * mean_derivative;
                const bool carries_derivative = child == target || active[child];
                for (int a = 0; a < states; ++a) {
                    const Float message = keep[child] * value[child][a] + common;
                    if (active[parent]) derivative[parent][a] *= message;
                    if (carries_derivative) {
                        const Float tangent = child == target
                            ? keep[child] * (mean - value[child][a])
                            : keep[child] * derivative[child][a] + common_derivative;
                        if (active[parent]) derivative[parent][a] += value[parent][a] * tangent;
                        else derivative[parent][a] = value[parent][a] * tangent;
                    }
                    value[parent][a] *= message;
                }
                active[parent] |= carries_derivative;
            }
            Float probability = 0, tangent = 0;
            for (int a = 0; a < states; ++a) {
                probability += pi[a] * value[root][a];
                tangent += pi[a] * derivative[root][a];
            }
            if (!(probability > 0) || !isfinite(probability)) {
                invalid[site] = 1;
            } else {
                scores[site] = -(tangent / probability);
                if (!isfinite(scores[site])) invalid[site] = 1;
            }
        }
    }
    if (std::any_of(invalid.begin(), invalid.end(), [](int x) { return x != 0; })) return 1;
    Float total = 0;
    for (const Float &score : scores) total += score;
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<Float>::max_digits10) << (total / sites);
    const int written = std::snprintf(result, result_capacity, "%s", out.str().c_str());
    return written < 0 || written >= result_capacity ? 2 : 0;
}
}  // namespace

extern "C" int f81_dual_gradient(const uint8_t *observations, const double *profiles,
                                int sites, int leaves, int nodes, const int32_t *parents,
                                const double *lengths, int target, int bits, char *result) {
    try {
        if (target < 0 || target >= nodes - 1) return 4;
        if (bits == 192) {
            return differentiate<192>(observations, profiles, sites, leaves, nodes, parents, lengths, target, result);
        }
        if (bits == 256) {
            return differentiate<256>(observations, profiles, sites, leaves, nodes, parents, lengths, target, result);
        }
    } catch (...) {
        return 3;
    }
    return 4;
}
