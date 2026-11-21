// Independent edge-split evaluation of selected mean negative log-likelihood scores.
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
int split_score(const uint8_t *observations, const double *profiles, int sites,
                int leaves, int nodes, const int32_t *order, const int32_t *parents,
                const double *lengths, int left, int right, double split_length,
                char *result) {
    using Float = boost::multiprecision::number<
        boost::multiprecision::cpp_bin_float<Bits, boost::multiprecision::digit_base_2>,
        boost::multiprecision::et_off>;
    using Partial = std::array<Float, states>;
    std::vector<Float> stay(nodes), redraw(nodes), scores(sites);
    std::vector<int> invalid(sites, 0);
    for (int node = 0; node < nodes; ++node) {
        if (parents[node] < 0) continue;
        const Float x = -Float(lengths[node]);
        stay[node] = exp(x);
        redraw[node] = -expm1(x);
    }
    const Float split_stay = exp(-Float(split_length));
    const Float split_redraw = -expm1(-Float(split_length));

// Postorder multiplies incoming subtree messages into their parent, independently
// for the two connected components defined by removing the selected edge.
    #pragma omp parallel
    {
        std::vector<Partial> partial(nodes);
        #pragma omp for schedule(static)
        for (int site = 0; site < sites; ++site) {
            Partial pi;
            for (int state = 0; state < states; ++state) {
                pi[state] = profiles[static_cast<size_t>(site) * states + state];
            }
            for (int leaf = 0; leaf < leaves; ++leaf) {
                partial[leaf].fill(Float(0));
                partial[leaf][observations[static_cast<size_t>(site) * leaves + leaf]] = 1;
            }
            for (int node = leaves; node < nodes; ++node) partial[node].fill(Float(1));
            for (int index = 0; index < nodes; ++index) {
                const int node = order[index];
                const int parent = parents[node];
                if (parent < 0) continue;
                const auto &inside = partial[node];
                auto &receiving = partial[parent];
                Float marginal = 0;
                for (int i = 0; i < states; ++i) marginal += pi[i] * inside[i];
                const Float mixed = redraw[node] * marginal;
                for (int i = 0; i < states; ++i) {
                    receiving[i] *= (stay[node] * inside[i] + mixed);
                }
            }
            Float same = 0, left_mass = 0, right_mass = 0;
            for (int i = 0; i < states; ++i) {
                const Float a = pi[i] * partial[left][i];
                same += a * partial[right][i];
                left_mass += a;
                right_mass += pi[i] * partial[right][i];
            }
            const Float independent = left_mass * right_mass;
            const Float mass = split_stay * same + split_redraw * independent;
            if (!(mass > 0) || !isfinite(mass)) {
                invalid[site] = 1;
            } else {
                scores[site] = split_stay * (same - independent) / mass;
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

extern "C" int f81_split_gradient(const uint8_t *observations, const double *profiles,
                                 int sites, int leaves, int nodes, const int32_t *order,
                                 const int32_t *parents, const double *lengths, int left,
                                 int right, double split_length, int bits, char *result) {
    try {
        if (bits == 192) {
            return split_score<192>(observations, profiles, sites, leaves, nodes, order,
                                    parents, lengths, left, right, split_length, result);
        }
        if (bits == 256) {
            return split_score<256>(observations, profiles, sites, leaves, nodes, order,
                                    parents, lengths, left, right, split_length, result);
        }
    } catch (...) {
        return 3;
    }
    return 4;
}
