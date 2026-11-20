"""Finite-look exact median bounds for a proposed i.i.d. mixture design."""
from __future__ import annotations

from fractions import Fraction
import json
import math


LOOKS = (12, 24, 48, 96)
ALPHA = Fraction(1, 20)
ALPHA_PER_TAIL = ALPHA / (len(LOOKS) * 2 * 5 * 2)


def binomial_half_tail(n, through):
    return Fraction(sum(math.comb(n, j) for j in range(through + 1)), 2**n)


def order_index(n, alpha_per_tail=ALPHA_PER_TAIL):
    """Return zero if no nontrivial symmetric order-statistic pair is possible."""
    k = 0
    for i in range(1, n // 2 + 1):
        if binomial_half_tail(n, i - 1) > alpha_per_tail:
            break
        k = i
    return k


def bracket(intervals, support):
    """Honor original draw order and retain every interval, including full support."""
    n = len(intervals)
    if n not in LOOKS:
        raise ValueError("Not a prescribed analysis look")
    if any(not (support[0] <= lo <= hi <= support[1]) for lo, hi in intervals):
        raise ValueError("All record intervals must be ordered and inside the declared support")
    k = order_index(n)
    if not k:
        return support
    return (sorted(lo for lo, _ in intervals)[k - 1],
            sorted(hi for _, hi in intervals)[n - k])


def assert_math():
    # Ties at the population lower median make either one-sided comparison
    # more conservative. Check several exact binomial laws for both tails.
    results = []
    for n in LOOKS:
        k = order_index(n)
        tail = binomial_half_tail(n, k - 1)
        assert k and tail <= ALPHA_PER_TAIL < binomial_half_tail(n, k)
        for p_less, p_at_median in ((Fraction(1, 2), Fraction(1, 2)), (Fraction(1, 5), Fraction(4, 5)),
                                    (Fraction(0), Fraction(1)), (Fraction(49, 100), Fraction(1, 2))):
            lower_error = sum(math.comb(n, j) * p_at_median**j * (1-p_at_median)**(n-j)
                              for j in range(k))
            upper_error = sum(math.comb(n, j) * p_less**j * (1-p_less)**(n-j)
                              for j in range(n-k+1, n+1))
            assert max(lower_error, upper_error) <= tail
        assert bracket([(0, 1)] * n, (0, 1)) == (0, 1)
        complete = [(0, 0)] * k + [(1, 1)] * (n - k)
        partial = [(0, 1)] * (k + 1) + complete[k+1:]
        assert bracket(complete, (0, 1)) == bracket(partial, (0, 1)) == (0, 1)
        assert bracket([(0.5, 0.5)] * n, (0, 1)) == (0.5, 0.5)
        results.append({"n_per_arm": n, "lower_order_1_based": k, "upper_order_1_based": n-k+1,
                        "one_sided_worst_case_error": float(tail)})
    print(json.dumps({"design": "candidate not approved; iid draws from each frozen law",
                      "familywise_alpha": float(ALPHA), "alpha_per_tail": float(ALPHA_PER_TAIL),
                      "looks": results}, indent=2))


if __name__ == "__main__":
    assert_math()
