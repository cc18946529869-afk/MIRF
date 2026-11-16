"""Dirichlet mixture shared by μSearch sampling and its log-probability.

Sampling draws eta ~ Dirichlet(concentration) and forms

    mix_weight * pi + (1 - mix_weight) * eta.

That vector already sums to one. Averaging it over eta gives

    mix_weight * pi + (1 - mix_weight) / K.

Marginal mode writes the update log-probability from this average.
Legacy mode keeps the previous weight of 0.50. The Dirichlet draw itself
is unchanged in both modes.
"""

import numpy as np

POLICY_MIX_WEIGHT = 0.75
DIRICHLET_CONCENTRATION = 0.03
LEGACY_LOG_PROB_WEIGHT = 0.50


def mixed_action_probability(policy_probability, denominator, mode):
    """Probability of one action under the selected update formula."""
    if mode == "marginal":
        weight = POLICY_MIX_WEIGHT
    elif mode == "legacy":
        weight = LEGACY_LOG_PROB_WEIGHT
    else:
        raise ValueError("mixture mode must be 'marginal' or 'legacy'")
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return weight * policy_probability + (1.0 - weight) / denominator


def dirichlet_mixture_probabilities(policy_probability, rng=None):
    """One Dirichlet mixture vector. rng=None uses numpy's global random state."""
    policy_probability = np.asarray(policy_probability, dtype=np.float64)
    categories = int(policy_probability.shape[-1])
    generator = np.random if rng is None else rng
    noise = generator.dirichlet(np.full(categories, DIRICHLET_CONCENTRATION))
    mixed = (
        POLICY_MIX_WEIGHT * policy_probability
        + (1.0 - POLICY_MIX_WEIGHT) * noise
    )
    return mixed / mixed.sum()
