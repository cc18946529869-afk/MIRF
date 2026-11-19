"""Synthetic checks for the identity residual, not the Bootstrap result."""

from __future__ import annotations

import unittest

import numpy as np

import identity_audit


class IdentityAuditTests(unittest.TestCase):
    def test_normalized_residual_uses_one_when_the_terms_are_small(self) -> None:
        self.assertEqual(identity_audit.normalized_residual(2e-12, (0.0, 0.0, 0.0, 0.0)), 2e-12)

    def test_normalized_residual_uses_the_sum_of_absolute_values(self) -> None:
        self.assertEqual(identity_audit.normalized_residual(1.0, (2.0, -2.0)), 0.25)

    def test_random_parameters_meet_the_candidate_bound(self) -> None:
        generator = np.random.default_rng(17)
        length = 4
        states = identity_audit.STATES
        fields = generator.normal(size=(length, states))
        couplings = generator.normal(size=(length, length, states, states))
        couplings = 0.5 * (couplings + np.transpose(couplings, (1, 0, 3, 2)))
        for site in range(length):
            couplings[site, site] = 0.0
        background = generator.random((length, states))
        background /= background.sum(axis=1, keepdims=True)
        sequence = generator.integers(0, states, size=length)
        eye = np.eye(length, dtype=bool)
        lobe = np.array([0, 0, 1, 1])
        same = lobe[:, None] == lobe[None, :]
        other = ~same & ~eye
        both = np.ones((length, length), dtype=bool)
        np.fill_diagonal(both, False)
        both[0, :] = False
        both[:, 0] = False
        weight = np.zeros((length, length))
        weight[both] = 0.3
        summary = identity_audit.audit_parameters(
            sequence,
            fields,
            couplings,
            background,
            {
                "source": same & ~eye,
                "other": other,
                "other_known": other & both,
                "other_unknown": other & ~both,
                "weight": weight,
            },
        )
        self.assertLessEqual(summary["normalized_energy"], identity_audit.ACCEPTANCE)
        self.assertLessEqual(summary["normalized_geometry"], identity_audit.ACCEPTANCE)
        self.assertLess(summary["q_abs"], 1e-12)

    def test_nonfinite_couplings_raise_before_the_residual(self) -> None:
        fields = np.zeros((2, identity_audit.STATES))
        couplings = np.zeros((2, 2, identity_audit.STATES, identity_audit.STATES))
        couplings[0, 1, 1, 1] = np.nan
        with self.assertRaises(ValueError):
            identity_audit.require_saved_parameters(fields, couplings, length=2)

    def test_a_truncated_coupling_block_raises(self) -> None:
        fields = np.zeros((2, identity_audit.STATES))
        couplings = np.zeros((2, 2, identity_audit.STATES, identity_audit.STATES - 1))
        with self.assertRaises(ValueError):
            identity_audit.require_saved_parameters(fields, couplings, length=2)


if __name__ == "__main__":
    unittest.main()
