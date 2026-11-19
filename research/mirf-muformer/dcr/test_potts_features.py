"""Synthetic checks for feature order, masks, and the sample standard deviation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

import identity_audit
import potts_energy
import potts_features

PREP = Path(__file__).resolve().parents[1] / "prep"
if str(PREP) not in sys.path:
    sys.path.insert(0, str(PREP))
import msa_prepare


def _masks(length: int, excluded: int | None = None) -> dict[str, np.ndarray]:
    eye = np.eye(length, dtype=bool)
    lobe = np.array([0 if site < length // 2 else 1 for site in range(length)])
    same = lobe[:, None] == lobe[None, :]
    other = ~same & ~eye
    both = np.ones((length, length), dtype=bool)
    np.fill_diagonal(both, False)
    if excluded is not None:
        both[excluded, :] = False
        both[:, excluded] = False
    weight = np.zeros((length, length))
    weight[both] = 0.25
    return {
        "source": same & ~eye,
        "other": other,
        "other_known": other & both,
        "other_unknown": other & ~both,
        "weight": weight,
    }


class PottsFeatureTests(unittest.TestCase):
    def test_sample_std_uses_divisor_count_minus_one(self) -> None:
        values = np.array([[1.0], [3.0]])
        self.assertAlmostEqual(float(potts_features.sample_std(values)[0]), float(np.sqrt(2.0)))

    def test_feature_order_keeps_means_and_standard_deviations_apart(self) -> None:
        means = {name: float(index + 1) for index, name in enumerate(potts_features.MEAN_NAMES)}
        deviations = {name: float(index + 10) for index, name in enumerate(potts_features.COMPONENT_NAMES)}
        packed = potts_features.pack_features(means, deviations)
        self.assertEqual(packed.shape, (14,))
        self.assertEqual(packed[0], means["r_mean"])
        self.assertEqual(packed[1], means["delta_e"])
        self.assertEqual(packed[2], deviations["r_mean"])
        self.assertEqual(packed[5], deviations["r_source"])
        self.assertEqual(packed[7], means["r_contact"])
        self.assertEqual(packed[13], deviations["r_unknown"])

    def test_a_signal_peptide_change_is_masked(self) -> None:
        wild = msa_prepare.TARGET_SEQUENCE
        sequence = "A" + wild[1:]
        self.assertIsNone(potts_features.describe_substitution(wild, sequence))

    def test_one_mature_substitution_uses_the_reference_position(self) -> None:
        wild = msa_prepare.TARGET_SEQUENCE
        position = 23
        sequence = wild[:position] + ("A" if wild[position] != "A" else "C") + wild[position + 1 :]
        described = potts_features.describe_substitution(wild, sequence)
        self.assertEqual(described["ref_pos"], 24)
        self.assertEqual(described["mature_index"], 0)
        self.assertEqual(described["mutation"][0], wild[position])

    def test_components_match_mutation_energy_and_drop_geometry_without_coordinates(self) -> None:
        generator = np.random.default_rng(5)
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
        masks = _masks(length, excluded=0)
        parts = potts_features.all_components(sequence, fields, couplings, background, masks)
        state = 1 if int(sequence[0]) != 1 else 2
        self.assertEqual(float(parts["r_contact"][0, state]), 0.0)
        self.assertEqual(float(parts["r_remote"][0, state]), 0.0)
        self.assertAlmostEqual(float(parts["r_unknown"][0, state]), float(parts["r_other"][0, state]))
        for site in range(length):
            for target in range(states):
                delta_e = potts_energy.mutation_energy(sequence, site, target, fields, couplings)
                self.assertAlmostEqual(float(parts["delta_e"][site, target]), delta_e, places=10)


if __name__ == "__main__":
    unittest.main()
