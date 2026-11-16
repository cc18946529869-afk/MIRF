"""AdaLead's returned batch uses select_top_indices, including batch size 1."""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "flexs"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

import flexs.baselines.explorers.adalead as adalead_module
from flexs.baselines.explorers.adalead import Adalead


class _RankModel:
    def __init__(self):
        self.cost = 0
        self.scores = {}
        self.next_score = 0.0

    def get_fitness(self, sequences):
        values = []
        for sequence in sequences:
            sequence = str(sequence)
            if sequence not in self.scores:
                self.scores[sequence] = self.next_score
                self.next_score += 1.0
            values.append(self.scores[sequence])
        self.cost += len(sequences)
        return np.asarray(values, dtype=float)


def _child_factory():
    counter = {"n": 0}

    def generate(_sequence, _mu, _alphabet):
        counter["n"] += 1
        return "CHILD{0:04d}".format(counter["n"])

    return generate


def _propose(batch_size, child_count):
    adalead_module.s_utils.generate_random_mutant = _child_factory()
    model = _RankModel()
    explorer = Adalead(
        model,
        rounds=1,
        sequences_batch_size=batch_size,
        model_queries_per_batch=child_count + min(batch_size, 20),
        starting_sequence="WTSEQ",
        alphabet="ACDEFGHIKLMNPQRSTVWY",
        mu=1,
        recomb_rate=0,
        threshold=0.05,
        rho=0,
        eval_batch_size=20,
    )
    measured = pd.DataFrame({"sequence": ["WTSEQ"], "true_score": [0.0]})
    sequences, scores = explorer.propose_sequences(measured)
    return sequences, np.asarray(scores, dtype=float), model


def test_return_counts_and_descending_order():
    cases = (
        (100, 200, 100),
        (100, 100, 100),
        (100, 20, 20),
        (1, 1, 1),
    )
    for batch_size, child_count, expected in cases:
        sequences, scores, model = _propose(batch_size, child_count)
        if len(sequences) != expected:
            raise SystemExit(
                "batch {0}, children {1}: returned {2}".format(
                    batch_size, child_count, len(sequences)
                )
            )
        if len(scores) != expected:
            raise SystemExit("score length {0}".format(len(scores)))
        if expected > 1 and not np.all(scores[:-1] >= scores[1:]):
            raise SystemExit("scores are not descending: {0}".format(scores[:5]))
        child_scores = [
            score for sequence, score in model.scores.items() if sequence != "WTSEQ"
        ]
        best = sorted(child_scores, reverse=True)[:expected]
        if list(scores) != best:
            raise SystemExit(
                "returned scores {0} are not the top {1}".format(list(scores), best)
            )
        print(
            "adalead_topk batch={0} children={1} returned={2} top={3} bottom={4}".format(
                batch_size, child_count, len(sequences), scores[0], scores[-1]
            )
        )
    print("adalead_topk_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(test_return_counts_and_descending_order())
