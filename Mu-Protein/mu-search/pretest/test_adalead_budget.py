"""AdaLead's last scoring batch must stay inside the remaining query budget."""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "flexs"))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from flexs.baselines.explorers.adalead import Adalead
from flexs.utils.sequence_utils import AAS


class CountingModel:
    def __init__(self):
        self.cost = 0
        self.batches = []
        self.calls = []
        self.name = "CountingModel"

    def get_fitness(self, sequences):
        sequences = [str(sequence) for sequence in list(sequences)]
        self.batches.append(len(sequences))
        self.calls.append(sequences)
        self.cost += len(sequences)
        return np.zeros(len(sequences))

    def train(self, sequences, labels):
        return None


def propose(limit):
    model = CountingModel()
    explorer = Adalead(
        model,
        rounds=1,
        sequences_batch_size=20,
        model_queries_per_batch=limit,
        starting_sequence="A" * 12,
        alphabet=AAS,
        mu=1,
        recomb_rate=0,
        threshold=0.05,
        rho=0,
        eval_batch_size=20,
    )
    measured = pd.DataFrame(
        {
            "sequence": ["A" * 12],
            "true_score": [-0.1],
            "model_score": [np.nan],
            "round": [0],
        }
    )
    explorer.propose_sequences(measured)
    return model.batches, model.cost


def propose_cached(limit):
    from flexs.musearch_baseline import ExactScoreCache

    inner = CountingModel()
    inner.query_trace = [
        {"sequence": "A" * 12, "score": -0.1, "purpose": "search_measurement"}
    ]
    cache = ExactScoreCache(inner)
    explorer = Adalead(
        cache,
        rounds=1,
        sequences_batch_size=20,
        model_queries_per_batch=limit,
        starting_sequence="A" * 12,
        alphabet=AAS,
        mu=1,
        recomb_rate=0,
        threshold=0.05,
        rho=0,
        eval_batch_size=20,
    )
    measured = pd.DataFrame(
        {
            "sequence": ["A" * 12],
            "true_score": [-0.1],
            "model_score": [np.nan],
            "round": [0],
        }
    )
    explorer.propose_sequences(measured)
    forwarded = [sequence for batch in inner.calls for sequence in batch]
    return inner.batches, inner.cost, cache, forwarded


def main():
    batches, cost = propose(25)
    print("limit 25 batches", batches, "cost", cost)
    if cost != 25 or batches != [20, 5]:
        raise SystemExit("last batch was not the 5 remaining queries")
    batches, cost = propose(10)
    print("limit 10 batches", batches, "cost", cost)
    if cost != 10 or batches != [10]:
        raise SystemExit("first batch was not cut to the remaining 10")
    if any(size > 20 for size in batches):
        raise SystemExit("a batch exceeded eval_batch_size")
    cached_batches, cached_cost, cache, forwarded = propose_cached(25)
    print(
        "cached limit 25 batches",
        cached_batches,
        "cost",
        cached_cost,
        "requests",
        cache.requests,
        "hits",
        cache.hits,
    )
    if cached_cost != 25 or cache.misses != 25:
        raise SystemExit("cache misses did not fill the oracle budget")
    if cache.requests <= cache.misses or cache.hits != cache.requests - cache.misses:
        raise SystemExit("cache hits were not separated from oracle forwards")
    if sum(cached_batches) != 25 or any(size > 20 for size in cached_batches):
        raise SystemExit("cached batches changed the request size limit")
    if "A" * 12 in forwarded:
        raise SystemExit("a known sequence was forwarded to the oracle")
    print("adalead_budget_ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
