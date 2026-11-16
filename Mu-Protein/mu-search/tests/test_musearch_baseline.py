"""Unit tests for the corrected μSearch baseline. They do not train a policy."""

import importlib.util
import json
import os
import tempfile
import zipfile

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load(name, relative_path):
    path = os.path.join(ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = _load(
    "musearch_baseline",
    "src/flexs/flexs/musearch_baseline.py",
)
mixture = _load(
    "mixture",
    "src/flexs/flexs/baselines/explorers/stable_baselines3/"
    "stable_baselines3/common/mixture.py",
)


def test_marginal_matches_sampler_expectation():
    probability = 0.7
    categories = 20
    got = mixture.mixed_action_probability(probability, categories, "marginal")
    expected = 0.75 * probability + 0.25 / categories
    assert abs(got - expected) < 1e-12


def test_legacy_formula_is_separate():
    probability = 0.7
    categories = 20
    legacy = mixture.mixed_action_probability(probability, categories, "legacy")
    marginal = mixture.mixed_action_probability(probability, categories, "marginal")
    assert abs(legacy - (0.50 * probability + 0.50 / categories)) < 1e-12
    assert abs(legacy - marginal) > 1e-6


def test_dirichlet_samples_have_marginal_frequencies():
    rng = np.random.default_rng(0)
    policy = np.full(8, 0.02)
    policy[0] = 1.0 - 0.02 * 7
    draws = 40000
    counts = np.zeros(8)
    for _ in range(draws):
        mixed = mixture.dirichlet_mixture_probabilities(policy, rng)
        counts[int(rng.choice(8, p=mixed))] += 1
    frequency = counts / draws
    expected = np.array(
        [mixture.mixed_action_probability(p, 8, "marginal") for p in policy]
    )
    assert np.max(np.abs(frequency - expected)) < 0.015


def test_topk_length_and_order():
    scores = np.array([1.0, 4.0, 3.0, 2.0, 0.5])
    chosen = baseline.select_top_indices(scores, 3)
    assert len(chosen) == 3
    assert list(scores[chosen]) == [4.0, 3.0, 2.0]
    assert len(baseline.select_top_indices(scores, 100)) == 5
    assert len(baseline.select_top_indices(scores, 0)) == 0
    one = baseline.select_top_indices(scores, 1)
    assert len(one) == 1
    assert scores[one[0]] == 4.0
    old = np.argsort(scores)[: -3 : -1]
    assert len(old) == 2


def test_history_filter_is_off_by_default_and_records_shortfall():
    sequences = ["AAA", "BBB", "CCC", "DDD"]
    scores = [1.0, 4.0, 3.0, 2.0]
    kept, kept_scores, report = baseline.candidate_round_report(
        sequences, scores, measured=["BBB"], exclude=False, batch_size=2
    )
    assert kept == ["BBB", "CCC"]
    assert kept_scores == [4.0, 3.0]
    assert report["removed"] == 0
    assert report["exclusion_shortfall"] == 0

    kept, kept_scores, report = baseline.candidate_round_report(
        sequences, scores, measured=["BBB", "CCC"], exclude=True, batch_size=3
    )
    assert kept == ["DDD", "AAA"]
    assert report["removed"] == 2
    assert report["returned"] == 2
    assert report["exclusion_shortfall"] == 1


def test_worked_budget_counts_each_forward_once():
    summary = baseline.worked_budget_example()
    muformer = summary["events"]["muformer_oracle"]
    cnn = summary["events"]["cnn_surrogate"]
    assert muformer == {
        "initialization": 4,
        "reward": 2,
        "search_measurement": 1,
        "return_rescore": 1,
        "gradient": 0,
    }
    assert summary["total_by_channel"]["muformer_oracle"] == 4 + 1 + 2 + 1
    assert cnn["reward"] == 1
    assert cnn["gradient"] == 1
    assert summary["total_by_channel"]["cnn_surrogate"] == 2
    assert summary["cache_hits"]["cnn_surrogate"] == 1
    assert summary["duplicate_calls"]["muformer_oracle"] == 2
    assert summary["unique_sequences"]["muformer_oracle"] == 6
    assert (
        summary["unique_sequences"]["muformer_oracle"]
        + summary["duplicate_calls"]["muformer_oracle"]
        == summary["total_by_channel"]["muformer_oracle"]
    )
    assert "forwards" not in summary
    assert "initialization_calls" not in summary
    assert "return_rescore_calls" not in summary
    assert "reward_model" not in summary
    try:
        baseline.check_budget_summary({"forwards": {}, **summary})
    except ValueError as error:
        assert "repeats" in str(error)
    else:
        raise AssertionError("overlapping budget fields were accepted")
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "run.budget.json")
        ledger = baseline.ScoreLedger()
        # The example already checked its own summary. Rewrite that summary
        # through the public checker used by the command-line tool.
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle)
        stdout = sys_stdout()
        try:
            assert baseline.main(["--summary", path]) == 0
        finally:
            restore_stdout(stdout)
        del ledger


def sys_stdout():
    import io
    import sys

    captured = io.StringIO()
    sys._baseline_stdout = sys.stdout
    sys.stdout = captured
    return captured


def restore_stdout(_captured):
    import sys

    sys.stdout = sys._baseline_stdout


def test_counting_model_delegates_cost_without_a_second_count():
    class CNN:
        def __init__(self):
            self.cost = 7
            self.name = "fake"

        def get_fitness(self, sequences):
            self.cost += len(sequences)
            baseline.note_forward(self, sequences)
            return [float(len(sequence)) for sequence in sequences]

        def train(self, sequences, labels):
            return len(sequences)

    ledger = baseline.ScoreLedger()
    cnn = CNN()
    cnn._score_ledger = ledger
    wrapped = baseline.ScoreCountingModel(cnn, ledger)
    wrapped.cost = 0
    assert wrapped.get_fitness(["AC", "DE"]) == [2.0, 2.0]
    assert wrapped.cost == 2
    assert ledger.events["cnn_surrogate"]["reward"] == 2
    assert ledger.total_by_channel()["cnn_surrogate"] == 2
    assert wrapped.name == "fake"
    assert wrapped.train(["AC"], [1.0]) == 1


def test_checkpoint_header_does_not_unpickle():
    with tempfile.TemporaryDirectory() as directory:
        zip_path = os.path.join(directory, "weights.pt")
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("archive/data.pkl", b"not-a-real-pickle")
            archive.writestr("archive/data/0", b"tensor")
        info = baseline.inspect_torch_checkpoint(zip_path)
        assert info["format"] == "zip"
        assert info["contains_pickle"] is True
        legacy = os.path.join(directory, "legacy.pt")
        with open(legacy, "wb") as handle:
            handle.write(b"\x80\x04\x95\x00")
        assert baseline.inspect_torch_checkpoint(legacy)["format"] == "legacy_pickle"
        try:
            baseline.refuse_unsafe_checkpoints(zip_path)
        except baseline.UnsafeCheckpointError as error:
            message = str(error)
            assert "not-a-real-pickle" not in message
            assert "contains_pickle=True" in message
        else:
            raise AssertionError("unsafe checkpoint was accepted")


def test_score_hooks_are_on_the_real_call_paths():
    files = {
        "src/flexs/flexs/landscape.py": ("note_forward",),
        "src/flexs/flexs/explorer.py": (
            "begin_score_budget",
            "search_measurement",
            "marking_return_rescore",
            "finish_score_budget",
        ),
        "src/flexs/flexs/baselines/explorers/ggs.py": (
            "begin_score_budget",
            "search_measurement",
            "marking_return_rescore",
            "finish_score_budget",
        ),
        "src/flexs/flexs/landscapes/muformer.py": (
            "note_forward",
            "append_query_trace",
        ),
        "src/flexs/flexs/baselines/models/cnn.py": ('kind="gradient"',),
        "src/flexs/flexs/baselines/models/noisy_abstract_model.py": ("note_cache_hit",),
    }
    for relative, needles in files.items():
        with open(os.path.join(ROOT, relative), encoding="utf-8") as handle:
            source = handle.read()
        for needle in needles:
            assert needle in source, relative
    with open(
        os.path.join(ROOT, "src/flexs/flexs/musearch_baseline.py"), encoding="utf-8"
    ) as handle:
        source = handle.read()
    wrapper = source.split("class ScoreCountingModel", 1)[1].split("def begin_score_budget", 1)[0]
    assert "ledger.add" not in wrapper
    assert "total_score_calls" not in source.split("def check_budget_summary", 1)[0]


def test_floor_schedule_keeps_three_rollouts():
    schedule = baseline.floor_rollout_schedule(4998, 5, 3)
    assert schedule["episodes_per_update"] == 1666
    assert schedule["n_steps"] == 8330
    assert schedule["total_timesteps"] == 24990
    assert schedule["rollouts"] == 3
    assert schedule["timesteps_collected"] == 24990
    assert schedule["terminal_scores"] == 4998
    unchanged = baseline.floor_rollout_schedule(5000, 5, 3)
    assert unchanged["episodes_per_update"] == 1666
    assert unchanged["rollouts"] == 3
    assert unchanged["terminal_scores"] == 4998


def test_exact_terminal_schedule_hits_5000():
    schedule = baseline.exact_terminal_schedule(5000, 5, 3)
    assert schedule["requested_episodes_per_update"] == 1666
    assert schedule["episodes_per_update"] == 1250
    assert schedule["updates"] == 4
    assert schedule["n_steps"] == 6250
    assert schedule["total_timesteps"] == 25000
    assert schedule["terminal_scores"] == 5000


def test_clip_to_remaining_keeps_a_short_tail():
    assert list(baseline.clip_to_remaining(list(range(20)), 0, 10)) == list(range(10))
    assert list(baseline.clip_to_remaining(list(range(20)), 4985, 5000)) == list(
        range(15)
    )
    assert list(baseline.clip_to_remaining(["a", "b"], 5000, 5000)) == []


def test_query_trace_best_ignores_initialization():
    class Landscape:
        pass

    landscape = Landscape()
    baseline.append_query_trace(landscape, "INIT", -1.0)
    ledger = baseline.ScoreLedger()
    landscape._score_ledger = ledger
    with ledger.marking_purpose("search_measurement"):
        baseline.append_query_trace(landscape, "WT", -0.07)
    baseline.append_query_trace(landscape, "A", -0.05)
    baseline.append_query_trace(landscape, "B", -0.04)
    rows = baseline.query_trace_rows(landscape.query_trace)
    assert rows[0]["role"] == "initialization"
    assert rows[0]["candidate_best_so_far"] == ""
    assert rows[1]["role"] == "start"
    assert rows[1]["candidate_best_so_far"] == ""
    assert rows[2]["candidate_index"] == 1
    assert rows[2]["candidate_best_so_far"] == -0.05
    assert rows[3]["candidate_best_so_far"] == -0.04


def test_candidate_cap_stops_at_the_limit():
    class Model:
        def __init__(self):
            self.cost = 0
            self.name = "Model"

        def get_fitness(self, sequences):
            self.cost += len(sequences)
            return [0.0 for _ in sequences]

    cap = baseline.CandidateQueryCap(Model(), 20)
    cap.get_fitness(["a"] * 20)
    try:
        cap.get_fitness(["b"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("cap allowed query 21")
    assert cap.candidate_calls == 20


def test_budget_changes_do_not_touch_gwg_or_the_learn_loop():
    ggs = os.path.join(ROOT, "src/flexs/flexs/baselines/explorers/ggs.py")
    learn = os.path.join(
        ROOT,
        "src/flexs/flexs/baselines/explorers/stable_baselines3/"
        "stable_baselines3/common/on_policy_algorithm.py",
    )
    with open(ggs, encoding="utf-8") as handle:
        ggs_source = handle.read()
    with open(learn, encoding="utf-8") as handle:
        learn_source = handle.read()
    assert "clip_to_remaining" not in ggs_source
    assert "exact_terminal_schedule" not in ggs_source
    assert "if self.num_timesteps + self.n_steps > total_timesteps" in learn_source
    adalead = os.path.join(ROOT, "src/flexs/flexs/baselines/explorers/adalead.py")
    ppo = os.path.join(ROOT, "src/flexs/flexs/baselines/explorers/dirichlet_ppo.py")
    with open(adalead, encoding="utf-8") as handle:
        assert "clip_to_remaining" in handle.read()
    with open(ppo, encoding="utf-8") as handle:
        source = handle.read()
    assert "floor_rollout_schedule" in source
    assert "exact_terminal_schedule" not in source
    assert "model_queries_per_batch // update_frequency" not in source


def test_distribution_source_uses_the_shared_mixture():
    path = os.path.join(
        ROOT,
        "src/flexs/flexs/baselines/explorers/stable_baselines3/"
        "stable_baselines3/common/distributions.py",
    )
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert "dirichlet_mixture_probabilities" in source
    assert "POLICY_MIX_WEIGHT" in source
    assert "LEGACY_LOG_PROB_WEIGHT" in source
    assert "alpha = 0.75" not in source
    assert "alpha = 0.50" not in source
    assert ".cuda()" not in source


def test_horizon_floor_keeps_three_rollouts_and_4998_scores():
    for horizon in (5, 7, 10):
        schedule = baseline.floor_rollout_schedule(4998, horizon, 3)
        assert schedule["episodes_per_update"] == 1666
        assert schedule["n_steps"] == 1666 * horizon
        assert schedule["total_timesteps"] == 4998 * horizon
        assert schedule["rollouts"] == 3
        assert schedule["timesteps_collected"] == 4998 * horizon
        assert schedule["terminal_scores"] == 4998


class _CacheModel:
    def __init__(self):
        self.cost = 0
        self.calls = []
        self.query_trace = [
            {"sequence": "AAA", "score": 1.25, "purpose": "search_measurement"}
        ]
        self._score_ledger = baseline.ScoreLedger()
        self.name = "CacheModel"

    def get_fitness(self, sequences):
        sequences = list(sequences)
        self.calls.append(sequences)
        self.cost += len(sequences)
        return [float(len(sequence)) for sequence in sequences]


def test_exact_cache_does_not_rescore_a_known_sequence():
    model = _CacheModel()
    cache = baseline.ExactScoreCache(model)
    first = cache.get_fitness(["AAA", "BBB", "AAA"])
    assert list(first) == [1.25, 3.0, 1.25]
    assert model.calls == [["BBB"]]
    assert model.cost == 1
    assert cache.requests == 3
    assert cache.hits == 2
    assert cache.misses == 1
    second = cache.get_fitness(["BBB", "CCC", "BBB"])
    assert list(second) == [3.0, 3.0, 3.0]
    assert model.cost == 2
    assert cache.requests == 6
    assert cache.hits == 4
    assert cache.misses == 2
    assert model._score_ledger.total_by_channel()["other_landscape"] == 0
    assert model._score_ledger.cache_hits["other_landscape"] == 4
    assert cache.stats()["muformer_forwards"] == model.cost
    assert [row["outcome"] for row in cache.request_rows] == [
        "hit", "miss", "hit", "hit", "miss", "hit"
    ]


def test_cache_does_not_inherit_initialization():
    model = _CacheModel()
    model.query_trace = [
        {"sequence": "ESBL_A", "score": 9.0, "purpose": "initialization"},
        {"sequence": "WT", "score": 1.0, "purpose": "initialization"},
        {"sequence": "WT", "score": 1.5, "purpose": "search_measurement"},
    ]

    def get_fitness(sequences):
        sequences = list(sequences)
        model.calls.append(sequences)
        model.cost += len(sequences)
        baseline.note_forward(model, sequences)
        return [4.0 for _ in sequences]

    model.get_fitness = get_fitness
    cache = baseline.ExactScoreCache(model)
    esbl = cache.get_fitness(["ESBL_A"])
    assert list(esbl) == [4.0]
    assert model.calls == [["ESBL_A"]]
    wild_type = cache.get_fitness(["WT"])
    assert list(wild_type) == [1.5]
    assert model.calls == [["ESBL_A"]]
    before = model._score_ledger.total_by_channel()["other_landscape"]
    first = cache.get_fitness(["C"])
    assert list(first) == [4.0]
    after_miss = model._score_ledger.total_by_channel()["other_landscape"]
    assert after_miss == before + 1
    second = cache.get_fitness(["C"])
    assert list(second) == [4.0]
    assert model._score_ledger.total_by_channel()["other_landscape"] == after_miss
    assert model.cost == 2
    assert [row["outcome"] for row in cache.request_rows] == [
        "miss", "hit", "miss", "hit"
    ]
    assert [row["sequence"] for row in cache.request_rows] == [
        "ESBL_A", "WT", "C", "C"
    ]


def _run_all():
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:
                failures += 1
                print("FAIL", name, error)
            else:
                print("ok", name)
    return failures


if __name__ == "__main__":
    raise SystemExit(_run_all())
