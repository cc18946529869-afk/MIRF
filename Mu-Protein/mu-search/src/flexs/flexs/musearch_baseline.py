"""Corrected μSearch baseline helpers: ranking, history filter, and score budget.

This module does not implement leave-one-out credit and does not import torch.
"""

import argparse
import csv
import json
import os
import time
import zipfile

import numpy as np


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint can only be read by unpickling."""


def select_top_indices(scores, batch_size):
    """Indices of the highest scores, descending, length min(batch_size, n)."""
    scores = np.asarray(scores)
    if int(batch_size) < 0:
        raise ValueError("batch_size must be non-negative")
    count = min(int(batch_size), int(scores.shape[0]))
    if count == 0:
        return np.empty(0, dtype=int)
    order = np.argsort(scores, kind="mergesort")
    return order[::-1][:count]


def partition_measured(sequences, scores, measured, exclude):
    """Drop already measured sequences only when exclude is true."""
    sequences = list(sequences)
    scores = list(scores)
    if len(sequences) != len(scores):
        raise ValueError("sequences and scores must have the same length")
    info = {
        "exclude_measured": bool(exclude),
        "removed": 0,
        "available_before": len(sequences),
        "available_after": len(sequences),
    }
    if not exclude:
        return sequences, scores, info
    measured = set(measured)
    kept_sequences = []
    kept_scores = []
    for sequence, score in zip(sequences, scores):
        if sequence in measured:
            info["removed"] += 1
        else:
            kept_sequences.append(sequence)
            kept_scores.append(score)
    info["available_after"] = len(kept_sequences)
    return kept_sequences, kept_scores, info


def candidate_round_report(sequences, scores, measured, exclude, batch_size):
    """Rank after the optional history filter and record any shortfall it causes."""
    kept_sequences, kept_scores, info = partition_measured(
        sequences, scores, measured, exclude
    )
    ranked = select_top_indices(kept_scores, batch_size)
    chosen_sequences = [kept_sequences[int(index)] for index in ranked]
    chosen_scores = [kept_scores[int(index)] for index in ranked]
    returned = len(chosen_sequences)
    requested = int(batch_size)
    if info["exclude_measured"]:
        filled_before = min(requested, info["available_before"])
        exclusion_shortfall = filled_before - returned
    else:
        exclusion_shortfall = 0
    info.update(
        {
            "requested": requested,
            "returned": returned,
            "unfilled": max(0, requested - returned),
            "exclusion_shortfall": exclusion_shortfall,
        }
    )
    return chosen_sequences, chosen_scores, info


def _sequence_list(sequences):
    if sequences is None:
        return []
    if isinstance(sequences, str):
        return [sequences]
    return [str(sequence) for sequence in sequences]


FORWARD_CHANNELS = (
    "muformer_oracle",
    "cnn_surrogate",
    "other_landscape",
    "other_surrogate",
)

# Each completed scoring call has exactly one purpose. The channel total is
# the sum of these purposes. They are labels, not extra counts.
PURPOSES = (
    "initialization",
    "reward",
    "return_rescore",
    "search_measurement",
    "gradient",
)

# These objects call another scorer. Counting them and the callee would
# record one network evaluation twice. Cache hits are recorded separately.
DELEGATORS = frozenset(
    {
        "LandscapeAsModel",
        "Ensemble",
        "AdaptiveEnsemble",
        "DynaPPOEnsemble",
        "NoisyAbstractModel",
        "ScoreCountingModel",
        "MuformerLandscapeList",
        "CandidateQueryCap",
        "ExactScoreCache",
    }
)


def channel_for(owner):
    """Classify the object that actually evaluates sequences.

    CNN and MuFormer stay on different channels. A Keras model that is not
    the CNN class, and every non-MuFormer landscape, stay on their own
    channels so they are not added into either of those two.
    """
    names = {cls.__name__ for cls in type(owner).__mro__}
    if "MuformerLandscape" in names:
        return "muformer_oracle"
    if "CNN" in names:
        return "cnn_surrogate"
    if "Model" in names:
        return "other_surrogate"
    return "other_landscape"


def _scores_sequences(obj):
    return callable(getattr(obj, "get_fitness", None)) or callable(
        getattr(obj, "_fitness_function", None)
    )


def bind_score_ledger(explorer, landscape=None):
    """Attach one ledger to every scorer a run can reach.

    GWG keeps the raw predictor on the sampler, and Evoplay keeps the raw
    model on its sequence environment. Both objects are the ones that score,
    so the ledger has to be on them, not only on the explorer wrapper.
    """
    ledger = explorer.score_ledger
    pending = [explorer, landscape, getattr(explorer, "model", None)]
    seen = set()
    while pending:
        obj = pending.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        try:
            obj._score_ledger = ledger
        except Exception:
            continue
        for attr in (
            "_model",
            "landscape",
            "predictor",
            "sampler",
            "training_pipeline",
            "seq_env",
            "tf_env",
            "pyenv",
        ):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and child is not obj:
                pending.append(child)
        try:
            model = getattr(obj, "model", None)
        except Exception:
            model = None
        if model is not None and model is not obj and _scores_sequences(model):
            pending.append(model)
        try:
            models = getattr(obj, "models", None)
        except Exception:
            models = None
        if isinstance(models, (list, tuple)):
            pending.extend(list(models))
        try:
            envs = getattr(obj, "envs", None)
        except Exception:
            envs = None
        if isinstance(envs, (list, tuple)):
            pending.extend(list(envs))
        try:
            members = getattr(obj, "muformer_models", None)
        except Exception:
            members = None
        if isinstance(members, (list, tuple)):
            pending.extend(list(members))


def note_forward(owner, sequences, kind="score"):
    """Count one completed scoring call on the owner's own channel.

    Delegators are skipped. A wrapper and the MuFormer it calls therefore
    cannot both record the same forward.
    """
    if owner is None or type(owner).__name__ in DELEGATORS:
        return
    ledger = getattr(owner, "_score_ledger", None)
    if ledger is None:
        return
    if kind not in ("score", "gradient"):
        raise ValueError("kind must be 'score' or 'gradient'")
    purpose = "gradient" if kind == "gradient" else ledger.purpose
    ledger.record(channel_for(owner), sequences, purpose)


def note_cache_hit(owner, sequences):
    """Count sequences served from an existing cache. This is not a forward."""
    if owner is None:
        return
    items = _sequence_list(sequences)
    if not items:
        return
    ledger = getattr(owner, "_score_ledger", None)
    if ledger is None:
        return
    ledger.record_cache_hit(channel_for(owner), items)


class _PurposeScope:
    def __init__(self, ledger, purpose):
        self.ledger = ledger
        self.purpose = purpose
        self.previous = None

    def __enter__(self):
        if self.purpose not in PURPOSES:
            raise ValueError("unknown purpose {0}".format(self.purpose))
        self.previous = self.ledger.purpose
        self.ledger.purpose = self.purpose
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ledger.purpose = self.previous
        return False


class ScoreLedger:
    """One count per completed scoring call.

    `events[channel][purpose]` partitions that channel. `total_by_channel`
    is the sum of the purposes. Adding a purpose to the total counts the
    same call twice, so the summary does not store a second copy.
    CNN and MuFormer stay on different channels and are not added together.
    Cache hits are outside the total.
    """

    CHANNELS = FORWARD_CHANNELS

    def __init__(self):
        self.events = {
            channel: {purpose: 0 for purpose in PURPOSES}
            for channel in self.CHANNELS
        }
        self.duplicates = {channel: 0 for channel in self.CHANNELS}
        self.seen = {channel: set() for channel in self.CHANNELS}
        self.cache_hits = {channel: 0 for channel in self.CHANNELS}
        self._any = set()
        self.candidate_rounds = []
        self.metadata = {}
        self.purpose = "reward"
        self.init_absorbed = False
        self.init_identities_recorded = False
        self.init_identity_mismatch = False

    def _require(self, channel, purpose):
        if channel not in self.events:
            raise KeyError(channel)
        if purpose not in PURPOSES:
            raise ValueError("unknown purpose {0}".format(purpose))

    def _observe_identities(self, channel, sequences):
        for sequence in _sequence_list(sequences):
            if sequence in self.seen[channel]:
                self.duplicates[channel] += 1
            else:
                self.seen[channel].add(sequence)
            self._any.add(sequence)

    def record(self, channel, sequences, purpose):
        self._require(channel, purpose)
        items = _sequence_list(sequences)
        self.events[channel][purpose] += len(items)
        self._observe_identities(channel, items)

    def record_cache_hit(self, channel, sequences):
        if channel not in self.events:
            raise KeyError(channel)
        self.cache_hits[channel] += len(_sequence_list(sequences))

    def marking_purpose(self, purpose):
        return _PurposeScope(self, purpose)

    def marking_return_rescore(self):
        return self.marking_purpose("return_rescore")

    def absorb_initialization(self, landscape):
        """Record pre-run scores once, under the initialization label.

        Constructor scores and the baseline.py probe happen before this
        ledger is attached. Their count is `landscape.cost`. They are not
        passed through `note_forward`, so they are not recorded again.
        """
        if self.init_absorbed:
            raise RuntimeError("initialization was already absorbed")
        channel = channel_for(landscape)
        count = int(getattr(landscape, "cost", 0))
        self.events[channel]["initialization"] += count
        identities = getattr(landscape, "_budget_sequences", None)
        if identities is None:
            self.init_identities_recorded = count == 0
        else:
            items = _sequence_list(identities)
            self.init_identities_recorded = True
            self.init_identity_mismatch = len(items) != count
            if not self.init_identity_mismatch:
                self._observe_identities(channel, items)
        self.init_absorbed = True

    def record_candidate_round(self, report):
        self.candidate_rounds.append(dict(report))

    def total_by_channel(self):
        return {
            channel: sum(self.events[channel].values()) for channel in self.CHANNELS
        }

    def summary(self):
        events = {
            channel: dict(self.events[channel]) for channel in self.CHANNELS
        }
        return {
            "events": events,
            "total_by_channel": self.total_by_channel(),
            "unique_sequences": {
                channel: len(self.seen[channel]) for channel in self.CHANNELS
            },
            "unique_sequences_any_channel": len(self._any),
            "duplicate_calls": dict(self.duplicates),
            "cache_hits": dict(self.cache_hits),
            "init_identities_recorded": self.init_identities_recorded,
            "init_identity_mismatch": self.init_identity_mismatch,
            "candidate_rounds": list(self.candidate_rounds),
            "metadata": dict(self.metadata),
            "accounting": {
                "events": (
                    "each completed scoring call is one event in exactly one "
                    "purpose: initialization, reward, return_rescore, "
                    "search_measurement, or gradient"
                ),
                "total_by_channel": "sum of that channel's purposes; purposes are not added again",
                "channels_are_separate": (
                    "muformer_oracle and cnn_surrogate are not added together"
                ),
                "cache_hits": "cache reads are not scoring calls and are outside total_by_channel",
            },
        }

    def write_summary(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
            handle.write("\n")


class ScoreCountingModel:
    """Delegates cost and scoring. The inner scorer records the forward."""

    def __init__(self, model, ledger, channel=None):
        self._model = model
        self._ledger = ledger
        self._channel = channel

    @property
    def cost(self):
        return self._model.cost

    @cost.setter
    def cost(self, value):
        self._model.cost = value

    def get_fitness(self, sequences):
        return self._model.get_fitness(sequences)

    def __getattr__(self, name):
        return getattr(self._model, name)


class CandidateQueryCap:
    """Stop a propose-phase call that would pass the candidate-query limit.

    Initialization, the round-0 measurement, and return rescoring go through
    the landscape, not this wrapper, so they stay outside the limit.
    """

    def __init__(self, model, limit):
        self._model = model
        self.limit = int(limit)
        self.candidate_calls = 0
        self.name = getattr(model, "name", "CandidateQueryCap")

    @property
    def cost(self):
        return self._model.cost

    @cost.setter
    def cost(self, value):
        self._model.cost = value

    def get_fitness(self, sequences):
        sequences = list(sequences)
        count = len(sequences)
        if self.candidate_calls + count > self.limit:
            raise RuntimeError(
                "candidate queries would become {0}, limit is {1}".format(
                    self.candidate_calls + count, self.limit
                )
            )
        self.candidate_calls += count
        return self._model.get_fitness(sequences)

    def train(self, sequences, labels):
        train = getattr(self._model, "train", None)
        if callable(train):
            return train(sequences, labels)
        return None

    def __getattr__(self, name):
        return getattr(self._model, name)


def _scoring_owner(model):
    """The object that evaluates a cache miss. Wrappers are not scorers."""
    obj = model
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        if type(obj).__name__ in DELEGATORS:
            nxt = getattr(obj, "_model", None)
            if nxt is None:
                nxt = getattr(obj, "landscape", None)
            obj = nxt
            continue
        return obj
    return model


class ExactScoreCache:
    """Return a stored MuFormer score for a repeated sequence.

    A hit does not call the inner scorer and does not increase ``cost``.
    The inner cost stays the number of cache misses. AdaLead's mutation
    and recombination code is not part of this wrapper.
    """

    def __init__(self, model):
        self._model = model
        self._scores = {}
        self._absorbed = False
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.ledger = None
        self.name = getattr(model, "name", "ExactScoreCache")

    @property
    def cost(self):
        return self._model.cost

    @cost.setter
    def cost(self, value):
        self._model.cost = value

    def stats(self):
        return {
            "enabled": True,
            "requests": self.requests,
            "cache_hits": self.hits,
            "muformer_forwards": self.misses,
        }

    def _remember(self, sequence, score):
        key = str(sequence)
        if key not in self._scores:
            self._scores[key] = float(score)

    def _absorb_existing_scores(self):
        if self._absorbed:
            return
        self._absorbed = True
        owner = _scoring_owner(self._model)
        for item in getattr(owner, "query_trace", None) or []:
            self._remember(item["sequence"], item["score"])

    def _publish(self):
        if self.ledger is not None:
            self.ledger.metadata["score_cache"] = self.stats()

    def get_fitness(self, sequences):
        self._absorb_existing_scores()
        items = [str(sequence) for sequence in list(sequences)]
        self.requests += len(items)
        missing = []
        seen_missing = set()
        for sequence in items:
            if sequence not in self._scores and sequence not in seen_missing:
                missing.append(sequence)
                seen_missing.add(sequence)
        if missing:
            scored = np.asarray(self._model.get_fitness(missing), dtype=float).reshape(-1)
            if int(scored.shape[0]) != len(missing):
                raise RuntimeError(
                    "cache miss returned {0} scores for {1} sequences".format(
                        int(scored.shape[0]), len(missing)
                    )
                )
            for sequence, score in zip(missing, scored):
                self._remember(sequence, score)
        forwarded = set()
        hit_sequences = []
        for sequence in items:
            if sequence in seen_missing and sequence not in forwarded:
                forwarded.add(sequence)
                continue
            hit_sequences.append(sequence)
        self.hits += len(hit_sequences)
        self.misses += len(missing)
        if hit_sequences:
            note_cache_hit(_scoring_owner(self._model), hit_sequences)
        self._publish()
        return np.asarray([self._scores[sequence] for sequence in items], dtype=float)

    def __getattr__(self, name):
        return getattr(self._model, name)


def floor_rollout_schedule(queries, horizon, update_frequency):
    """Report the original PPO split. It does not retarget the update count.

    ``episodes = queries // update_frequency`` and
    ``total_timesteps = queries * horizon``. The existing learn loop then
    drops a rollout that would pass ``total_timesteps``.
    """
    if queries < 1 or horizon < 1 or update_frequency < 1:
        raise ValueError("queries, horizon, and update_frequency must be positive")
    episodes = queries // update_frequency
    if episodes < 1:
        raise ValueError("update_frequency leaves no complete trajectory")
    n_steps = episodes * horizon
    total_timesteps = queries * horizon
    collected = 0
    rollouts = 0
    while collected < total_timesteps:
        if collected + n_steps > total_timesteps:
            break
        collected += n_steps
        rollouts += 1
    return {
        "episodes_per_update": episodes,
        "n_steps": n_steps,
        "total_timesteps": total_timesteps,
        "rollouts": rollouts,
        "timesteps_collected": collected,
        "terminal_scores": rollouts * episodes,
        "requested_update_frequency": update_frequency,
    }


def exact_terminal_schedule(queries, horizon, update_frequency):
    """Choose a rollout length the existing PPO loop can finish exactly.

    That loop starts a rollout only when the whole rollout still fits in
    ``total_timesteps``. ``queries // update_frequency`` is not always a
    divisor of ``queries``, so the floor split stops short of the budget.
    The nearest divisor keeps the update cadence as close as that loop allows.
    """
    if queries < 1 or horizon < 1 or update_frequency < 1:
        raise ValueError("queries, horizon, and update_frequency must be positive")
    target = max(queries // update_frequency, 1)
    divisors = [divisor for divisor in range(1, queries + 1) if queries % divisor == 0]
    episodes = min(divisors, key=lambda divisor: (abs(divisor - target), divisor))
    n_steps = episodes * horizon
    total_timesteps = queries * horizon
    if total_timesteps % n_steps != 0:
        raise ValueError("rollout length does not divide the timestep budget")
    updates = total_timesteps // n_steps
    terminal_scores = updates * episodes
    if terminal_scores != queries:
        raise ValueError("terminal scores {0} != {1}".format(terminal_scores, queries))
    return {
        "episodes_per_update": episodes,
        "n_steps": n_steps,
        "total_timesteps": total_timesteps,
        "updates": updates,
        "terminal_scores": terminal_scores,
        "requested_episodes_per_update": target,
        "requested_update_frequency": update_frequency,
    }


def clip_to_remaining(items, spent, limit):
    """Keep the prefix that still fits in the query budget."""
    remaining = limit - spent
    if remaining <= 0:
        return items[:0]
    if len(items) <= remaining:
        return items
    return items[:remaining]


def append_query_trace(landscape, sequence, score):
    """Record one MuFormer score. Initialization is any call before a ledger."""
    if not hasattr(landscape, "query_trace"):
        landscape.query_trace = []
    if getattr(landscape, "_query_trace_t0", None) is None:
        landscape._query_trace_t0 = time.perf_counter()
    ledger = getattr(landscape, "_score_ledger", None)
    purpose = "initialization" if ledger is None else ledger.purpose
    landscape.query_trace.append(
        {
            "purpose": purpose,
            "sequence": str(sequence),
            "score": float(score),
            "t_sec": time.perf_counter() - landscape._query_trace_t0,
            "model_input_sequences": 2,
        }
    )


_QUERY_ROLES = {
    "initialization": "initialization",
    "search_measurement": "start",
    "reward": "candidate",
    "return_rescore": "return_rescore",
    "gradient": "gradient",
}


def query_trace_rows(trace):
    """Add the candidate-query index and the running best candidate score."""
    rows = []
    candidate_index = 0
    best = None
    for call_index, item in enumerate(trace, start=1):
        purpose = item["purpose"]
        role = _QUERY_ROLES.get(purpose, purpose)
        row = {
            "call_index": call_index,
            "purpose": purpose,
            "role": role,
            "sequence": item["sequence"],
            "score": item["score"],
            "t_sec": item["t_sec"],
            "model_input_sequences": item["model_input_sequences"],
            "candidate_index": "",
            "candidate_best_so_far": "",
        }
        if role == "candidate":
            candidate_index += 1
            best = item["score"] if best is None else max(best, item["score"])
            row["candidate_index"] = candidate_index
            row["candidate_best_so_far"] = best
        rows.append(row)
    return rows


def write_query_trace(landscape, path):
    trace = getattr(landscape, "query_trace", None)
    if not trace:
        return None
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    rows = query_trace_rows(trace)
    fieldnames = [
        "call_index",
        "purpose",
        "role",
        "sequence",
        "score",
        "t_sec",
        "model_input_sequences",
        "candidate_index",
        "candidate_best_so_far",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def begin_score_budget(explorer, landscape):
    bind_score_ledger(explorer, landscape)
    explorer.score_ledger.query_landscape = landscape
    explorer.score_ledger.absorb_initialization(landscape)


def finish_score_budget(explorer):
    budget = explorer.score_ledger.summary()
    print(
        "score_budget total_by_channel={total} events={events} "
        "unique={unique} duplicate={duplicate} cache_hits={cache_hits} "
        "score_cache={score_cache}".format(
            total=budget["total_by_channel"],
            events=budget["events"],
            unique=budget["unique_sequences"],
            duplicate=budget["duplicate_calls"],
            cache_hits=budget["cache_hits"],
            score_cache=budget["metadata"].get("score_cache"),
        )
    )
    if getattr(explorer, "log_file", None) is not None:
        explorer.score_ledger.write_summary(budget_path_for_log(explorer.log_file))
        landscape = getattr(explorer.score_ledger, "query_landscape", None)
        if landscape is not None:
            write_query_trace(landscape, query_path_for_log(explorer.log_file))
    return budget


def check_budget_summary(summary):
    """Reject a ledger whose purposes can be added to the channel total."""
    overlapping = (
        "total_score_calls",
        "forwards",
        "initialization_calls",
        "return_rescore_calls",
        "reward_model",
    )
    present = [key for key in overlapping if key in summary]
    if present:
        raise ValueError(
            "budget repeats calls under {0}".format(", ".join(present))
        )
    if "events" not in summary or "total_by_channel" not in summary:
        raise ValueError("budget is missing events or total_by_channel")
    for channel, purposes in summary["events"].items():
        if set(purposes) != set(PURPOSES):
            raise ValueError("channel {0} does not partition the purposes".format(channel))
        purpose_sum = sum(purposes.values())
        if summary["total_by_channel"][channel] != purpose_sum:
            raise ValueError(
                "channel {0} total {1} != purpose sum {2}".format(
                    channel, summary["total_by_channel"][channel], purpose_sum
                )
            )
    for count in summary["cache_hits"].values():
        if count < 0:
            raise ValueError("negative cache hits")
    return True


def inspect_torch_checkpoint(path):
    """Read a checkpoint header. Does not unpickle and does not call torch.load."""
    path = os.fspath(path)
    info = {
        "path": path,
        "exists": os.path.isfile(path),
        "format": "missing",
        "contains_pickle": None,
    }
    if not info["exists"]:
        return info
    info["bytes"] = os.path.getsize(path)
    with open(path, "rb") as handle:
        magic = handle.read(8)
    info["magic_hex"] = magic.hex()
    if magic[:2] == b"PK":
        info["format"] = "zip"
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        info["zip_entry_count"] = len(names)
        info["zip_entries_head"] = names[:20]
        info["contains_pickle"] = any(
            name.endswith("data.pkl") or name.endswith(".pkl") for name in names
        )
    elif magic[:1] == b"\x80":
        info["format"] = "legacy_pickle"
        info["contains_pickle"] = True
    else:
        info["format"] = "unknown"
    return info


def refuse_unsafe_checkpoints(*paths):
    """Stop before any pickle-based loader. Always raises."""
    reports = [inspect_torch_checkpoint(path) for path in paths]
    details = []
    for report in reports:
        details.append(
            "{path}: format={format}, bytes={bytes}, contains_pickle={pickle}".format(
                path=report["path"],
                format=report["format"],
                bytes=report.get("bytes"),
                pickle=report.get("contains_pickle"),
            )
        )
    raise UnsafeCheckpointError(
        "Refusing to load MuFormer weights. torch.load unpickles the checkpoint, "
        "and these files are not opened that way. " + " | ".join(details)
    )


def budget_path_for_log(log_file):
    root, _ext = os.path.splitext(log_file)
    return root + ".budget.json"


def query_path_for_log(log_file):
    root, _ext = os.path.splitext(log_file)
    return root + ".queries.csv"


def worked_budget_example():
    """One MuFormer forward is one event. Purposes partition that channel.

    Pre-run, with no ledger attached:
      get_fitness(["WT"])
      get_fitness(["E1", "E2"])
      get_fitness(["PROBE"])          # baseline.py line 51
    cost is 4. Those four calls are not noted yet.

    After Explorer.run attaches the ledger and absorbs cost:
      initialization = 4
      search_measurement of "WT" = 1
      LandscapeAsModel.get_fitness(["R1", "R2"]) = 2 reward events
      return measurement of "R1" = 1
      a separate CNN get_fitness(["C1"]) = 1 reward event on cnn_surrogate

    MuFormer total = 4 + 1 + 2 + 1 = 8.
    The wrapper and LandscapeAsModel add nothing on top of the two reward calls.
    CNN total = 1. The two channel totals are not added.
    """

    class MuformerLandscape:
        def __init__(self):
            self.cost = 0
            self._budget_sequences = []
            self.name = "MuformerLandscape"

        def predict(self, sequence):
            note_forward(self, [sequence])
            return 0.0

        def _fitness_function(self, sequences):
            return [self.predict(sequence) for sequence in sequences]

        def get_fitness(self, sequences):
            scores = [self.predict(sequence) for sequence in sequences]
            self.cost += len(list(sequences))
            self._budget_sequences.extend(str(sequence) for sequence in sequences)
            return scores

    class LandscapeAsModel:
        def __init__(self, landscape):
            self.landscape = landscape
            self.cost = 0
            self.name = "LandscapeAsModel"

        def get_fitness(self, sequences):
            self.cost += len(list(sequences))
            scores = self.landscape._fitness_function(sequences)
            note_forward(self, sequences)
            return scores

    class CNN:
        def __init__(self):
            self.cost = 0
            self.name = "CNN"

        def get_fitness(self, sequences):
            self.cost += len(list(sequences))
            note_forward(self, sequences)
            return [1.0 for _ in sequences]

        def gradient_function(self, sequences):
            note_forward(self, sequences, kind="gradient")

    oracle = MuformerLandscape()
    oracle.get_fitness(["WT"])
    oracle.get_fitness(["E1", "E2"])
    oracle.get_fitness(["PROBE"])
    cnn = CNN()
    reward_model = LandscapeAsModel(oracle)
    ledger = ScoreLedger()
    wrapped = ScoreCountingModel(reward_model, ledger)

    class Sampler:
        def __init__(self):
            self.predictor = cnn

    class Explorer:
        def __init__(self):
            self.score_ledger = ledger
            self.model = wrapped
            self.sampler = Sampler()
            self.log_file = None

    begin_score_budget(Explorer(), oracle)
    with ledger.marking_purpose("search_measurement"):
        oracle.get_fitness(["WT"])
    wrapped.get_fitness(["R1", "R2"])
    with ledger.marking_return_rescore():
        oracle.get_fitness(["R1"])
    cnn.gradient_function(["C1"])
    cnn.get_fitness(["C1"])
    note_cache_hit(cnn, ["C1"])
    summary = ledger.summary()
    check_budget_summary(summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Account or inspect μSearch score files.")
    parser.add_argument("--summary", help="Print an existing budget JSON file.")
    parser.add_argument(
        "--inspect-checkpoint",
        action="append",
        default=[],
        help="Read one checkpoint header. May be repeated. Does not unpickle.",
    )
    args = parser.parse_args(argv)
    if args.summary:
        with open(args.summary, encoding="utf-8") as handle:
            summary = json.load(handle)
        check_budget_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    for path in args.inspect_checkpoint:
        print(json.dumps(inspect_torch_checkpoint(path), indent=2, sort_keys=True))
    if not args.summary and not args.inspect_checkpoint:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
