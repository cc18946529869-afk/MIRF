"""Instantiate the real PPO action distribution and check its probabilities.

This file does not stub gym. If the interpreter cannot import gym, torch, or
the project distribution class, it exits 2 and names the missing import.
A missing dependency is not a passed check.
"""

import json
import math
import os
import sys
import traceback


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SB3_ROOT = os.path.join(
    ROOT,
    "src/flexs/flexs/baselines/explorers/stable_baselines3",
)
DRAWS = 40000
FREQUENCY_TOLERANCE = 0.02


def _fail(report, status, code):
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return code


def _import_stack():
    if SB3_ROOT not in sys.path:
        sys.path.insert(0, SB3_ROOT)
    import gym
    import numpy as np
    import torch as th
    from stable_baselines3.common.distributions import MultiCategoricalDistribution
    from stable_baselines3.common.mixture import mixed_action_probability
    from stable_baselines3.common.policies import ActorCriticPolicy

    return gym, np, th, MultiCategoricalDistribution, mixed_action_probability, ActorCriticPolicy


def _marginal(probability, categories):
    return 0.75 * float(probability) + 0.25 / float(categories)


def _legacy(probability, categories):
    return 0.50 * float(probability) + 0.50 / float(categories)


def _frequency(distribution, categories, draws):
    import numpy as np

    counts = np.zeros(categories, dtype=np.int64)
    devices = set()
    for _ in range(draws):
        action = distribution.sample()
        devices.add(str(action.device))
        counts[int(action.item())] += 1
    return counts / draws, sorted(devices)


def _policy(actor_cls, gym, categories):
    import numpy as np

    observation = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(4,), dtype=np.float32
    )
    action = gym.spaces.MultiDiscrete([categories])
    policy = actor_cls(
        observation,
        action,
        lambda _progress: 1e-3,
        net_arch=[dict(pi=[16], vf=[16])],
        ortho_init=False,
    )
    policy.action_dist.mixture_mode = "marginal"
    return policy


def _fixed_distribution(distribution_cls, torch, categories, mode):
    logits = torch.zeros(1, categories)
    logits[0, 0] = 2.0
    logits[0, 1] = 0.5
    distribution = distribution_cls([categories], mixture_mode=mode)
    distribution.proba_distribution(logits)
    return distribution, logits


def run_checks(modules, report):
    gym, np, th, distribution_cls, mixed, actor_cls = modules
    np.random.seed(0)
    th.manual_seed(0)
    report["distribution_class"] = (
        distribution_cls.__module__ + "." + distribution_cls.__name__
    )
    report["checks"] = {}

    for label, categories in (("site", 8), ("amino_acid", 20)):
        distribution, _logits = _fixed_distribution(
            distribution_cls, th, categories, "marginal"
        )
        probabilities = distribution.distribution[0].probs.detach().cpu().numpy()[0]
        expected = np.array(
            [_marginal(p, categories) for p in probabilities]
        )
        frequency, devices = _frequency(distribution, categories, DRAWS)
        gap = float(np.max(np.abs(frequency - expected)))
        report["checks"][label + "_empirical_frequency"] = {
            "draws": DRAWS,
            "categories": categories,
            "max_abs_error": gap,
            "sample_devices": devices,
            "passed": gap < FREQUENCY_TOLERANCE and devices == ["cpu"],
        }
        action = th.tensor([[0]])
        logged = float(distribution.log_prob(action).item())
        formula = math.log(float(mixed(probabilities[0], categories, "marginal")))
        report["checks"][label + "_log_prob"] = {
            "logged": logged,
            "marginal_formula": formula,
            "passed": abs(logged - formula) < 1e-5,
        }

    # The TEM-1 site factor has one categorical over the sequence length.
    # One evaluation checks that cardinality. The frequency test above uses
    # eight categories so 40,000 draws can resolve the bins.
    long_site, _logits = _fixed_distribution(distribution_cls, th, 286, "marginal")
    long_probability = float(long_site.distribution[0].probs.detach().cpu().numpy()[0, 3])
    long_logged = float(long_site.log_prob(th.tensor([[3]])).item())
    long_formula = math.log(_marginal(long_probability, 286))
    report["checks"]["site_length_286_log_prob"] = {
        "logged": long_logged,
        "marginal_formula": long_formula,
        "passed": abs(long_logged - long_formula) < 1e-5,
    }

    legacy, _logits = _fixed_distribution(distribution_cls, th, 20, "legacy")
    legacy_probability = legacy.distribution[0].probs.detach().cpu().numpy()[0]
    legacy_logged = float(legacy.log_prob(th.tensor([[0]])).item())
    legacy_formula = math.log(float(mixed(legacy_probability[0], 20, "legacy")))
    marginal_formula = math.log(_marginal(legacy_probability[0], 20))
    np.random.seed(0)
    legacy_frequency, _devices = _frequency(legacy, 20, DRAWS)
    sampling_target = np.array([_marginal(p, 20) for p in legacy_probability])
    legacy_target = np.array([_legacy(p, 20) for p in legacy_probability])
    report["checks"]["legacy_log_prob_is_not_the_sampler"] = {
        "logged": legacy_logged,
        "legacy_formula": legacy_formula,
        "log_prob_matches_legacy": abs(legacy_logged - legacy_formula) < 1e-5,
        "log_prob_is_not_marginal": abs(legacy_logged - marginal_formula) > 1e-4,
        "sample_max_abs_error_versus_marginal": float(
            np.max(np.abs(legacy_frequency - sampling_target))
        ),
        "sample_max_abs_error_versus_legacy_formula": float(
            np.max(np.abs(legacy_frequency - legacy_target))
        ),
        "passed": (
            abs(legacy_logged - legacy_formula) < 1e-5
            and abs(legacy_logged - marginal_formula) > 1e-4
            and float(np.max(np.abs(legacy_frequency - sampling_target)))
            < FREQUENCY_TOLERANCE
            and float(np.max(np.abs(legacy_frequency - legacy_target))) > 0.02
        ),
    }

    for label, categories in (("site_policy", 8), ("amino_acid_policy", 20)):
        policy = _policy(actor_cls, gym, categories).cpu()
        observation = th.zeros(1, 4)
        with th.no_grad():
            actions, _values, old_log_prob = policy.forward(observation)
            _values_2, new_log_prob, _entropy = policy.evaluate_actions(
                observation, actions
            )
        ratio = th.exp(new_log_prob - old_log_prob)
        distribution = policy.get_distribution(observation)
        chosen = int(actions.view(-1)[0].item())
        probability = float(
            distribution.distribution[0].probs.detach().cpu().numpy()[0, chosen]
        )
        formula = math.log(_marginal(probability, categories))
        report["checks"][label + "_evaluate_actions"] = {
            "old_log_prob": float(old_log_prob.item()),
            "new_log_prob": float(new_log_prob.item()),
            "ratio": float(ratio.item()),
            "marginal_formula": formula,
            "action_device": str(actions.device),
            "observation_device": str(observation.device),
            "passed": (
                abs(float(old_log_prob.item()) - float(new_log_prob.item())) < 1e-5
                and abs(float(ratio.item()) - 1.0) < 1e-5
                and abs(float(new_log_prob.item()) - formula) < 1e-5
                and actions.device.type == "cpu"
                and actions.device == observation.device
            ),
        }

    gpu = {"available": bool(th.cuda.is_available()), "passed": None}
    if not th.cuda.is_available():
        gpu["reason"] = "torch.cuda.is_available() is false in this interpreter"
        gpu["passed"] = None
        gpu["status"] = "not_available"
    else:
        try:
            policy = _policy(actor_cls, gym, 20).to("cuda")
            observation = th.zeros(1, 4, device="cuda")
            with th.no_grad():
                actions, _values, old_log_prob = policy.forward(observation)
                _values_2, new_log_prob, _entropy = policy.evaluate_actions(
                    observation, actions
                )
            distribution = policy.get_distribution(observation)
            sampled = distribution.sample()
            gpu.update(
                {
                    "action_device": str(actions.device),
                    "sampled_device": str(sampled.device),
                    "probability_device": str(distribution.distribution[0].probs.device),
                    "old_log_prob": float(old_log_prob.item()),
                    "new_log_prob": float(new_log_prob.item()),
                    "ratio": float(th.exp(new_log_prob - old_log_prob).item()),
                    "passed": (
                        actions.device.type == "cuda"
                        and sampled.device == distribution.distribution[0].probs.device
                        and abs(float(old_log_prob.item()) - float(new_log_prob.item()))
                        < 1e-5
                        and abs(float(th.exp(new_log_prob - old_log_prob).item()) - 1.0)
                        < 1e-5
                    ),
                    "status": "ran",
                }
            )
        except Exception as error:
            gpu["status"] = "error"
            gpu["passed"] = False
            gpu["error"] = "{0}: {1}".format(type(error).__name__, error)
    report["checks"]["gpu_device"] = gpu
    required = [
        item
        for name, item in report["checks"].items()
        if item.get("status") != "not_available"
    ]
    report["required_checks_passed"] = all(item.get("passed") is True for item in required)
    report["gpu_check_status"] = gpu["status"]
    return report


def main():
    report = {
        "interpreter": sys.executable,
        "python": sys.version.split()[0],
        "status": "not_run",
        "gym_stub": False,
    }
    try:
        modules = _import_stack()
    except Exception as error:
        report["status"] = "blocked"
        report["reason"] = "{0}: {1}".format(type(error).__name__, error)
        report["traceback"] = traceback.format_exc()
        report["checks_not_run"] = [
            "site sampling probability",
            "amino-acid sampling probability",
            "40000 empirical frequencies",
            "evaluate_actions log probability",
            "old_log_prob equals new_log_prob when parameters are unchanged",
            "PPO probability ratio equals 1 when the policy is unchanged",
            "action tensor device on CPU and on an available GPU",
        ]
        return _fail(report, "blocked", 2)
    try:
        run_checks(modules, report)
    except Exception as error:
        report["status"] = "failed"
        report["reason"] = "{0}: {1}".format(type(error).__name__, error)
        report["traceback"] = traceback.format_exc()
        return _fail(report, "failed", 1)
    if not report.get("required_checks_passed"):
        report["status"] = "failed"
        code = 1
    elif report.get("gpu_check_status") == "not_available":
        report["status"] = "incomplete"
        report["reason"] = (
            "CPU checks ran. No GPU is available in this interpreter, "
            "so the GPU device check did not run and is not a pass."
        )
        code = 3
    elif report.get("gpu_check_status") != "ran":
        report["status"] = "failed"
        code = 1
    else:
        report["status"] = "passed"
        code = 0
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
