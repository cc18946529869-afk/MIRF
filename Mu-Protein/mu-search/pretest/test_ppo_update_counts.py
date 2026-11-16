"""Real PPO train on a synthetic buffer. Does not load MuFormer.

batch_size stays 512 and n_epochs stays 10. The test calls train() once per
rollout, three times, on a buffer the size of one horizon rollout.
"""

import importlib.util
import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "flexs"))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import stable_baselines3.ppo.ppo as _installed_ppo

_PPO_PATH = os.path.join(
    ROOT,
    "flexs",
    "baselines",
    "explorers",
    "stable_baselines3",
    "stable_baselines3",
    "ppo",
    "ppo.py",
)
_spec = importlib.util.spec_from_file_location("stable_baselines3.ppo.ppo", _PPO_PATH)
_ppo_module = importlib.util.module_from_spec(_spec)
sys.modules["stable_baselines3.ppo.ppo"] = _ppo_module
_spec.loader.exec_module(_ppo_module)
PPO = _ppo_module.PPO
del _installed_ppo

from flexs.baselines.explorers.environments.env import LandscapeEnv
from flexs.musearch_baseline import floor_rollout_schedule


class _SilentModel:
    def get_fitness(self, sequences):
        return [0.0 for _ in sequences]


def _agent(horizon, n_steps):
    torch.manual_seed(0)
    np.random.seed(0)
    model = _SilentModel()
    env = LandscapeEnv(model, "AB", horizon, "AAAAAAAA")
    example = LandscapeEnv(model, "AB", horizon, "AAAAAAAA")
    return PPO(
        ["MlpPolicy", "MlpPolicy"],
        env,
        batch_size=512,
        n_epochs=10,
        verbose=0,
        n_steps=n_steps,
        for_protein=True,
        score_threshold=-np.inf,
        device="cpu",
        tensorboard_log=None,
        policy_kwargs={
            "net_arch": [dict(pi=[16], vf=[16])],
            "sub_step_observation_spaces1": example.observation_space1,
            "sub_step_observation_spaces2": example.observation_space2,
            "sub_step_action_spaces1": example.action_space1,
            "sub_step_action_spaces2": example.action_space2,
        },
    )


def _fill(agent):
    buffer = agent.rollout_buffer
    buffer.reset()
    buffer.full = True


def test_three_rollouts():
    expected = {5: 510, 7: 690, 10: 990}
    for horizon, steps_per_policy in expected.items():
        schedule = floor_rollout_schedule(4998, horizon, 3)
        agent = _agent(horizon, schedule["n_steps"])
        ppo_file = os.path.realpath(PPO.__module__ and sys.modules[PPO.__module__].__file__)
        if "MIRF-review-fixes-2026-11-16" not in ppo_file:
            raise SystemExit("PPO was not loaded from the fixes worktree: {0}".format(ppo_file))
        per_epoch = math.ceil(schedule["n_steps"] / agent.batch_size)
        agent._setup_learn(schedule["total_timesteps"], None, True, "PPO", False)
        agent._update_current_progress_remaining(0, schedule["total_timesteps"])
        _fill(agent)
        for _rollout in range(schedule["rollouts"]):
            agent.train()
        if agent.optimizer_steps != steps_per_policy or agent.optimizer_steps_2 != steps_per_policy:
            raise SystemExit(
                "H={0} site={1} amino={2}".format(
                    horizon, agent.optimizer_steps, agent.optimizer_steps_2
                )
            )
        if agent._n_updates != schedule["rollouts"] * agent.n_epochs:
            raise SystemExit("H={0} n_updates={1}".format(horizon, agent._n_updates))
        if agent._n_updates == agent.optimizer_steps:
            raise SystemExit("n_updates collapsed onto optimizer steps")
        if agent.minibatch_counts != [per_epoch] * (schedule["rollouts"] * agent.n_epochs):
            raise SystemExit("H={0} minibatch counts {1}".format(horizon, set(agent.minibatch_counts)))
        tail = schedule["n_steps"] % agent.batch_size
        one_epoch = agent.minibatch_sizes[0]
        if len(one_epoch) != per_epoch or one_epoch[-1] != tail or any(size != 512 for size in one_epoch[:-1]):
            raise SystemExit("H={0} minibatch sizes {1}".format(horizon, one_epoch[-3:]))
        if agent.ppo_train_seconds <= 0:
            raise SystemExit("train time was not recorded")
        if schedule["timesteps_collected"] != 4998 * horizon:
            raise SystemExit("env steps drifted")
        print(
            "ppo_updates H={0} optimizer_steps={1} n_updates={2} minibatches_per_epoch={3} tail={4} train_seconds={5:.3f} env_steps={6}".format(
                horizon,
                agent.optimizer_steps,
                agent._n_updates,
                per_epoch,
                tail,
                agent.ppo_train_seconds,
                schedule["timesteps_collected"],
            )
        )
    print("ppo_update_counts_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(test_three_rollouts())
