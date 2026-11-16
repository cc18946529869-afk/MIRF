"""Execute the real learn loop at H=7 and H=10. Does not load MuFormer.

The full budgets use the real learn() break. ``execute_full_collect`` replaces
``agent.train`` with a no-op, so it checks 4998 terminal scores and does not
run ``optimizer.step``. Optimizer-step counts are checked by
``test_ppo_update_counts.py``. A short run for each horizon uses the real
LandscapeEnv, real collect_rollouts, and real train, and checks that fitness
is requested once per completed episode.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "flexs"))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.ppo.ppo import PPO

from flexs.baselines.explorers.environments.env import LandscapeEnv
from flexs.musearch_baseline import floor_rollout_schedule
from flexs.utils.sequence_utils import AAS


class StepHarness:
    def __init__(self, n_steps):
        self.n_steps = n_steps
        self.num_timesteps = 0
        self.rollouts = 0
        self.for_protein = False
        self.env = None
        self.rollout_buffer = None

    def _setup_learn(self, total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar):
        class Callback:
            def on_training_start(self, *args, **kwargs):
                return None

            def on_training_end(self):
                return None

        return total_timesteps, Callback()

    def collect_rollouts(self, *args, **kwargs):
        self.num_timesteps += self.n_steps
        self.rollouts += 1
        return True

    def train(self):
        return None

    def _update_current_progress_remaining(self, *args, **kwargs):
        return None


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.name = "CountingModel"

    def get_fitness(self, sequences):
        sequences = list(sequences)
        self.calls += len(sequences)
        return [0.0 for _ in sequences]


def execute_learn_loop(horizon):
    schedule = floor_rollout_schedule(4998, horizon, 3)
    harness = StepHarness(schedule["n_steps"])
    OnPolicyAlgorithm.learn(harness, schedule["total_timesteps"], log_interval=None)
    print(
        "learn_loop H={0} rollouts={1} num_timesteps={2} n_steps={3} total={4}".format(
            horizon, harness.rollouts, harness.num_timesteps,
            schedule["n_steps"], schedule["total_timesteps"],
        )
    )
    if harness.rollouts != 3 or harness.num_timesteps != 4998 * horizon:
        raise SystemExit("H={0} learn loop did not collect 4998 episodes".format(horizon))
    if harness.num_timesteps // horizon != 4998:
        raise SystemExit("H={0} timesteps are not 4998 episodes".format(horizon))
    return schedule


def execute_full_collect(horizon):
    """Real collect_rollouts and the real learn loop.

    train is replaced by a no-op. This does not verify PPO optimization.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    model = CountingModel()
    schedule = floor_rollout_schedule(4998, horizon, 3)
    env = LandscapeEnv(model, AAS, horizon, "A" * 8)
    example = LandscapeEnv(model, AAS, horizon, "A" * 8)
    agent = PPO(
        ["MlpPolicy", "MlpPolicy"],
        env,
        batch_size=512,
        n_epochs=10,
        verbose=0,
        n_steps=schedule["n_steps"],
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
    agent.train = lambda: None
    agent.learn(schedule["total_timesteps"], log_interval=None)
    print(
        "full_collect H={0} fitness_calls={1} num_timesteps={2}".format(
            horizon, model.calls, agent.num_timesteps
        )
    )
    if model.calls != 4998 or agent.num_timesteps != 4998 * horizon:
        raise SystemExit(
            "H={0} scored {1} sequences in {2} steps".format(
                horizon, model.calls, agent.num_timesteps
            )
        )


def execute_short_env(horizon):
    torch.manual_seed(0)
    np.random.seed(0)
    model = CountingModel()
    episodes = 2
    n_steps = episodes * horizon
    env = LandscapeEnv(model, AAS, horizon, "A" * 8)
    example = LandscapeEnv(model, AAS, horizon, "A" * 8)
    agent = PPO(
        ["MlpPolicy", "MlpPolicy"],
        env,
        batch_size=n_steps,
        n_epochs=1,
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
    agent.learn(n_steps, log_interval=None)
    print(
        "short_env H={0} fitness_calls={1} num_timesteps={2}".format(
            horizon, model.calls, agent.num_timesteps
        )
    )
    if model.calls != episodes or agent.num_timesteps != n_steps:
        raise SystemExit(
            "H={0} scored {1} sequences in {2} steps".format(
                horizon, model.calls, agent.num_timesteps
            )
        )


def main():
    for horizon in (7, 10):
        schedule = execute_learn_loop(horizon)
        if schedule["terminal_scores"] != 4998 or schedule["episodes_per_update"] != 1666:
            raise SystemExit("floor schedule drifted at H={0}".format(horizon))
        execute_full_collect(horizon)
        execute_short_env(horizon)
    print("horizon_schedule_ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
