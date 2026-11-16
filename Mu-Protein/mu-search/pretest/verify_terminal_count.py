"""Count terminal scores with the real PPO loop. Does not load MuFormer.

The short run uses the real collect_rollouts, LandscapeEnv, and train.
The 25,000-step runs use the real collect_rollouts and the real learn loop.
Their train method is a no-op: fitness is requested in collect_rollouts,
before train, and train does not score sequences.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "flexs"))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from stable_baselines3.ppo.ppo import PPO

from flexs.baselines.explorers.environments.env import LandscapeEnv
from flexs.musearch_baseline import exact_terminal_schedule
from flexs.utils.sequence_utils import AAS


class CountingModel:
    def __init__(self):
        self.calls = 0
        self.name = "CountingModel"

    def get_fitness(self, sequences):
        sequences = list(sequences)
        self.calls += len(sequences)
        return [0.0 for _ in sequences]


def make_agent(model, n_steps, batch_size, n_epochs):
    env = LandscapeEnv(model, AAS, 5, "A" * 8)
    example = LandscapeEnv(model, AAS, 5, "A" * 8)
    agent = PPO(
        ["MlpPolicy", "MlpPolicy"],
        env,
        batch_size=batch_size,
        n_epochs=n_epochs,
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
    return agent


def run_case(label, n_steps, total_timesteps, batch_size, n_epochs, noop_train):
    torch.manual_seed(0)
    np.random.seed(0)
    model = CountingModel()
    agent = make_agent(model, n_steps, batch_size, n_epochs)
    if noop_train:
        agent.train = lambda: None
    agent.learn(total_timesteps, log_interval=None)
    print(
        "{0}: fitness_calls={1} num_timesteps={2} n_steps={3} total={4}".format(
            label, model.calls, agent.num_timesteps, n_steps, total_timesteps
        )
    )
    return model.calls, agent.num_timesteps


def main():
    schedule = exact_terminal_schedule(5000, 5, 3)
    if schedule["terminal_scores"] != 5000 or schedule["n_steps"] != 6250:
        raise SystemExit("exact schedule is not 4 x 1250")

    short_calls, short_steps = run_case(
        "short_real_train", 10, 10, batch_size=10, n_epochs=1, noop_train=False
    )
    if short_calls != 2 or short_steps != 10:
        raise SystemExit("10 steps did not produce 2 terminal scores")

    old_calls, old_steps = run_case(
        "old_floor_split", 8330, 25000, batch_size=64, n_epochs=1, noop_train=True
    )
    if old_calls != 4998 or old_steps != 24990:
        raise SystemExit("old schedule was not 4998 terminal scores")

    new_calls, new_steps = run_case(
        "exact_divisor", 6250, 25000, batch_size=64, n_epochs=1, noop_train=True
    )
    if new_calls != 5000 or new_steps != 25000:
        raise SystemExit("new schedule was not 5000 terminal scores")
    print("terminal_count_ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
