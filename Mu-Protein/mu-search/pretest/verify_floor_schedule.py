"""Confirm the real learn loop keeps three rollouts at 24,990 steps.

collect_rollouts only advances the step counter. It does not score MuFormer.
"""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "flexs"))
sys.path.insert(0, ROOT)

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from flexs.musearch_baseline import floor_rollout_schedule


class Harness:
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


def run(total):
    harness = Harness(8330)
    OnPolicyAlgorithm.learn(harness, total, log_interval=None)
    print(
        "total={0} rollouts={1} num_timesteps={2}".format(
            total, harness.rollouts, harness.num_timesteps
        )
    )
    return harness.rollouts, harness.num_timesteps


def main():
    schedule = floor_rollout_schedule(4998, 5, 3)
    if schedule["terminal_scores"] != 4998 or schedule["rollouts"] != 3:
        raise SystemExit("floor schedule is not three rollouts of 1666")
    rollouts, steps = run(24990)
    if rollouts != 3 or steps != 24990:
        raise SystemExit("24990 steps did not keep three rollouts")
    rollouts, steps = run(25000)
    if rollouts != 3 or steps != 24990:
        raise SystemExit("25000 steps changed the three-rollout count")
    print("floor_schedule_ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
