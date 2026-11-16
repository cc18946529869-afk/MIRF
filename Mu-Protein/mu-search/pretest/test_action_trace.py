"""Action traces record edits. They do not reject repeats or empty edits."""

import csv
import os
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "flexs"))
sys.path.insert(0, ROOT)

from flexs.baselines.explorers.environments.env import LandscapeEnv
from flexs.musearch_baseline import write_action_trace


class _SilentModel:
    def get_fitness(self, sequences):
        return [0.0 for _ in sequences]


def test_trace_keeps_noop_and_revisit():
    env = LandscapeEnv(_SilentModel(), "AB", 3, "AAAA")
    env.step([0, 1])
    env.step([0, 1])
    _obs, _reward, done, _info = env.step([1, 0])
    if not done:
        raise SystemExit("horizon 3 did not finish")
    rows = env.action_trace
    if len(rows) != 3:
        raise SystemExit("trace length {0}".format(len(rows)))
    if [row["site"] for row in rows] != [1, 1, 2]:
        raise SystemExit(rows)
    if [row["old_aa"] for row in rows] != ["A", "B", "A"]:
        raise SystemExit(rows)
    if [row["new_aa"] for row in rows] != ["B", "B", "A"]:
        raise SystemExit(rows)
    if [row["no_op"] for row in rows] != [0, 1, 1]:
        raise SystemExit(rows)
    if [row["revisited"] for row in rows] != [0, 1, 0]:
        raise SystemExit(rows)
    if [row["final_hamming"] for row in rows] != [1, 1, 1]:
        raise SystemExit(rows)
    env.reset()
    env.step([2, 1])
    if env.action_trace[3]["trajectory"] != 2 or env.action_trace[3]["final_hamming"] != "":
        raise SystemExit(env.action_trace[3])
    with tempfile.TemporaryDirectory() as directory:
        path = write_action_trace(env.action_trace, os.path.join(directory, "action_trace.csv"))
        with open(path, encoding="utf-8", newline="") as handle:
            written = list(csv.DictReader(handle))
    if len(written) != 4 or written[1]["revisited"] != "1" or written[0]["final_hamming"] != "1":
        raise SystemExit(written)
    print("action_trace_ok rows={0} final_hamming={1}".format(len(rows), rows[0]["final_hamming"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(test_trace_keeps_noop_and_revisit())
