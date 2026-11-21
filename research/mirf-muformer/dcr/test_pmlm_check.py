"""A missing PMLM checkpoint is recorded as a failure."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pmlm_check


class PmlmCheckTests(unittest.TestCase):
    def test_a_missing_checkpoint_is_not_reported_as_a_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "pmlm.json"
            result = pmlm_check.verify_frozen_pmlm(
                Path(directory) / "missing.pt",
                Path("domain_map.tsv"),
                destination,
                "A",
            )
            self.assertFalse(result["passed"])
            self.assertIn("error", result)
            saved = json.loads(destination.read_text())
            self.assertFalse(saved["passed"])
            self.assertFalse(saved["downloaded"])
            self.assertFalse(saved["trained"])


if __name__ == "__main__":
    unittest.main()
