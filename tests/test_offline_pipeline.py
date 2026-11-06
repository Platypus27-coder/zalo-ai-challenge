"""Tests for Offline Pipeline Runner & Submission Linter (§10 of Plan C)."""
import json
import tempfile
import unittest
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_ROOT
from src.part_a.runner import PipelineRunner
from src.part_a.submission_linter import SubmissionLinter
from src.part_b.asr_module import GoldTranscriptASR
from src.part_c.agent import RuleBasedAgent


class TestOfflinePipeline(unittest.TestCase):

    def test_runner_on_train_samples(self):
        """Run RuleBasedAgent through PipelineRunner on 3 samples and verify output structure."""
        catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
        agent = RuleBasedAgent(catalog_path=catalog_path)
        asr = GoldTranscriptASR()

        runner = PipelineRunner(
            split="train",
            asr_fn=asr,
            agent_fn=agent.run,
            sample_ids=["train_0001", "train_0002", "train_0003"],
            verbose=False,
        )

        results = runner.run()
        self.assertEqual(len(results), 3)

        for res in results:
            self.assertIn("id", res)
            self.assertIn("tool_calls", res)
            self.assertIn("state_hash", res)
            self.assertEqual(len(res["state_hash"]), 64)
            # Since RuleBasedAgent does nothing, tool_calls must be empty
            self.assertEqual(len(res["tool_calls"]), 0)

    def test_submission_linter_passes_on_valid_output(self):
        """SubmissionLinter should pass on valid outputs with real state_hashes."""
        catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
        agent = RuleBasedAgent(catalog_path=catalog_path)
        asr = GoldTranscriptASR()

        runner = PipelineRunner(
            split="train",
            asr_fn=asr,
            agent_fn=agent.run,
            sample_ids=["train_0001", "train_0002"],
            verbose=False,
        )

        results = runner.run()
        
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            for r in results:
                tmp.write(json.dumps(r, ensure_ascii=False) + "\n")

        try:
            linter = SubmissionLinter(
                split="train",
                catalog_path=catalog_path,
                jsonl_path=DATA_ROOT / "train" / "train.jsonl"
            )
            # Expected samples restricted to these 2
            linter.expected_samples = {r["id"]: {} for r in results}
            report = linter.lint_file(tmp_path)
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["critical_errors"]), 0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_submission_linter_fails_on_empty_state_hash(self):
        """SubmissionLinter must fail if any state_hash is empty."""
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            tmp_path = Path(tmp.name)
            bad_entry = {"id": "train_0001", "tool_calls": [], "state_hash": ""}
            tmp.write(json.dumps(bad_entry) + "\n")

        try:
            catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
            linter = SubmissionLinter(split="train", catalog_path=catalog_path)
            linter.expected_samples = {"train_0001": {}}
            report = linter.lint_file(tmp_path)
            self.assertFalse(report["passed"])
            self.assertGreater(len(report["critical_errors"]), 0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
