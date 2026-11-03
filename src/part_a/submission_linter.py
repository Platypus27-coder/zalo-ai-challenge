"""Part A — Submission Linter & Pre-Flight Verifier.

Enforces zero-defect submission files to prevent:
1. Disqualification due to format errors or missing samples.
2. Zero score due to invalid state_hash or calling tools from uninstalled apps.
3. Schema violations (missing required parameters, wrong types, invalid enums).

Usage:
    python -m src.part_a.submission_linter --submission outputs/submissions/submission.jsonl --split public_test
"""
import json
import re
import argparse
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import (
    DATA_ROOT, PUBLIC_TEST_JSONL, PUBLIC_TEST_CATALOG,
    TRAIN_JSONL, TRAIN_CATALOG, SPLITS_DIR
)
from src.part_a.validator import load_catalog, get_tool_specs, validate_call


SHA256_HEX_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


class SubmissionLinter:
    """Validates submission files against dataset ground rules and catalog constraints."""

    def __init__(self, split: str = "public_test"):
        self.split = split
        if split == "public_test":
            self.jsonl_path = PUBLIC_TEST_JSONL
            self.catalog_path = PUBLIC_TEST_CATALOG
        else:
            self.jsonl_path = TRAIN_JSONL
            self.catalog_path = TRAIN_CATALOG

        self.catalog = load_catalog(self.catalog_path)
        self.tool_specs = get_tool_specs(self.catalog)
        self.expected_samples = self._load_expected_samples()

    def _load_expected_samples(self) -> dict[str, dict]:
        samples = {}
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    samples[item["id"]] = item

        if self.split in ("dev", "holdout"):
            splits_file = SPLITS_DIR / "splits.json"
            if splits_file.exists():
                with open(splits_file, encoding="utf-8") as f:
                    target_ids = set(json.load(f)["splits"].get(self.split, []))
                samples = {k: v for k, v in samples.items() if k in target_ids}

        return samples

    def lint_file(self, submission_path: Path | str) -> dict[str, Any]:
        """Lint a submission file and return structured report."""
        submission_path = Path(submission_path)
        report = {
            "file": str(submission_path),
            "split": self.split,
            "total_lines": 0,
            "passed": True,
            "critical_errors": [],
            "warnings": [],
            "stats": {
                "total_samples": 0,
                "oos_count": 0,
                "call_count": 0,
                "max_calls": 0,
                "avg_calls": 0.0,
                "empty_hashes": 0,
                "valid_hashes": 0,
            }
        }

        if not submission_path.exists():
            report["passed"] = False
            report["critical_errors"].append(f"Submission file does not exist: {submission_path}")
            return report

        seen_ids = set()
        submission_ids = []
        parsed_entries = []

        with open(submission_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                report["total_lines"] += 1
                line = line.strip()
                if not line:
                    report["warnings"].append(f"Line {line_idx}: Empty line ignored.")
                    continue

                try:
                    entry = json.loads(line)
                except Exception as e:
                    report["passed"] = False
                    report["critical_errors"].append(f"Line {line_idx}: Invalid JSON syntax ({e})")
                    continue

                # 1. Structure check: must have id, tool_calls, state_hash
                for key in ["id", "tool_calls", "state_hash"]:
                    if key not in entry:
                        report["passed"] = False
                        report["critical_errors"].append(f"Line {line_idx}: Missing required key '{key}'")

                sample_id = entry.get("id")
                if not sample_id:
                    report["passed"] = False
                    report["critical_errors"].append(f"Line {line_idx}: Empty or null 'id'")
                    continue

                if sample_id in seen_ids:
                    report["passed"] = False
                    report["critical_errors"].append(f"Line {line_idx}: Duplicate sample ID '{sample_id}'")
                seen_ids.add(sample_id)
                submission_ids.append(sample_id)

                if sample_id not in self.expected_samples:
                    report["passed"] = False
                    report["critical_errors"].append(f"Line {line_idx}: ID '{sample_id}' not found in {self.split}")
                    continue

                parsed_entries.append((line_idx, entry, self.expected_samples[sample_id]))

        expected_ids = set(self.expected_samples.keys())
        missing_ids = expected_ids - seen_ids
        if missing_ids:
            report["passed"] = False
            report["critical_errors"].append(
                f"Missing {len(missing_ids)} expected samples from {self.split}. Sample missing: {list(missing_ids)[:5]}"
            )

        total_tool_calls = 0
        max_calls = 0

        # Detailed entry checks
        for line_idx, entry, expected in parsed_entries:
            sample_id = entry["id"]
            tool_calls = entry.get("tool_calls", [])
            state_hash = entry.get("state_hash", "")

            if not isinstance(tool_calls, list):
                report["passed"] = False
                report["critical_errors"].append(f"Line {line_idx} ({sample_id}): 'tool_calls' must be a list")
                continue

            n_calls = len(tool_calls)
            total_tool_calls += n_calls
            max_calls = max(max_calls, n_calls)

            if n_calls == 0:
                report["stats"]["oos_count"] += 1
            else:
                report["stats"]["call_count"] += n_calls

            # Hash check
            if not isinstance(state_hash, str):
                report["passed"] = False
                report["critical_errors"].append(f"Line {line_idx} ({sample_id}): 'state_hash' must be string")
            else:
                if state_hash == "":
                    report["stats"]["empty_hashes"] += 1
                    if n_calls > 0:
                        report["warnings"].append(
                            f"Line {line_idx} ({sample_id}): tool_calls is non-empty ({n_calls} calls) but state_hash is empty"
                        )
                elif SHA256_HEX_PATTERN.match(state_hash):
                    report["stats"]["valid_hashes"] += 1
                else:
                    report["passed"] = False
                    report["critical_errors"].append(
                        f"Line {line_idx} ({sample_id}): 'state_hash' is not a valid 64-char sha256 hex: '{state_hash[:10]}...'"
                    )

            # Installed tools check
            installed_tools = set()
            for app_name, app_info in expected.get("context", {}).get("installed_apps", {}).items():
                installed_tools.update(app_info.get("tools", []))
            installed_tools.add("sys.ask_missing_info")

            for call_idx, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    report["passed"] = False
                    report["critical_errors"].append(f"Line {line_idx} ({sample_id}) call #{call_idx}: call must be a dict")
                    continue

                call_name = call.get("name")
                call_args = call.get("arguments", {})

                if not call_name:
                    report["passed"] = False
                    report["critical_errors"].append(f"Line {line_idx} ({sample_id}) call #{call_idx}: missing 'name'")
                    continue

                if not isinstance(call_args, dict):
                    report["passed"] = False
                    report["critical_errors"].append(f"Line {line_idx} ({sample_id}) call #{call_idx}: 'arguments' must be dict")
                    continue

                # Pre-call validator check
                val = validate_call(call_name, call_args, self.tool_specs, installed_tools)
                if not val["ok"]:
                    for err in val["errors"]:
                        # Uninstalled app or unknown tool is critical
                        if "not in installed_apps" in err or "not in catalog" in err:
                            report["passed"] = False
                            report["critical_errors"].append(f"Line {line_idx} ({sample_id}): {err}")
                        else:
                            report["warnings"].append(f"Line {line_idx} ({sample_id}) [{call_name}]: {err}")

        n_samples = len(parsed_entries)
        report["stats"]["total_samples"] = n_samples
        report["stats"]["max_calls"] = max_calls
        report["stats"]["avg_calls"] = round(total_tool_calls / n_samples, 2) if n_samples > 0 else 0.0

        return report

    def print_report(self, report: dict[str, Any]):
        """Format and print linter report."""
        print("=" * 80)
        print("LIFEMATE SUBMISSION INTEGRITY REPORT")
        print("=" * 80)
        print(f"File:    {report['file']}")
        print(f"Split:   {report['split']}")
        print(f"Status:  {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
        print("-" * 80)
        stats = report["stats"]
        print(f"Total Samples:       {stats['total_samples']}")
        print(f"OOS (Empty Calls):   {stats['oos_count']} ({stats['oos_count']/max(1, stats['total_samples'])*100:.1f}%)")
        print(f"Total Tool Calls:    {stats['call_count']}")
        print(f"Avg Calls / Sample:  {stats['avg_calls']}")
        print(f"Max Calls / Sample:  {stats['max_calls']}")
        print(f"Valid SHA256 Hashes: {stats['valid_hashes']}")
        print(f"Empty Hashes:        {stats['empty_hashes']}")
        print("-" * 80)

        if report["critical_errors"]:
            print(f"🚨 CRITICAL ERRORS ({len(report['critical_errors'])}):")
            for err in report["critical_errors"][:15]:
                print(f"   ✖ {err}")
            if len(report["critical_errors"]) > 15:
                print(f"   ... and {len(report['critical_errors']) - 15} more errors")
            print()

        if report["warnings"]:
            print(f"⚠️  WARNINGS ({len(report['warnings'])}):")
            for w in report["warnings"][:10]:
                print(f"   ! {w}")
            if len(report["warnings"]) > 10:
                print(f"   ... and {len(report['warnings']) - 10} more warnings")
            print()

        if report["passed"]:
            print("🎉 Submission is structurally 100% compliant and safe to submit!")
        else:
            print("⛔ DO NOT SUBMIT THIS FILE. Fix critical errors above first.")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LifeMate Submission Linter")
    parser.add_argument("--submission", required=True, help="Path to submission.jsonl")
    parser.add_argument("--split", default="public_test", choices=["train", "public_test"])
    args = parser.parse_args()

    linter = SubmissionLinter(split=args.split)
    report = linter.lint_file(args.submission)
    linter.print_report(report)


if __name__ == "__main__":
    main()
