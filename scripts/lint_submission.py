"""Script wrapper for SubmissionLinter to validate submission files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.part_a.submission_linter import SubmissionLinter


def main():
    parser = argparse.ArgumentParser(description="LifeMate Submission Linter")
    parser.add_argument("pos_submission", nargs="?", default=None, help="Path to submission.jsonl")
    parser.add_argument("--submission", default=None, help="Path to submission.jsonl")
    parser.add_argument("--split", default="public_test", help="Split name (default: public_test)")
    parser.add_argument("--catalog", default=None, help="Path to catalog JSON")
    parser.add_argument("--jsonl", default=None, help="Path to ground truth JSONL")
    args = parser.parse_args()

    sub_path = args.submission or args.pos_submission
    if not sub_path:
        print("❌ Please specify submission file: python scripts/lint_submission.py <path_to_submission.jsonl>")
        sys.exit(1)

    sub_p = Path(sub_path)
    if not sub_p.exists():
        print(f"❌ Submission file not found: {sub_p}")
        sys.exit(1)

    linter = SubmissionLinter(split=args.split, jsonl_path=args.jsonl, catalog_path=args.catalog)
    report = linter.lint_file(str(sub_p))
    linter.print_report(report)

    if not report["passed"]:
        print("⛔ Submission validation FAILED. Do not submit this file.")
        sys.exit(1)
    else:
        print("🎉 Submission validation PASSED 100%! Ready to submit to Zalo AI Challenge!")


if __name__ == "__main__":
    main()
