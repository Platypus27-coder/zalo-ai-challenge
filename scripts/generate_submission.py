"""Generate submission JSONL for evaluation or submission (§4 and §13 of Plan C).

Usage:
    python scripts/generate_submission.py --split public_test --agent-mode transformers --asr-mode whisper
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATA_ROOT, SUBMISSIONS_DIR, DEFAULT_LLM_MODEL, DEFAULT_ASR_MODEL,
    LOCAL_LLM_DIR, LOCAL_ASR_DIR
)
from src.part_a.runner import PipelineRunner
from src.part_a.submission_linter import SubmissionLinter
from src.part_b.asr_module import ASRModule, GoldTranscriptASR
from src.part_c.agent import create_agent


def main():
    parser = argparse.ArgumentParser(description="Generate submission file")
    parser.add_argument("--split", default="public_test", choices=["train", "dev", "holdout", "public_test", "private_test"])
    parser.add_argument("--agent-mode", default="rule", choices=["rule", "transformers", "vllm"])
    parser.add_argument("--asr-mode", default="whisper", choices=["whisper", "gold"])
    parser.add_argument("--model-name", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    split = args.split

    # Safety check: do not use gold ASR on test splits
    if split in ("public_test", "private_test") and args.asr_mode == "gold":
        print(f"❌ ERROR: Cannot use gold ASR on test split '{split}' because ground truth audio_text does not exist!")
        sys.exit(1)

    # 1. Initialize ASR
    if args.asr_mode == "gold":
        asr = GoldTranscriptASR()
    else:
        model_path = str(LOCAL_ASR_DIR) if LOCAL_ASR_DIR.is_dir() else args.asr_model
        asr = ASRModule(model_name=model_path, device="cuda" if sys.platform != "win32" else "cpu")
        try:
            asr.load()
        except Exception:
            asr = ASRModule(model_name=model_path, device="cpu")
            asr.load()

    # 2. Initialize Agent
    data_split = "train" if split in ("train", "dev", "holdout") else split
    catalog_path = DATA_ROOT / data_split / f"{data_split}_tools_catalog.json"
    
    agent_model = str(LOCAL_LLM_DIR) if LOCAL_LLM_DIR.is_dir() else args.model_name
    agent = create_agent(
        mode=args.agent_mode,
        model_name=agent_model,
        catalog_path=catalog_path,
        verbose=args.verbose,
    )
    if hasattr(agent, "load"):
        agent.load()

    # 3. Run PipelineRunner
    runner = PipelineRunner(
        split=split,
        asr_fn=asr,
        agent_fn=agent.run,
        verbose=args.verbose or True,
    )

    results = runner.run()
    if args.max_samples:
        results = results[:args.max_samples]

    # 4. Save Submission
    out_file = runner.save_submission(results, output_path=args.output)

    # 5. Lint Submission
    print("\nVerifying submission with SubmissionLinter...")
    linter = SubmissionLinter(split=split)
    report = linter.lint_file(out_file)
    linter.print_report(report)

    if not report["passed"]:
        print("❌ Submission failed integrity checks!")
        sys.exit(1)


if __name__ == "__main__":
    main()
