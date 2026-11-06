"""Benchmark Part C — Paired Gold-vs-ASR Evaluation (§11 of Plan C).

Runs evaluation on the dev set to measure downstream Agent performance:
1. Gold transcript (measures Agent reasoning in isolation)
2. ASR transcript (measures end-to-end performance)
Reports hash-match accuracy, call precision, recall, and F1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import DATA_ROOT, TRAIN_CATALOG, SPLITS_DIR
from src.part_a.runner import PipelineRunner
from src.part_a.evaluator import Evaluator
from src.part_b.asr_module import GoldTranscriptASR, ASRModule
from src.part_c.agent import create_agent


def run_benchmark(
    split: str = "dev",
    max_samples: int | None = 50,
    agent_mode: str = "rule",
    compare_asr: bool = False,
):
    print("=" * 80)
    print("LIFEMATE PART C — BENCHMARK EVALUATION")
    print(f"Split: {split} | Max Samples: {max_samples} | Agent Mode: {agent_mode}")
    print("=" * 80)

    catalog_path = DATA_ROOT / "train" / "train_tools_catalog.json"
    agent = create_agent(mode=agent_mode, catalog_path=catalog_path)
    if hasattr(agent, "load"):
        agent.load()

    # 1. Gold Transcript Run
    print("\n[Run 1] Evaluating Agent with Gold Transcript (isolated LLM reasoning)...")
    gold_asr = GoldTranscriptASR()
    gold_runner = PipelineRunner(
        split=split,
        asr_fn=gold_asr,
        agent_fn=agent.run,
        verbose=False,
    )
    gold_results = gold_runner.run()
    if max_samples:
        gold_results = gold_results[:max_samples]

    evaluator = Evaluator(split=split)
    gold_metrics = evaluator.evaluate(gold_results)

    print("\n--- Gold Transcript Results ---")
    print(f"Score (with Hash Gate): {gold_metrics.get('score', 0):.4f}")
    print(f"Hash Match Rate:        {gold_metrics.get('hash_match_rate', 0):.4f}")
    print(f"Call Precision:         {gold_metrics.get('call_precision', 0):.4f}")
    print(f"Call Recall:            {gold_metrics.get('call_recall', 0):.4f}")
    print(f"Call F1:                {gold_metrics.get('call_f1', 0):.4f}")

    if compare_asr:
        print("\n[Run 2] Evaluating Agent with ASR Transcript...")
        # Note: requires whisper weights and audio files
        asr = ASRModule(device="cpu", beam_size=1)
        try:
            asr.load()
            asr_runner = PipelineRunner(
                split=split,
                asr_fn=asr,
                agent_fn=agent.run,
                verbose=False,
            )
            asr_results = asr_runner.run()
            if max_samples:
                asr_results = asr_results[:max_samples]
            asr_metrics = evaluator.evaluate(asr_results)
            print("\n--- ASR Transcript Results ---")
            print(f"Score (with Hash Gate): {asr_metrics.get('score', 0):.4f}")
            print(f"Hash Match Rate:        {asr_metrics.get('hash_match_rate', 0):.4f}")
            print(f"Call F1:                {asr_metrics.get('call_f1', 0):.4f}")
        except Exception as e:
            print(f"  ⚠️ Skipping ASR run: {e}")

    print("\n" + "=" * 80)
    print("Benchmark complete.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Part C Benchmark")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--agent-mode", default="rule")
    parser.add_argument("--compare-asr", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        split=args.split,
        max_samples=args.max_samples,
        agent_mode=args.agent_mode,
        compare_asr=args.compare_asr,
    )


if __name__ == "__main__":
    main()
