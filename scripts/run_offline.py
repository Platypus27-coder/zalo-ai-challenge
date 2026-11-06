"""Part A — Official Offline Inference Entrypoint for Docker (§12 of Plan C).

Execution rules:
- 100% offline (no external network or download calls).
- Fails immediately at startup if required local weights are missing.
- Default output follows BTC specification: 'result/submission.jsonl'.
- Runs SubmissionLinter and exits non-zero if validation fails.
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

from config import (
    LOCAL_ASR_DIR, LOCAL_LLM_DIR, DEFAULT_ASR_MODEL, DEFAULT_LLM_MODEL,
    MAX_TIME_PER_SAMPLE
)
from src.part_a.submission_linter import SubmissionLinter


def run_offline_pipeline(
    data_dir: Path,
    output_path: Path,
    whisper_model_path: Path = LOCAL_ASR_DIR,
    llm_model_path: Path = LOCAL_LLM_DIR,
    max_samples: int | None = None,
    verbose: bool = True
):
    print("=" * 80)
    print("LIFEMATE AGENT — 100% OFFLINE INFERENCE RUNNER")
    print(f"Data directory:   {data_dir.resolve()}")
    print(f"Output file:      {output_path.resolve()}")
    print("=" * 80)

    # 1. Identify split name and files
    data_dir = Path(data_dir)
    split_name = data_dir.name
    jsonl_path = data_dir / f"{split_name}.jsonl"
    if not jsonl_path.is_file():
        candidates = list(data_dir.glob("*.jsonl"))
        if candidates:
            jsonl_path = candidates[0]
            split_name = jsonl_path.stem
        else:
            raise FileNotFoundError(f"No .jsonl input file found in {data_dir}")

    catalog_path = data_dir / f"{split_name}_tools_catalog.json"
    if not catalog_path.is_file():
        candidates = list(data_dir.glob("*catalog*.json"))
        if candidates:
            catalog_path = candidates[0]
        else:
            raise FileNotFoundError(f"No catalog JSON found in {data_dir}")

    # Add local split's zaic_tools to sys.path
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    from zaic_tools import ToolEnv

    # 2. Check offline weights existence (FAIL FAST if missing)
    whisper_model_path = Path(whisper_model_path)
    llm_model_path = Path(llm_model_path)

    if not whisper_model_path.is_dir():
        print(f"❌ FATAL: Local Whisper model not found at: {whisper_model_path.resolve()}")
        print("  Offline Docker runtime requires pre-downloaded weights. Exiting.")
        sys.exit(1)

    if not llm_model_path.is_dir():
        print(f"❌ FATAL: Local LLM model not found at: {llm_model_path.resolve()}")
        print("  Offline Docker runtime requires pre-downloaded weights. Exiting.")
        sys.exit(1)

    # 3. Load samples
    print(f"\n[1/4] Loading input samples from {jsonl_path}...")
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]
    print(f"  Total samples to process: {len(samples)}")

    # 4. Initialize ASR Module (Offline)
    print(f"\n[2/4] Initializing faster-whisper from '{whisper_model_path}'...")
    from src.part_b.asr_module import ASRModule
    asr = ASRModule(
        model_name=str(whisper_model_path.resolve()),
        device="cuda",
        compute_type="float16",
        beam_size=1,  # greedy candidate
    )
    try:
        asr.load()
        print("  ✅ faster-whisper loaded successfully on GPU.")
    except Exception as e:
        print(f"  ⚠️ faster-whisper GPU load failed ({e}), attempting CPU...")
        asr = ASRModule(
            model_name=str(whisper_model_path.resolve()),
            device="cpu",
            compute_type="int8",
            beam_size=1,
        )
        asr.load()

    # 5. Initialize Agent Module (Offline)
    print(f"\n[3/4] Initializing Agent LLM from '{llm_model_path}'...")
    from src.part_c.agent import create_agent
    agent = create_agent(
        mode="transformers",
        model_name=str(llm_model_path.resolve()),
        catalog_path=catalog_path,
        verbose=False,
    )
    agent.load()
    print("  ✅ Qwen3-4B-Instruct loaded successfully.")

    # 6. Execute inference loop
    print(f"\n[4/4] Executing inference pipeline across {len(samples)} samples...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.part_a.runner import sanitize_sample

    start_total = time.monotonic()
    results = []

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, raw_sample in enumerate(samples):
            sample_id = raw_sample["id"]
            sample_start = time.monotonic()
            deadline = sample_start + MAX_TIME_PER_SAMPLE
            
            env = None
            try:
                # Exactly one ToolEnv instance
                env = ToolEnv(split_name, sample_id, root=str(data_dir.parent))
                
                # ASR
                audio_rel = raw_sample.get("audio", f"audio/{sample_id}.wav")
                audio_file = data_dir / audio_rel
                transcript = asr(str(audio_file), raw_sample)
                
                # Agent
                clean_sample = sanitize_sample(raw_sample)
                agent.run(transcript, clean_sample, env, deadline=deadline)
                
                tool_calls = env.get_tool_calls()
                state_hash = env.get_state_hash()
                
            except Exception as e:
                print(f"  ✗ Error on sample {sample_id}: {e}")
                if env is not None:
                    try:
                        tool_calls = env.get_tool_calls()
                        state_hash = env.get_state_hash()
                    except Exception:
                        tool_calls = []
                        state_hash = ""
                else:
                    tool_calls = []
                    state_hash = ""

            entry = {
                "id": sample_id,
                "tool_calls": tool_calls,
                "state_hash": state_hash,
            }
            results.append(entry)
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()

            if verbose and (i + 1) % 25 == 0:
                elapsed = time.monotonic() - sample_start
                print(f"  [{i+1}/{len(samples)}] {sample_id} — {len(tool_calls)} calls, {elapsed:.2f}s")

    total_time = time.monotonic() - start_total
    print(f"\nInference completed in {total_time:.1f}s ({total_time/len(samples):.2f}s/sample)")

    # 7. Pre-flight verification with SubmissionLinter
    print("\nRunning Submission Integrity Linter...")
    linter = SubmissionLinter(split=split_name, jsonl_path=jsonl_path, catalog_path=catalog_path)
    report = linter.lint_file(output_path)
    linter.print_report(report)

    if not report["passed"]:
        print("❌ Submission linter FAILED. Exiting with error code 1.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Official LifeMate Offline Inference Runner")
    parser.add_argument("--data-dir", default="/data/public_test", help="Path to input split directory")
    parser.add_argument("--output", default="result/submission.jsonl", help="Path to output submission JSONL")
    parser.add_argument("--whisper-model", default=str(LOCAL_ASR_DIR))
    parser.add_argument("--llm-model", default=str(LOCAL_LLM_DIR))
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_offline_pipeline(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        whisper_model_path=Path(args.whisper_model),
        llm_model_path=Path(args.llm_model),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
