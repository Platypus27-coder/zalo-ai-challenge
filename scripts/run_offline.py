"""Part A — Official Offline Inference Entrypoint for Docker.

Entrypoint script executed by the competition evaluation platform:
- Operates 100% offline (no internet access)
- Loads local model weights (faster-whisper + Qwen2.5-3B-Instruct)
- Processes all test samples through the LifeMate pipeline
- Runs Submission Linter pre-flight verification before final export

Usage:
    python scripts/run_offline.py --data-dir /data/public_test --output /output/submission.jsonl
"""
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def run_offline_pipeline(
    data_dir: Path,
    output_path: Path,
    whisper_model_path: str = "models/whisper/large-v3-turbo",
    llm_model_path: str = "models/llm/Qwen_Qwen2.5-3B-Instruct",
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
    if not jsonl_path.exists():
        # Try finding any *.jsonl in data_dir
        candidates = list(data_dir.glob("*.jsonl"))
        if candidates:
            jsonl_path = candidates[0]
            split_name = jsonl_path.stem
        else:
            raise FileNotFoundError(f"No .jsonl file found in {data_dir}")

    catalog_path = data_dir / f"{split_name}_tools_catalog.json"
    if not catalog_path.exists():
        candidates = list(data_dir.glob("*catalog*.json"))
        if candidates:
            catalog_path = candidates[0]

    # Add local split's zaic_tools to sys.path
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    from zaic_tools import ToolEnv

    # 2. Load samples
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

    # 3. Initialize ASR Module (Offline)
    print(f"\n[2/4] Initializing ASR Module from '{whisper_model_path}'...")
    from src.part_b.asr_module import ASRModule
    whisper_path_obj = Path(whisper_model_path)
    asr_load_path = str(whisper_path_obj.resolve()) if whisper_path_obj.exists() else "large-v3-turbo"
    asr = ASRModule(model_name=asr_load_path, device="cuda", compute_type="float16")
    try:
        asr.load()
        print("  ✅ faster-whisper loaded successfully on GPU.")
    except Exception as e:
        print(f"  ⚠️ faster-whisper GPU load failed ({e}), attempting CPU fallback...")
        asr = ASRModule(model_name=asr_load_path, device="cpu", compute_type="int8")
        asr.load()

    # 4. Initialize Agent Module (Offline with Fail-Fast)
    print(f"\n[3/4] Initializing Agent Module from '{llm_model_path}'...")
    from src.part_c.agent import create_agent
    llm_path_obj = Path(llm_model_path)
    llm_load_path = str(llm_path_obj.resolve()) if llm_path_obj.exists() else "Qwen/Qwen3-4B-Instruct-2507"
    
    # In competition offline mode, silent fallback to RuleAgent is strictly disabled
    try:
        agent = create_agent(
            mode="transformers",
            model_name=llm_load_path,
            catalog_path=catalog_path,
            verbose=False
        )
        if hasattr(agent, "load"):
            agent.load()
        print(f"  ✅ LLM Agent initialized from {llm_load_path}")
    except Exception as e:
        raise RuntimeError(
            f"FATAL: Offline LLM Agent failed to load from '{llm_load_path}': {e}\n"
            f"Per Member C audit: silent fallback to Rule Agent is prohibited to prevent 0-score disaster.\n"
            f"Ensure model weights are pre-downloaded via `python scripts/download_models.py`."
        ) from e

    # 5. Execute inference loop
    print(f"\n[4/4] Executing inference pipeline...")
    results = []
    latencies = []
    errors = []
    start_total = time.monotonic()

    # Parent directory of split (needed by ToolEnv)
    data_root = data_dir.parent

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample["id"]
        t0 = time.monotonic()

        try:
            # Instantiate single ToolEnv session
            env = ToolEnv(split_name, sample_id, root=str(data_root))

            # Transcribe audio
            audio_rel = sample.get("audio", f"audio/{sample_id}/{sample_id}.wav")
            audio_full_path = data_dir / audio_rel
            if not audio_full_path.exists():
                # Try finding audio in standard location
                audio_full_path = data_dir / "audio" / sample_id / f"{sample_id}.wav"

            res = asr.transcribe(str(audio_full_path), context=sample.get("context"))
            transcript = res.get("normalized_transcript") or res.get("raw_transcript", "")

            # Run Agent
            agent.run(transcript, sample, env)

            tool_calls = env.get_tool_calls()
            state_hash = env.get_state_hash()
            elapsed = time.monotonic() - t0
            latencies.append(elapsed)

            results.append({
                "id": sample_id,
                "tool_calls": tool_calls,
                "state_hash": state_hash
            })

            if verbose and (idx % 25 == 0 or idx == len(samples)):
                avg_t = sum(latencies) / len(latencies)
                print(f"  [{idx}/{len(samples)}] {sample_id} ({len(tool_calls)} calls, {elapsed:.2f}s, avg {avg_t:.2f}s)")

        except Exception as e:
            elapsed = time.monotonic() - t0
            errors.append({"id": sample_id, "error": str(e)})
            results.append({
                "id": sample_id,
                "tool_calls": [],
                "state_hash": ""
            })
            if verbose:
                print(f"  ❌ [{idx}/{len(samples)}] {sample_id} FAILED: {e}")

    total_time = time.monotonic() - start_total
    print(f"\nPipeline finished: {len(results)} samples processed in {total_time:.2f}s.")
    if latencies:
        latencies.sort()
        print(f"Latency distribution: P50={latencies[len(latencies)//2]:.2f}s, "
              f"P95={latencies[int(len(latencies)*0.95)]:.2f}s, "
              f"Max={latencies[-1]:.2f}s")

    # 6. Save output JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved submission to: {output_path}")

    # 7. Run Submission Linter on output
    print("\nRunning Submission Linter verification...")
    from src.part_a.submission_linter import SubmissionLinter
    linter = SubmissionLinter(split=split_name)
    lint_res = linter.lint_file(output_path)
    if lint_res["passed"]:
        print("  🎉 SUBMISSION LINTER PASSED (0 Critical Errors)!")
    else:
        print(f"  ❌ CRITICAL: Submission Linter FAILED ({len(lint_res['critical_errors'])} errors):")
        for err in lint_res["critical_errors"][:5]:
            print(f"     • {err}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Official LifeMate Offline Inference Runner")
    parser.add_argument("--data-dir", required=True, help="Directory containing <split>.jsonl, catalog, and audio/")
    parser.add_argument("--output", default="submission.jsonl", help="Output submission.jsonl path")
    parser.add_argument("--whisper-model", default="models/whisper/large-v3-turbo")
    parser.add_argument("--llm-model", default="models/llm/Qwen_Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    run_offline_pipeline(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        whisper_model_path=args.whisper_model,
        llm_model_path=args.llm_model,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
