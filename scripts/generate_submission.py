from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
"""Generate submission.jsonl with real-time checkpointing and auto-resume."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import DATA_ROOT, SUBMISSIONS_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate submission file with auto-resume")
    parser.add_argument("--split", default="public_test", 
                        choices=["public_test", "private_test"])
    parser.add_argument("--agent-mode", default="rule",
                        choices=["rule", "transformers", "vllm"])
    parser.add_argument("--asr-mode", default="whisper",
                        choices=["gold", "whisper", "precomputed"])
    parser.add_argument("--transcripts-file", default=None,
                        help="Path to precomputed transcripts JSON")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Quantize LLM to 4-bit (bitsandbytes) to save VRAM on Colab T4")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Quantize LLM to 8-bit")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing output file if present (default: True)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Overwrite existing output file from scratch")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    split = args.split
    
    # Auto-switch to whisper on test splits if gold was erroneously requested
    if args.transcripts_file or args.asr_mode == "precomputed":
        args.asr_mode = "precomputed"
        if not args.transcripts_file:
            cand = Path(__file__).resolve().parent.parent / "outputs" / f"transcripts_{split}.json"
            if cand.exists():
                args.transcripts_file = str(cand)
                print(f"  ⚡ Found precomputed transcripts for {split}: {cand}")
    elif split in ("public_test", "private_test") and args.asr_mode == "gold":
        print(f"  ⚠️ Warning: {split} does not contain gold transcripts! Auto-switching --asr-mode to 'whisper'.")
        args.asr_mode = "whisper"
    
    # Add zaic_tools to path
    sys.path.insert(0, str(DATA_ROOT / split))
    from zaic_tools import ToolEnv
    
    # Load samples
    jsonl_path = DATA_ROOT / split / f"{split}.jsonl"
    print(f"Loading {split} samples from {jsonl_path}...")
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"  {len(samples)} samples loaded")
    
    # Output file determination
    if args.output:
        output_path = Path(args.output)
    else:
        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = SUBMISSIONS_DIR / f"submission_{split}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Google Drive persistence path
    drive_backup = Path("/content/drive/MyDrive/LifeMateAgent/submissions") / output_path.name
    
    # Checkpoint restore from Drive if local file was wiped
    if not output_path.exists() and drive_backup.exists() and args.resume:
        try:
            print(f"  📥 Restoring existing checkpoint from Google Drive: {drive_backup}")
            shutil.copy(drive_backup, output_path)
        except Exception as e:
            print(f"  ⚠️ Could not restore from Drive: {e}")
    
    # Check for existing completed samples
    completed_ids = set()
    results = []
    if output_path.exists() and args.resume:
        with open(output_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        entry = json.loads(line)
                        if "id" in entry and entry["id"] not in completed_ids:
                            results.append(entry)
                            completed_ids.add(entry["id"])
                    except json.JSONDecodeError:
                        print(f"  ⚠️ Ignored broken/partial line {line_no} from previous run in {output_path}")
        # Guard: If existing checkpoint has all empty calls (e.g. initial rule-based baseline), but we are running LLM:
        all_empty = (len(results) >= 20 and all(len(r.get("tool_calls", [])) == 0 for r in results))
        if all_empty and args.agent_mode != "rule":
            print(f"  ℹ️ Existing checkpoint ({len(results)} samples) is an empty baseline (0 tool calls).")
            print(f"     Automatically resetting to run fresh LLM inference for --agent-mode {args.agent_mode}...")
            completed_ids.clear()
            results.clear()
            if output_path.exists():
                output_path.unlink()
        elif completed_ids:
            # Re-write verified clean entries to guarantee valid JSONL without broken lines
            with open(output_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  ⚡ Auto-Resume: Found {len(completed_ids)} completed samples in {output_path}.")
            print(f"     Will skip them and process remaining {max(0, len(samples) - len(completed_ids))} samples...")
    
    # Check if already 100% finished
    remaining_samples = [s for s in samples if s["id"] not in completed_ids]
    if len(remaining_samples) == 0:
        print(f"\n🎉 All {len(samples)} samples are already completed in {output_path}!")
    else:
        # Create ASR
        from src.part_b.asr_module import create_asr
        asr = create_asr(
            mode=args.asr_mode,
            model_name=args.asr_model,
            transcripts_path=args.transcripts_file,
        )
        if args.asr_mode != "gold":
            asr.load()
        
        # Create Agent
        catalog_path = DATA_ROOT / split / f"{split}_tools_catalog.json"
        from src.part_c.agent import create_agent
        agent = create_agent(
            mode=args.agent_mode,
            model_name=args.model_name,
            catalog_path=catalog_path,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            verbose=args.verbose,
        )
        if hasattr(agent, 'load'):
            agent.load()
        
        # Run pipeline with real-time append streaming
        print(f"\nRunning pipeline: ASR({args.asr_mode}) → Agent({args.agent_mode})...")
        start = time.monotonic()
        
        file_mode = "a" if (output_path.exists() and args.resume and len(completed_ids) > 0) else "w"
        with open(output_path, file_mode, encoding="utf-8") as fout:
            for i, sample in enumerate(samples):
                sample_id = sample["id"]
                if sample_id in completed_ids:
                    continue
                
                env = None
                try:
                    # Create ToolEnv
                    env = ToolEnv(split, sample_id, root=str(DATA_ROOT))
                    
                    # ASR
                    audio_path = str(DATA_ROOT / split / sample["audio"])
                    transcript = asr(audio_path, sample)
                    
                    # Agent
                    agent.run(transcript, sample, env)
                    
                    entry = {
                        "id": sample_id,
                        "tool_calls": env.get_tool_calls(),
                        "state_hash": env.get_state_hash(),
                    }
                except Exception as e:
                    print(f"  ✗ {sample_id}: {e}")
                    tool_calls = env.get_tool_calls() if env is not None else []
                    state_hash = env.get_state_hash() if env is not None else ""
                    entry = {
                        "id": sample_id,
                        "tool_calls": tool_calls,
                        "state_hash": state_hash,
                    }
                finally:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except ImportError:
                        pass
                
                # Write immediately to disk (checkpointing every sample)
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()
                results.append(entry)
                completed_ids.add(sample_id)
                
                # Periodic log and Google Drive backup every 25 samples
                if len(results) % 25 == 0 or len(results) == len(samples):
                    elapsed = time.monotonic() - start
                    print(f"  [{len(results)}/{len(samples)}] {elapsed:.1f}s elapsed (checkpoint saved)")
                    if drive_backup.parent.exists():
                        try:
                            shutil.copy(output_path, drive_backup)
                        except Exception:
                            pass
        
        total_time = time.monotonic() - start
        print(f"\nDone: {len(results)} samples processed in {total_time:.1f}s "
              f"(avg {total_time/max(len(remaining_samples),1):.2f}s/new sample)")
    
    # Final backup to Drive
    if drive_backup.parent.exists():
        try:
            drive_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(output_path, drive_backup)
            print(f"💾 Checkpoint permanently synced to Google Drive: {drive_backup}")
        except Exception:
            pass
    
    # Validate format
    print("\nValidating submission format...")
    n_empty = sum(1 for r in results if not r["tool_calls"])
    n_with_calls = sum(1 for r in results if r["tool_calls"])
    n_with_hash = sum(1 for r in results if r["state_hash"])
    
    print(f"  Total samples: {len(results)}")
    print(f"  Empty (OOS):   {n_empty}")
    print(f"  With calls:    {n_with_calls}")
    print(f"  With hash:     {n_with_hash}")
    
    if n_with_hash < len(results):
        print(f"  ⚠️  {len(results) - n_with_hash} samples missing state_hash!")
    
    print(f"\n✅ Submission file ready: {output_path}")
    
    # Run official 7-layer Submission Linter
    try:
        from src.part_a.submission_linter import SubmissionLinter
        print("\n" + "=" * 60)
        print("RUNNING OFFICIAL 7-LAYER SUBMISSION LINTER")
        print("=" * 60)
        linter = SubmissionLinter(split=split)
        lint_report = linter.lint_file(output_path)
        linter.print_report(lint_report)
    except Exception as e:
        print(f"  [Notice] Could not run automatic linter: {e}")


if __name__ == "__main__":
    main()
