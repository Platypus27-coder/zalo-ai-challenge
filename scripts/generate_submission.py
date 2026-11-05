"""Generate submission.jsonl for public_test or private_test."""
import sys
import json
import time
import argparse
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
    parser = argparse.ArgumentParser(description="Generate submission file")
    parser.add_argument("--split", default="public_test", 
                        choices=["public_test", "private_test"])
    parser.add_argument("--agent-mode", default="rule",
                        choices=["rule", "transformers", "vllm"])
    parser.add_argument("--asr-mode", default="gold",
                        choices=["gold", "whisper"])
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    split = args.split
    
    # Auto-switch to whisper on test splits if gold was erroneously requested
    if split in ("public_test", "private_test") and args.asr_mode == "gold":
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
    
    # Create ASR
    from src.part_b.asr_module import create_asr
    asr = create_asr(
        mode=args.asr_mode,
        model_name=args.asr_model,
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
        verbose=args.verbose,
    )
    if hasattr(agent, 'load'):
        agent.load()
    
    # Run pipeline
    print(f"\nRunning pipeline: ASR({args.asr_mode}) → Agent({args.agent_mode})...")
    results = []
    start = time.monotonic()
    
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        try:
            # Create ToolEnv
            env = ToolEnv(split, sample_id, root=str(DATA_ROOT))
            
            # ASR
            audio_path = str(DATA_ROOT / split / sample["audio"])
            transcript = asr(audio_path, sample)
            
            # Agent
            agent.run(transcript, sample, env)
            
            results.append({
                "id": sample_id,
                "tool_calls": env.get_tool_calls(),
                "state_hash": env.get_state_hash(),
            })
        except Exception as e:
            print(f"  ✗ {sample_id}: {e}")
            results.append({
                "id": sample_id,
                "tool_calls": [],
                "state_hash": "",
            })
        
        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - start
            print(f"  [{i+1}/{len(samples)}] {elapsed:.1f}s elapsed")
    
    total_time = time.monotonic() - start
    print(f"\nDone: {len(results)} samples in {total_time:.1f}s "
          f"(avg {total_time/max(len(results),1):.2f}s/sample)")
    
    # Save submission
    if args.output:
        output_path = Path(args.output)
    else:
        SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = SUBMISSIONS_DIR / f"submission_{split}_{timestamp}.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"Submission saved: {output_path} ({len(results)} samples)")
    
    # Validate format
    print("\nValidating submission format...")
    n_empty = sum(1 for r in results if not r["tool_calls"])
    n_with_calls = sum(1 for r in results if r["tool_calls"])
    n_with_hash = sum(1 for r in results if r["state_hash"])
    
    print(f"  Empty (OOS): {n_empty}")
    print(f"  With calls:  {n_with_calls}")
    print(f"  With hash:   {n_with_hash}")
    
    if n_with_hash < len(results):
        print(f"  ⚠️  {len(results) - n_with_hash} samples missing state_hash!")
    
    print(f"\n✅ Submission file ready: {output_path}")
    
    # Run official 7-layer Submission Linter
    try:
        from src.part_a.submission_linter import SubmissionLinter
        print("\n" + "=" * 60)
        print("RUNNING OFFICIAL 7-LAYER SUBMISSION LINTER")
        print("=" * 60)
        linter = SubmissionLinter(
            submission_path=output_path,
            catalog_path=catalog_path,
            input_jsonl=jsonl_path,
            split=split,
        )
        lint_report = linter.lint()
        if lint_report["errors"]:
            print(f"❌ Submission Linter found {len(lint_report['errors'])} critical errors:")
            for err in lint_report["errors"][:5]:
                print(f"  - {err}")
        else:
            print("🎉 0 Critical Errors! Submission is 100% compliant with Zalo AI Challenge rules.")
    except Exception as e:
        print(f"  [Notice] Could not run automatic linter: {e}")


if __name__ == "__main__":
    main()
