"""Run baseline pipeline — gold transcript replay + evaluation.

This validates:
1. ToolEnv works correctly (gold replay should give perfect hash match)
2. Our evaluator matches official scoring
3. Pipeline can run end-to-end
"""
import sys
import json
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import DATA_ROOT, TRAIN_JSONL, SUBMISSIONS_DIR, OUTPUTS_DIR


def main():
    print("=" * 80)
    print("LIFEMATE — BASELINE PIPELINE")
    print("=" * 80)
    
    split = "train"
    
    # Add zaic_tools to path
    sys.path.insert(0, str(DATA_ROOT / split))
    from zaic_tools import ToolEnv
    
    # Load training data
    print("\n[1/5] Loading training data...")
    samples = []
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples")
    
    # Step 2: Gold replay baseline
    print("\n[2/5] Running gold replay baseline (perfect score test)...")
    start = time.monotonic()
    
    results = []
    errors = []
    
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        try:
            env = ToolEnv(split, sample_id, root=str(DATA_ROOT))
            
            # Replay gold tool calls
            gold_calls = sample.get("label", {}).get("tool_calls", [])
            for tc in gold_calls:
                env.call(tc["name"], tc.get("arguments", {}))
            
            results.append({
                "id": sample_id,
                "tool_calls": env.get_tool_calls(),
                "state_hash": env.get_state_hash(),
            })
        except Exception as e:
            errors.append({"id": sample_id, "error": str(e)})
            results.append({
                "id": sample_id,
                "tool_calls": [],
                "state_hash": "",
            })
        
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(samples)}] processed...")
    
    elapsed = time.monotonic() - start
    print(f"  Gold replay done in {elapsed:.1f}s ({len(errors)} errors)")
    if errors:
        for e in errors[:5]:
            print(f"    ✗ {e['id']}: {e['error']}")
    
    # Step 3: Save submission
    print("\n[3/5] Saving submission...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBMISSIONS_DIR / "baseline_gold_replay.jsonl"
    with open(sub_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved to {sub_path}")
    
    # Step 4: Compute gold hashes for evaluation
    print("\n[4/5] Computing gold state hashes for evaluation...")
    gold_hashes = {}
    for i, sample in enumerate(samples):
        sample_id = sample["id"]
        try:
            env = ToolEnv(split, sample_id, root=str(DATA_ROOT))
            gold_calls = sample.get("label", {}).get("tool_calls", [])
            for tc in gold_calls:
                env.call(tc["name"], tc.get("arguments", {}))
            gold_hashes[sample_id] = env.get_state_hash()
        except Exception:
            gold_hashes[sample_id] = ""
        
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(samples)}]...")
    
    # Step 5: Evaluate
    print("\n[5/5] Evaluating...")
    from src.part_a.evaluator import load_catalog_fuzzy_params, evaluate_submission
    
    fuzzy_params = load_catalog_fuzzy_params()
    eval_result = evaluate_submission(results, samples, fuzzy_params, gold_hashes)
    
    print(f"\n{'=' * 60}")
    print(f"  BASELINE RESULTS (Gold Replay)")
    print(f"{'=' * 60}")
    print(f"  Final Score:      {eval_result['final_score']:.2f}/100")
    print(f"  Avg Score:        {eval_result['avg_score']:.4f}")
    print(f"  Hash Match Rate:  {eval_result['hash_match_rate']:.2%}")
    print(f"  N Samples:        {eval_result['n_samples']}")
    print(f"  Type Counts:      {eval_result['type_counts']}")
    print(f"{'=' * 60}")
    
    if eval_result['final_score'] < 99.0:
        print("\n⚠️  WARNING: Gold replay should score ~100. Check for issues!")
        # Print some failing samples
        failing = [r for r in eval_result['results'] if r['score'] < 1.0]
        print(f"  {len(failing)} samples scored < 1.0:")
        for f_sample in failing[:10]:
            print(f"    {f_sample['id']}: score={f_sample['score']:.3f}, "
                  f"hash_match={f_sample.get('hash_match')}, "
                  f"M={f_sample.get('M')}, U={f_sample.get('U')}, G={f_sample.get('G')}")
    else:
        print("\n✅ Gold replay baseline validated! Evaluator working correctly.")
    
    # Save evaluation results
    eval_path = OUTPUTS_DIR / "baseline_evaluation.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    # Don't save all per-sample results (too large)
    eval_summary = {k: v for k, v in eval_result.items() if k != "results"}
    eval_summary["n_perfect"] = sum(1 for r in eval_result["results"] if r["score"] >= 1.0)
    eval_summary["n_failing"] = sum(1 for r in eval_result["results"] if r["score"] < 1.0)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation summary saved to {eval_path}")


if __name__ == "__main__":
    main()
