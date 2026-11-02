"""Part A — Pipeline Runner: Orchestrates ASR → Agent → ToolEnv → Submission.

This is the main execution loop that:
1. Loads samples from JSONL
2. Runs ASR on audio (via Part B module)
3. Runs Agent loop with validator (via Part C module)
4. Collects tool_calls and state_hash
5. Generates submission.jsonl
6. Optionally evaluates against gold labels
"""
import json
import time
import unicodedata
from pathlib import Path
from typing import Callable, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    DATA_ROOT, TRAIN_JSONL, PUBLIC_TEST_JSONL,
    TRAIN_CATALOG, PUBLIC_TEST_CATALOG,
    MAX_TIME_PER_SAMPLE, SUBMISSIONS_DIR
)


def load_samples(split: str = "train") -> list[dict]:
    """Load samples from JSONL."""
    jsonl_path = TRAIN_JSONL if split == "train" else PUBLIC_TEST_JSONL
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def get_installed_tools(sample: dict) -> set[str]:
    """Extract all tool names from installed_apps."""
    tools = set()
    for app_data in sample.get("context", {}).get("installed_apps", {}).values():
        tools.update(app_data.get("tools", []))
    tools.add("sys.ask_missing_info")
    return tools


def normalize_nfc(text: str) -> str:
    """Normalize text to NFC (for fuzzy params from ASR/LLM output)."""
    if isinstance(text, str):
        return unicodedata.normalize("NFC", text)
    return text


def normalize_arguments(arguments: dict) -> dict:
    """NFC-normalize string values in arguments (but preserve IDs from tool results)."""
    normalized = {}
    for key, value in arguments.items():
        if isinstance(value, str):
            normalized[key] = normalize_nfc(value)
        else:
            normalized[key] = value
    return normalized


class PipelineRunner:
    """Orchestrates the full inference pipeline for all samples."""
    
    def __init__(
        self,
        split: str = "train",
        asr_fn: Callable | None = None,
        agent_fn: Callable | None = None,
        sample_ids: list[str] | None = None,
        max_time: float = MAX_TIME_PER_SAMPLE,
        verbose: bool = True
    ):
        """
        Args:
            split: "train" or "public_test"
            asr_fn: Function(audio_path, sample) -> str (transcript)
            agent_fn: Function(transcript, sample, env, validator) -> None 
                      (should call env.call() internally)
            sample_ids: If provided, only run these sample IDs
            max_time: Maximum seconds per sample
            verbose: Print progress
        """
        self.split = split
        self.asr_fn = asr_fn
        self.agent_fn = agent_fn
        self.sample_ids = set(sample_ids) if sample_ids else None
        self.max_time = max_time
        self.verbose = verbose
        
        # Add zaic_tools to path
        split_dir = DATA_ROOT / split
        if str(split_dir) not in sys.path:
            sys.path.insert(0, str(split_dir))
    
    def run(self, samples: list[dict] | None = None) -> list[dict]:
        """Run pipeline on all samples. Returns list of submission entries."""
        from zaic_tools import ToolEnv
        
        if samples is None:
            samples = load_samples(self.split)
        
        # Filter by sample_ids if specified
        if self.sample_ids:
            samples = [s for s in samples if s["id"] in self.sample_ids]
        
        results = []
        errors = []
        latencies = []
        
        for i, sample in enumerate(samples):
            sample_id = sample["id"]
            start_time = time.monotonic()
            
            try:
                # Create fresh ToolEnv for this sample
                env = ToolEnv(self.split, sample_id, root=str(DATA_ROOT))
                
                # Step 1: ASR
                if self.asr_fn:
                    audio_path = DATA_ROOT / self.split / sample["audio"]
                    transcript = self.asr_fn(str(audio_path), sample)
                else:
                    # Fallback: use gold transcript (for baseline/debugging)
                    transcript = sample.get("audio_text", "")
                
                # Step 2: Agent loop
                if self.agent_fn:
                    self.agent_fn(transcript, sample, env)
                
                # Step 3: Collect results
                tool_calls = env.get_tool_calls()
                state_hash = env.get_state_hash()
                
                elapsed = time.monotonic() - start_time
                latencies.append(elapsed)
                
                result = {
                    "id": sample_id,
                    "tool_calls": tool_calls,
                    "state_hash": state_hash,
                }
                results.append(result)
                
                if self.verbose and (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(samples)}] {sample_id} — "
                          f"{len(tool_calls)} calls, {elapsed:.2f}s")
                
            except Exception as e:
                elapsed = time.monotonic() - start_time
                errors.append({"id": sample_id, "error": str(e), "time": elapsed})
                
                # Still need to output something
                results.append({
                    "id": sample_id,
                    "tool_calls": [],
                    "state_hash": "",
                })
                
                if self.verbose:
                    print(f"  ✗ {sample_id}: {e}")
        
        if self.verbose:
            print(f"\nCompleted: {len(results)} samples")
            if latencies:
                latencies.sort()
                print(f"Latency: p50={latencies[len(latencies)//2]:.2f}s, "
                      f"p95={latencies[int(len(latencies)*0.95)]:.2f}s, "
                      f"max={latencies[-1]:.2f}s")
            if errors:
                print(f"Errors: {len(errors)}")
        
        return results
    
    def save_submission(self, results: list[dict], output_path: Path | None = None) -> Path:
        """Save results as submission JSONL."""
        if output_path is None:
            SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = SUBMISSIONS_DIR / f"submission_{self.split}_{timestamp}.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Submission saved: {output_path} ({len(results)} samples)")
        return output_path


def run_gold_baseline(split: str = "train", sample_ids: list[str] | None = None) -> list[dict]:
    """Baseline: use gold transcript + replay gold tool calls.
    
    This should produce a perfect score and validates our ToolEnv + evaluator setup.
    """
    from zaic_tools import ToolEnv
    
    samples = load_samples(split)
    if sample_ids:
        sample_ids_set = set(sample_ids)
        samples = [s for s in samples if s["id"] in sample_ids_set]
    
    results = []
    
    for sample in samples:
        sample_id = sample["id"]
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
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LifeMate Pipeline Runner")
    parser.add_argument("--split", default="train", choices=["train", "public_test"])
    parser.add_argument("--gold-baseline", action="store_true", help="Run gold replay baseline")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()
    
    if args.gold_baseline:
        print(f"Running gold baseline on {args.split}...")
        samples = load_samples(args.split)
        if args.max_samples:
            samples = samples[:args.max_samples]
        
        results = run_gold_baseline(args.split, [s["id"] for s in samples])
        
        runner = PipelineRunner(split=args.split)
        output_path = runner.save_submission(results)
        
        # Evaluate if on train split
        if args.split == "train":
            from src.part_a.evaluator import (
                load_catalog_fuzzy_params, evaluate_submission,
                compute_gold_hashes_via_replay
            )
            
            print("\nEvaluating gold baseline (should be ~100)...")
            fuzzy_params = load_catalog_fuzzy_params()
            gold_hashes = compute_gold_hashes_via_replay(samples, args.split)
            eval_result = evaluate_submission(results, samples, fuzzy_params, gold_hashes)
            print(f"Final Score: {eval_result['final_score']:.2f}")
            print(f"Hash Match Rate: {eval_result['hash_match_rate']:.2%}")
            print(f"Type Counts: {eval_result['type_counts']}")
