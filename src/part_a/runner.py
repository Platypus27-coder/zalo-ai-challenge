"""Part A — Pipeline Runner: Orchestrates ASR → Agent → ToolEnv → Submission.

Features:
- Exactly one ToolEnv per sample (contract enforcement).
- Monotonic shared deadline computation and propagation.
- Sample sanitization (removes gold labels/final_state before passing to Agent).
- Faithful finalization: retains trace and real state_hash from existing env on error.
"""
import copy
import inspect
import json
import time
import unicodedata
from pathlib import Path
from typing import Callable, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    DATA_ROOT, TRAIN_JSONL, PUBLIC_TEST_JSONL,
    TRAIN_CATALOG, PUBLIC_TEST_CATALOG,
    MAX_TIME_PER_SAMPLE, SUBMISSIONS_DIR, SPLITS_DIR
)


def load_samples(split: str = "train") -> list[dict]:
    """Load samples from JSONL, supporting train, dev, holdout, and public_test."""
    if split in ("train", "dev", "holdout"):
        jsonl_path = TRAIN_JSONL
    else:
        jsonl_path = PUBLIC_TEST_JSONL

    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if split in ("dev", "holdout"):
        splits_file = SPLITS_DIR / "splits.json"
        if splits_file.exists():
            with open(splits_file, encoding="utf-8") as f:
                split_ids = set(json.load(f)["splits"].get(split, []))
            samples = [s for s in samples if s["id"] in split_ids]

    return samples


def sanitize_sample(sample: dict) -> dict:
    """Sanitize sample to ensure agent cannot accidentally read gold labels or final states."""
    sanitized = copy.deepcopy(sample)
    for forbidden_key in ("label", "gold_calls", "gold_state", "final_state", "answer"):
        sanitized.pop(forbidden_key, None)
    return sanitized


class PipelineRunner:
    """Orchestrates the full inference pipeline for all samples."""
    
    def __init__(
        self,
        split: str = "train",
        asr_fn: Callable | None = None,
        agent_fn: Callable | None = None,
        sample_ids: list[str] | None = None,
        max_time: float = MAX_TIME_PER_SAMPLE,
        data_root: Path = DATA_ROOT,
        verbose: bool = True
    ):
        self.split = split
        self.asr_fn = asr_fn
        self.agent_fn = agent_fn
        self.sample_ids = set(sample_ids) if sample_ids else None
        self.max_time = max_time
        self.data_root = Path(data_root)
        self.verbose = verbose
        
        self.data_split = "train" if split in ("train", "dev", "holdout") else split

        # Add zaic_tools to sys.path
        split_dir = self.data_root / self.data_split
        if str(split_dir) not in sys.path:
            sys.path.insert(0, str(split_dir))
    
    def run(self, samples: list[dict] | None = None) -> list[dict]:
        """Run pipeline on all samples with strict single-env enforcement."""
        from zaic_tools import ToolEnv
        
        if samples is None:
            samples = load_samples(self.split)
        
        if self.sample_ids:
            samples = [s for s in samples if s["id"] in self.sample_ids]
        
        results = []
        errors = []
        latencies = []
        
        # Check if agent_fn accepts deadline keyword argument
        supports_deadline = False
        if self.agent_fn:
            try:
                sig = inspect.signature(self.agent_fn)
                supports_deadline = "deadline" in sig.parameters
            except Exception:
                pass
        
        for i, raw_sample in enumerate(samples):
            sample_id = raw_sample["id"]
            start_time = time.monotonic()
            deadline = start_time + self.max_time
            
            env = None
            try:
                # Exactly one ToolEnv created per sample
                env = ToolEnv(self.data_split, sample_id, root=str(self.data_root))
                
                # Step 1: ASR Transcription
                if self.asr_fn:
                    audio_rel_path = raw_sample.get("audio", f"audio/{sample_id}.wav")
                    audio_path = self.data_root / self.data_split / audio_rel_path
                    transcript = self.asr_fn(str(audio_path), raw_sample)
                else:
                    transcript = raw_sample.get("audio_text", "")
                
                # Step 2: Sanitize sample to prevent gold leakage
                clean_sample = sanitize_sample(raw_sample)
                
                # Step 3: Agent Execution
                if self.agent_fn:
                    if supports_deadline:
                        self.agent_fn(transcript, clean_sample, env, deadline=deadline)
                    else:
                        self.agent_fn(transcript, clean_sample, env)
                
                # Step 4: Collect faithful results directly from env
                tool_calls = env.get_tool_calls()
                state_hash = env.get_state_hash()
                
                elapsed = time.monotonic() - start_time
                latencies.append(elapsed)
                
                results.append({
                    "id": sample_id,
                    "tool_calls": tool_calls,
                    "state_hash": state_hash,
                })
                
                if self.verbose and (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(samples)}] {sample_id} — {len(tool_calls)} calls, {elapsed:.2f}s")
                
            except Exception as e:
                elapsed = time.monotonic() - start_time
                errors.append({"id": sample_id, "error": str(e), "time": elapsed})
                
                # Faithful finalization: retrieve whatever calls and state_hash from existing env
                if env is not None:
                    try:
                        fallback_calls = env.get_tool_calls()
                        fallback_hash = env.get_state_hash()
                    except Exception:
                        fallback_calls = []
                        fallback_hash = ""
                else:
                    fallback_calls = []
                    fallback_hash = ""
                
                results.append({
                    "id": sample_id,
                    "tool_calls": fallback_calls,
                    "state_hash": fallback_hash,
                })
                
                if self.verbose:
                    print(f"  ✗ {sample_id}: {e}")
        
        if self.verbose:
            print(f"\nCompleted: {len(results)} samples")
            if latencies:
                latencies.sort()
                p50 = latencies[len(latencies) // 2]
                p95 = latencies[int(len(latencies) * 0.95)]
                max_lat = latencies[-1]
                print(f"Latency: p50={p50:.2f}s, p95={p95:.2f}s, max={max_lat:.2f}s")
            if errors:
                print(f"Errors encountered: {len(errors)}")
        
        return results
    
    def save_submission(self, results: list[dict], output_path: Path | None = None) -> Path:
        """Save results as valid UTF-8 JSONL submission."""
        if output_path is None:
            SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = SUBMISSIONS_DIR / f"submission_{self.split}_{timestamp}.jsonl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        if self.verbose:
            print(f"Saved submission to: {output_path.resolve()}")
        return output_path
