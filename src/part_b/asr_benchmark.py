"""Part B — ASR Benchmark: Evaluate ASR quality against gold transcripts.

Metrics:
- WER (Word Error Rate)
- CER (Character Error Rate)
- Entity-specific error analysis (names, numbers, times, addresses)
- Latency per sample
"""
import json
import time
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_JSONL, DATA_ROOT, OUTPUTS_DIR


def compute_wer(reference: str, hypothesis: str) -> dict:
    """Compute WER and CER between reference and hypothesis.
    
    Uses simple token-level edit distance (no external dependencies needed).
    Falls back to jiwer if available.
    """
    try:
        import jiwer
        wer = jiwer.wer(reference, hypothesis)
        cer = jiwer.cer(reference, hypothesis)
        return {"wer": wer, "cer": cer}
    except ImportError:
        pass
    
    # Fallback: simple WER/CER computation
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    wer_val = _edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)
    
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    cer_val = _edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)
    
    return {"wer": wer_val, "cer": cer_val}


def _edit_distance(a: list, b: list) -> int:
    """Compute edit distance between two sequences."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def analyze_entity_errors(reference: str, hypothesis: str) -> dict:
    """Analyze errors on specific entity types."""
    errors = {}
    
    # Phone numbers
    ref_phones = set(re.findall(r'0\d{9,10}', reference))
    hyp_phones = set(re.findall(r'0\d{9,10}', hypothesis))
    if ref_phones:
        errors["phone"] = {
            "ref": list(ref_phones), 
            "hyp": list(hyp_phones),
            "correct": len(ref_phones & hyp_phones),
            "total": len(ref_phones)
        }
    
    # Numbers/amounts
    ref_nums = set(re.findall(r'\d+', reference))
    hyp_nums = set(re.findall(r'\d+', hypothesis))
    if ref_nums:
        errors["numbers"] = {
            "correct": len(ref_nums & hyp_nums),
            "total": len(ref_nums),
            "missed": list(ref_nums - hyp_nums)[:5],
            "extra": list(hyp_nums - ref_nums)[:5],
        }
    
    # Time expressions (HH:MM patterns)
    ref_times = set(re.findall(r'\d{1,2}:\d{2}', reference))
    hyp_times = set(re.findall(r'\d{1,2}:\d{2}', hypothesis))
    if ref_times:
        errors["time"] = {
            "correct": len(ref_times & hyp_times),
            "total": len(ref_times)
        }
    
    # Negation words
    neg_words = {"không", "đừng", "chưa", "chẳng", "khỏi", "thôi", "bỏ", "hủy"}
    ref_negs = set(w.lower() for w in reference.split() if w.lower() in neg_words)
    hyp_negs = set(w.lower() for w in hypothesis.split() if w.lower() in neg_words)
    if ref_negs or hyp_negs:
        errors["negation"] = {
            "ref": list(ref_negs),
            "hyp": list(hyp_negs),
            "match": ref_negs == hyp_negs
        }
    
    # English words (code-switching)
    eng_pattern = r'\b[a-zA-Z]{2,}\b'
    ref_eng = set(re.findall(eng_pattern, reference))
    hyp_eng = set(re.findall(eng_pattern, hypothesis))
    if ref_eng:
        errors["english"] = {
            "ref": list(ref_eng),
            "hyp": list(hyp_eng),
            "correct": len(ref_eng & hyp_eng),
            "total": len(ref_eng)
        }
    
    return errors


def benchmark_asr(
    asr_fn,
    samples: list[dict] | None = None,
    max_samples: int | None = None,
    split: str = "train",
    verbose: bool = True
) -> dict:
    """Run ASR benchmark on training samples with gold transcripts.
    
    Args:
        asr_fn: Function(audio_path, sample) -> str (transcript)
        samples: List of samples (loaded from JSONL if None)
        max_samples: Limit number of samples
        split: Data split name
        verbose: Print progress
    
    Returns comprehensive benchmark results.
    """
    if samples is None:
        with open(TRAIN_JSONL, encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
    
    if max_samples:
        samples = samples[:max_samples]
    
    results = []
    total_wer = 0
    total_cer = 0
    latencies = []
    entity_error_counts = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for i, sample in enumerate(samples):
        gold_text = sample.get("audio_text", "")
        if not gold_text:
            continue
        
        audio_path = str(DATA_ROOT / split / sample["audio"])
        
        start = time.monotonic()
        try:
            pred_text = asr_fn(audio_path, sample)
        except Exception as e:
            pred_text = ""
            if verbose:
                print(f"  ✗ {sample['id']}: {e}")
        elapsed = time.monotonic() - start
        latencies.append(elapsed)
        
        # Compute metrics
        metrics = compute_wer(gold_text, pred_text)
        entity_errors = analyze_entity_errors(gold_text, pred_text)
        
        total_wer += metrics["wer"]
        total_cer += metrics["cer"]
        
        # Aggregate entity errors
        for etype, edata in entity_errors.items():
            if "correct" in edata and "total" in edata:
                entity_error_counts[etype]["correct"] += edata["correct"]
                entity_error_counts[etype]["total"] += edata["total"]
        
        results.append({
            "id": sample["id"],
            "gold": gold_text,
            "pred": pred_text,
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "latency": elapsed,
            "entity_errors": entity_errors,
        })
        
        if verbose and (i + 1) % 100 == 0:
            avg_wer = total_wer / (i + 1)
            print(f"  [{i+1}/{len(samples)}] avg WER: {avg_wer:.3f}")
    
    n = len(results)
    
    # Worst samples by WER
    worst = sorted(results, key=lambda x: x["wer"], reverse=True)[:20]
    
    summary = {
        "n_samples": n,
        "avg_wer": total_wer / n if n > 0 else 0,
        "avg_cer": total_cer / n if n > 0 else 0,
        "latency": {
            "avg": sum(latencies) / len(latencies) if latencies else 0,
            "p50": sorted(latencies)[len(latencies)//2] if latencies else 0,
            "p95": sorted(latencies)[int(len(latencies)*0.95)] if latencies else 0,
            "max": max(latencies) if latencies else 0,
        },
        "entity_accuracy": {
            k: v["correct"] / max(v["total"], 1) 
            for k, v in entity_error_counts.items()
        },
        "entity_counts": dict(entity_error_counts),
        "worst_samples": [{
            "id": w["id"], 
            "wer": w["wer"],
            "gold": w["gold"][:100],
            "pred": w["pred"][:100]
        } for w in worst[:10]],
        "results": results,
    }
    
    return summary


def print_benchmark_report(summary: dict):
    """Print a formatted benchmark report."""
    print("=" * 70)
    print("ASR BENCHMARK REPORT")
    print("=" * 70)
    
    print(f"\nSamples: {summary['n_samples']}")
    print(f"Avg WER: {summary['avg_wer']:.4f} ({summary['avg_wer']*100:.2f}%)")
    print(f"Avg CER: {summary['avg_cer']:.4f} ({summary['avg_cer']*100:.2f}%)")
    
    lat = summary["latency"]
    print(f"\nLatency: avg={lat['avg']:.3f}s, p50={lat['p50']:.3f}s, "
          f"p95={lat['p95']:.3f}s, max={lat['max']:.3f}s")
    
    print(f"\nEntity Accuracy:")
    for etype, acc in summary["entity_accuracy"].items():
        counts = summary["entity_counts"][etype]
        print(f"  {etype:15s}: {acc:.2%} ({counts['correct']}/{counts['total']})")
    
    print(f"\nWorst Samples (by WER):")
    for w in summary["worst_samples"][:10]:
        print(f"  {w['id']}: WER={w['wer']:.3f}")
        print(f"    Gold: {w['gold']}")
        print(f"    Pred: {w['pred']}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Quick test with gold transcripts (WER should be 0)
    print("Testing ASR benchmark with gold transcripts (WER should be 0)...\n")
    
    from src.part_b.asr_module import GoldTranscriptASR
    
    asr = GoldTranscriptASR()
    summary = benchmark_asr(asr, max_samples=50, verbose=True)
    print_benchmark_report(summary)
