"""Diagnostic tool: Extract, analyze, and classify Negation errors on Dev set.

Usage:
    python scripts/diagnose_negation_dev.py [transcripts_path]

Default transcripts path: outputs/transcripts_dev.json (or Drive path if on Colab)
"""
import json
import sys
import os
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.part_b.normalizer import NEGATION_WORDS, normalize_transcript

DIALECT_NEGATIONS = {"hổng", "chả", "chớ", "thôi khỏi", "khỏi cần", "đâu có", "chẳng"}


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds using standard wave module."""
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def classify_negation_error(gold: str, pred: str, missing: set, duration: float, history_len: int) -> tuple[str, str]:
    """Classify the root cause of missing negation."""
    gold_words = [w.strip(".,?!;:") for w in gold.lower().split()]
    pred_words = [w.strip(".,?!;:") for w in pred.lower().split()]
    
    # N0: Audio is very short (< 1.5s) or very few words in history response -> VAD clipped
    if duration > 0 and duration < 1.5:
        return "N0", f"Audio cực ngắn ({duration:.2f}s) -> VAD cắt sạch"
    if history_len > 0 and len(gold_words) <= 4:
        return "N0", f"Câu đáp ngắn trong history ({len(gold_words)} từ) -> VAD cắt"
    
    # N3: Regional dialect or slang negation
    for m in missing:
        if m in DIALECT_NEGATIONS or any(d in gold.lower() for d in DIALECT_NEGATIONS):
            return "N3", f"Từ phủ định miền/khẩu ngữ ({m})"
            
    # N1: Negation word at start or end of utterance
    for m in missing:
        if gold_words and (gold_words[0] == m or gold_words[-1] == m):
            return "N1", f"Từ phủ định ở mép câu (đầu/cuối: '{m}') -> VAD speech_pad_ms xén mép"
            
    # N2: Phonetic confusion (pred has similar length or phonetically close word)
    if len(pred_words) >= len(gold_words) * 0.8:
        return "N2", f"Whisper giải mã ép âm/nhầm từ khác"
        
    # N4: Fast speech / noise / general drop
    return "N4", f"Âm thanh nhiễu hoặc người dùng nói lướt"


def diagnose_negation(transcripts_file: Path | None = None) -> dict:
    train_jsonl = PROJECT_ROOT / "train+public_test" / "train" / "train.jsonl"
    dev_ids_file = PROJECT_ROOT / "outputs" / "splits" / "dev_ids.json"
    
    if transcripts_file is None:
        transcripts_file = PROJECT_ROOT / "outputs" / "transcripts_dev.json"
        
    print(f"=== Lifemate Agent — Negation Error Diagnostic ===")
    print(f"Train data:       {train_jsonl}")
    print(f"Dev IDs:          {dev_ids_file}")
    print(f"Transcripts file: {transcripts_file}")
    
    if not train_jsonl.exists():
        print(f"❌ Error: {train_jsonl} not found!")
        return {}
        
    # Load gold train data
    all_samples = {}
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                all_samples[sample["id"]] = sample
                
    # Load Dev IDs
    if dev_ids_file.exists():
        with open(dev_ids_file, encoding="utf-8") as f:
            dev_ids = json.load(f)
    else:
        print(f"⚠️ Warning: dev_ids.json not found, using all train samples with negation!")
        dev_ids = list(all_samples.keys())
        
    # Load transcripts
    if not transcripts_file.exists():
        print(f"⚠️ Notice: Transcript file {transcripts_file} does not exist locally yet.")
        print(f"   (If you just finished running on Google Colab, copy 'transcripts_dev.json' to 'outputs/' first).")
        print(f"\nListing gold samples on Dev that contain Negation for reference:")
        neg_samples = []
        for sid in dev_ids:
            s = all_samples.get(sid, {})
            text = s.get("audio_text", "")
            negs = [w for w in text.lower().split() if w.strip(".,?!;:") in NEGATION_WORDS]
            if negs:
                neg_samples.append((sid, text, negs))
        print(f"Total samples with negation in Dev set: {len(neg_samples)}")
        for sid, text, negs in neg_samples[:10]:
            print(f"  - [{sid}] ({', '.join(negs)}): {text}")
        return {"total_gold_negation": len(neg_samples)}

    with open(transcripts_file, encoding="utf-8") as f:
        transcripts = json.load(f)
        
    failed = []
    category_counts = {"N0": 0, "N1": 0, "N2": 0, "N3": 0, "N4": 0}
    total_gold_with_negation = 0
    
    for sid in dev_ids:
        sample = all_samples.get(sid)
        if not sample:
            continue
            
        gold_text = sample.get("audio_text", "")
        pred_entry = transcripts.get(sid, {})
        pred_text = pred_entry.get("normalized_text", pred_entry.get("normalized_transcript", ""))
        
        gold_words = [w.strip(".,?!;:") for w in gold_text.lower().split()]
        pred_words = [w.strip(".,?!;:") for w in pred_text.lower().split()]
        
        gold_negs = set(w for w in gold_words if w in NEGATION_WORDS)
        pred_negs = set(w for w in pred_words if w in NEGATION_WORDS)
        
        if gold_negs:
            total_gold_with_negation += 1
            missing = gold_negs - pred_negs
            if missing:
                audio_path = PROJECT_ROOT / "train+public_test" / "train" / sample["audio"]
                duration = get_audio_duration(audio_path)
                history_len = len(sample.get("history", []))
                
                cat, reason = classify_negation_error(gold_text, pred_text, missing, duration, history_len)
                category_counts[cat] += 1
                
                tool_calls = [tc["name"] for tc in sample.get("label", {}).get("tool_calls", [])]
                
                failed.append({
                    "id": sid,
                    "category": cat,
                    "reason": reason,
                    "duration_sec": round(duration, 2),
                    "history_len": history_len,
                    "missing_negations": list(missing),
                    "gold": gold_text,
                    "pred": pred_text,
                    "tool_calls": tool_calls,
                })
                
    recall = (total_gold_with_negation - len(failed)) / max(total_gold_with_negation, 1)
    
    print("\n" + "=" * 70)
    print(f"NEGATION BENCHMARK SUMMARY (DEV SET)")
    print("=" * 70)
    print(f"Total samples with negation:   {total_gold_with_negation}")
    print(f"Correct samples:               {total_gold_with_negation - len(failed)}")
    print(f"Failed samples (missing neg):  {len(failed)}")
    print(f"Negation Intent Recall:        {recall:.1%}")
    print("\nError Breakdown by Root Cause:")
    print(f"  [N0] VAD Clipped (<1.5s audio / short turn): {category_counts['N0']} ({category_counts['N0']/max(len(failed),1):.1%})")
    print(f"  [N1] VAD Edge Clipped (start/end word):      {category_counts['N1']} ({category_counts['N1']/max(len(failed),1):.1%})")
    print(f"  [N2] Phonetic / Tokenizer Confusion:         {category_counts['N2']} ({category_counts['N2']/max(len(failed),1):.1%})")
    print(f"  [N3] Regional Dialect / Slang:               {category_counts['N3']} ({category_counts['N3']/max(len(failed),1):.1%})")
    print(f"  [N4] Fast speech / Audio noise:              {category_counts['N4']} ({category_counts['N4']/max(len(failed),1):.1%})")
    
    print("\nTop Failed Samples Details:")
    for f in failed[:15]:
        print(f"\n--- [{f['id']}] Category: {f['category']} ({f['reason']}) ---")
        print(f"  Duration:  {f['duration_sec']}s | History turns: {f['history_len']}")
        print(f"  Missing:   {f['missing_negations']}")
        print(f"  Gold:      {f['gold']}")
        print(f"  Pred:      {f['pred']}")
        print(f"  Gold tool: {f['tool_calls']}")
        
    out_file = PROJECT_ROOT / "outputs" / "negation_failure_analysis.json"
    with open(out_file, "w", encoding="utf-8") as fo:
        json.dump({
            "total_gold_with_negation": total_gold_with_negation,
            "failed_count": len(failed),
            "recall": recall,
            "category_counts": category_counts,
            "failed_samples": failed,
        }, fo, ensure_ascii=False, indent=2)
    print(f"\nDetailed report saved to: {out_file}")
    
    return {
        "total_gold": total_gold_with_negation,
        "failed": len(failed),
        "recall": recall,
        "category_counts": category_counts,
    }


if __name__ == "__main__":
    t_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    diagnose_negation(t_path)

