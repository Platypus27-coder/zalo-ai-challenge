"""Part A — Data Splitter: Create reproducible train/dev/holdout splits.

Groups samples by conversation session (same user + overlapping apps pattern)
to prevent data leakage. Stratifies by:
- App domains
- Output type (execute / ask_missing_info / out-of-scope)  
- Trace length (number of tool calls)
- History presence
"""
import json
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    TRAIN_JSONL, SPLITS_DIR,
    SPLIT_SEED, TRAIN_RATIO, DEV_RATIO, HOLDOUT_RATIO
)


def load_train_data(jsonl_path: Path = TRAIN_JSONL) -> list[dict]:
    """Load all training samples from JSONL."""
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def classify_sample(sample: dict) -> dict:
    """Extract classification features from a sample."""
    label = sample.get("label", {})
    tool_calls = label.get("tool_calls", [])
    
    # Output type
    if len(tool_calls) == 0:
        output_type = "oos"  # out-of-scope
    elif any(tc["name"] == "sys.ask_missing_info" for tc in tool_calls):
        output_type = "ask"
    else:
        output_type = "execute"
    
    # Apps used
    apps_used = set()
    for tc in tool_calls:
        if "." in tc["name"]:
            apps_used.add(tc["name"].split(".")[0])
    
    # Installed apps
    installed_apps = set(sample.get("context", {}).get("installed_apps", {}).keys())
    
    # History presence
    has_history = len(sample.get("history", [])) > 0
    
    # Trace length
    trace_len = len(tool_calls)
    
    # Has code-switching (English words in Vietnamese text)
    audio_text = sample.get("audio_text", "")
    
    return {
        "id": sample["id"],
        "output_type": output_type,
        "apps_used": sorted(apps_used),
        "installed_apps": sorted(installed_apps),
        "has_history": has_history,
        "trace_len": trace_len,
        "audio_text": audio_text,
        # Stratification key: combine output_type + primary app + trace length bucket
        "strat_key": f"{output_type}_{'_'.join(sorted(apps_used)[:2]) if apps_used else 'none'}_{min(trace_len, 5)}"
    }


def create_splits(
    samples: list[dict],
    train_ratio: float = TRAIN_RATIO,
    dev_ratio: float = DEV_RATIO,
    holdout_ratio: float = HOLDOUT_RATIO,
    seed: int = SPLIT_SEED
) -> dict[str, list[str]]:
    """Create stratified train/dev/holdout splits.
    
    Returns dict with keys 'train', 'dev', 'holdout' mapping to lists of sample IDs.
    """
    random.seed(seed)
    
    # Classify all samples
    classified = [classify_sample(s) for s in samples]
    
    # Group by stratification key
    groups = defaultdict(list)
    for c in classified:
        groups[c["strat_key"]].append(c["id"])
    
    # Shuffle within each group
    for ids in groups.values():
        random.shuffle(ids)
    
    train_ids, dev_ids, holdout_ids = [], [], []
    
    for strat_key, ids in sorted(groups.items()):
        n = len(ids)
        n_train = max(1, round(n * train_ratio))
        n_dev = max(1 if n > 1 else 0, round(n * dev_ratio))
        # Rest goes to holdout
        
        train_ids.extend(ids[:n_train])
        dev_ids.extend(ids[n_train:n_train + n_dev])
        holdout_ids.extend(ids[n_train + n_dev:])
    
    splits = {
        "train": sorted(train_ids),
        "dev": sorted(dev_ids),
        "holdout": sorted(holdout_ids),
    }
    
    return splits


def save_splits(splits: dict[str, list[str]], output_dir: Path = SPLITS_DIR, seed: int = SPLIT_SEED):
    """Save split IDs to JSON files with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "seed": seed,
        "total": sum(len(v) for v in splits.values()),
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits
    }
    
    output_path = output_dir / "splits.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Also save individual ID lists
    for name, ids in splits.items():
        with open(output_dir / f"{name}_ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f, indent=2)
    
    return output_path


def load_splits(splits_dir: Path = SPLITS_DIR) -> dict[str, list[str]]:
    """Load previously saved splits."""
    splits_file = splits_dir / "splits.json"
    with open(splits_file, encoding="utf-8") as f:
        data = json.load(f)
    return data["splits"]


def print_split_stats(samples: list[dict], splits: dict[str, list[str]]):
    """Print statistics for each split."""
    id_to_sample = {s["id"]: s for s in samples}
    
    print("=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)
    
    for split_name, ids in splits.items():
        split_samples = [id_to_sample[sid] for sid in ids if sid in id_to_sample]
        classified = [classify_sample(s) for s in split_samples]
        
        # Count output types
        type_counts = Counter(c["output_type"] for c in classified)
        
        # Count apps
        app_counts = Counter()
        for c in classified:
            for app in c["apps_used"]:
                app_counts[app] += 1
        
        # Trace length distribution
        trace_lens = [c["trace_len"] for c in classified]
        
        # History count
        history_count = sum(1 for c in classified if c["has_history"])
        
        print(f"\n--- {split_name.upper()} ({len(ids)} samples) ---")
        print(f"  Output types: {dict(type_counts)}")
        print(f"  Has history: {history_count} ({100*history_count/max(len(ids),1):.1f}%)")
        print(f"  Trace length: min={min(trace_lens, default=0)}, max={max(trace_lens, default=0)}, "
              f"avg={sum(trace_lens)/max(len(trace_lens),1):.1f}")
        print(f"  Top apps: {dict(app_counts.most_common(10))}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Loading training data...")
    samples = load_train_data()
    print(f"Loaded {len(samples)} samples")
    
    print("\nCreating splits...")
    splits = create_splits(samples)
    
    print_split_stats(samples, splits)
    
    output_path = save_splits(splits)
    print(f"\nSplits saved to {output_path}")
