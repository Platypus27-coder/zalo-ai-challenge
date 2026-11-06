"""Part A — Local Evaluator: Reproduce the official scoring exactly.

Scoring rules (from official docs + plan):
- OOS (gold = []): score=1 if predicted=[], else score=0
- Non-OOS (|G| > 0):
  - If state_hash mismatch or M=0: score=0
  - Else: F1 = 2*P*R / (P+R) where P=M/(M+U), R=M/|G|
  
Matching a call:
1. Tool name must match exactly
2. Non-fuzzy params must match exactly  
3. Fuzzy params (x-match: "fuzzy") must have token-F1 > 0.6 after NFC + lowercase

Final score = (100/N) * sum(Score_i)
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import TRAIN_JSONL, TRAIN_CATALOG, FUZZY_THRESHOLD


def load_catalog_fuzzy_params(catalog_path: Path = TRAIN_CATALOG) -> dict[str, set[str]]:
    """Load which params are fuzzy (x-match: "fuzzy") for each tool.
    
    Returns: {tool_name: {param_name, ...}}
    """
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    
    fuzzy_params = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            tool_name = tool["name"]
            params = tool.get("parameters", {}).get("properties", {})
            fuzzy = set()
            for pname, pspec in params.items():
                if pspec.get("x-match") == "fuzzy":
                    fuzzy.add(pname)
            if fuzzy:
                fuzzy_params[tool_name] = fuzzy
    
    return fuzzy_params


def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching: NFC + lowercase."""
    if not isinstance(text, str):
        return str(text).lower()
    return unicodedata.normalize("NFC", text).lower()


def tokenize(text: str) -> list[str]:
    """Tokenize text into word tokens (lowercase, NFC normalized)."""
    normalized = normalize_text(text)
    return re.findall(r'[^\W_]+', normalized, re.UNICODE)


def token_f1(pred_text: str, gold_text: str) -> float:
    """Compute token-level F1 between prediction and gold text.
    
    After NFC normalization and lowercasing.
    """
    pred_tokens = tokenize(pred_text)
    gold_tokens = tokenize(gold_text)
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    
    # Count token occurrences
    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    
    # Common tokens (min count for each)
    common = sum((pred_counter & gold_counter).values())
    
    if common == 0:
        return 0.0
    
    precision = common / sum(pred_counter.values())
    recall = common / sum(gold_counter.values())
    
    return 2 * precision * recall / (precision + recall)


def calls_match(pred_call: dict, gold_call: dict, fuzzy_params: dict[str, set[str]]) -> bool:
    """Check if a predicted call matches a gold call.
    
    Matching rules:
    1. Tool name must match exactly
    2. Non-fuzzy params must match exactly
    3. Fuzzy params must have token-F1 > 0.6 (NFC + lowercase)
    """
    # Rule 1: name must match
    if pred_call.get("name") != gold_call.get("name"):
        return False
    
    tool_name = pred_call["name"]
    pred_args = pred_call.get("arguments", {})
    gold_args = gold_call.get("arguments", {})
    fuzzy = fuzzy_params.get(tool_name, set())
    
    # Get all keys from both
    all_keys = set(gold_args.keys()) | set(pred_args.keys())
    
    for key in all_keys:
        pred_val = pred_args.get(key)
        gold_val = gold_args.get(key)
        
        # Skip if both are None/absent
        if pred_val is None and gold_val is None:
            continue
        
        if key in fuzzy:
            # Rule 3: Fuzzy match
            if pred_val is None or gold_val is None:
                # One is missing — check if it matters
                if gold_val is not None:
                    # Gold has value, pred doesn't — token-F1 will be 0
                    return False
                # Gold doesn't have it but pred does — doesn't affect matching
                continue
            f1 = token_f1(str(pred_val), str(gold_val))
            if f1 <= FUZZY_THRESHOLD:
                return False
        else:
            # Rule 2: Exact match for non-fuzzy
            if pred_val != gold_val:
                return False
    
    return True


def compute_matched_unmatched(
    pred_calls: list[dict],
    gold_calls: list[dict],
    fuzzy_params: dict[str, set[str]]
) -> tuple[int, int]:
    """Compute M (matched) and U (unmatched) using 1-1 matching (order ignored).
    
    Uses greedy bipartite matching (gold calls matched first-come-first-served).
    Returns (M, U).
    """
    # Build matching matrix
    n_pred = len(pred_calls)
    n_gold = len(gold_calls)
    
    if n_pred == 0:
        return 0, 0
    if n_gold == 0:
        return 0, n_pred
    
    # Track which gold calls have been matched
    gold_matched = [False] * n_gold
    matched = 0
    
    # For each predicted call, try to find a matching gold call
    for pred in pred_calls:
        for j, gold in enumerate(gold_calls):
            if not gold_matched[j] and calls_match(pred, gold, fuzzy_params):
                gold_matched[j] = True
                matched += 1
                break
    
    unmatched = n_pred - matched
    return matched, unmatched


def score_sample(
    pred_calls: list[dict],
    pred_state_hash: str,
    gold_calls: list[dict],
    gold_state_hash: str,
    fuzzy_params: dict[str, set[str]]
) -> dict:
    """Score a single sample.
    
    Returns dict with: score, M, U, |G|, hash_match, P, R, details
    """
    n_gold = len(gold_calls)
    
    # Case 1: OOS (gold is empty)
    if n_gold == 0:
        if len(pred_calls) == 0:
            return {"score": 1.0, "M": 0, "U": 0, "G": 0, "hash_match": True, 
                    "P": 1.0, "R": 1.0, "type": "oos_correct"}
        else:
            return {"score": 0.0, "M": 0, "U": len(pred_calls), "G": 0, "hash_match": True,
                    "P": 0.0, "R": 0.0, "type": "oos_wrong"}
    
    # Case 2: Non-OOS
    hash_match = (pred_state_hash == gold_state_hash)
    M, U = compute_matched_unmatched(pred_calls, gold_calls, fuzzy_params)
    
    if not hash_match or M == 0:
        P = M / (M + U) if (M + U) > 0 else 0.0
        R = M / n_gold if n_gold > 0 else 0.0
        return {"score": 0.0, "M": M, "U": U, "G": n_gold, "hash_match": hash_match,
                "P": P, "R": R, "type": "hash_fail" if not hash_match else "no_match"}
    
    P = M / (M + U)
    R = M / n_gold
    f1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    
    return {"score": f1, "M": M, "U": U, "G": n_gold, "hash_match": hash_match,
            "P": P, "R": R, "type": "scored"}


def evaluate_submission(
    submission: list[dict],
    gold_data: list[dict],
    fuzzy_params: dict[str, set[str]],
    gold_hashes: dict[str, str] | None = None
) -> dict:
    """Evaluate a full submission against gold data.
    
    Args:
        submission: list of {id, tool_calls, state_hash}
        gold_data: list of training samples with labels
        fuzzy_params: {tool_name: {fuzzy_param_names}}
        gold_hashes: {id: state_hash} precomputed gold state hashes (optional, for train only)
    
    Returns summary with per-sample scores and aggregate metrics.
    """
    # Index gold data by ID
    gold_by_id = {s["id"]: s for s in gold_data}
    
    results = []
    total_score = 0.0
    
    for sub in submission:
        sample_id = sub["id"]
        pred_calls = sub.get("tool_calls", [])
        pred_hash = sub.get("state_hash", "")
        
        gold = gold_by_id.get(sample_id)
        if gold is None:
            results.append({"id": sample_id, "score": 0.0, "error": "missing_gold"})
            continue
        
        gold_calls = gold.get("label", {}).get("tool_calls", [])
        
        # Get gold state hash — need to replay through ToolEnv for exact hash
        gold_hash = gold_hashes.get(sample_id, "") if gold_hashes else ""
        
        result = score_sample(pred_calls, pred_hash, gold_calls, gold_hash, fuzzy_params)
        result["id"] = sample_id
        results.append(result)
        total_score += result["score"]
    
    n = len(results)
    final_score = (100 / n * total_score) if n > 0 else 0.0
    
    # Aggregate stats
    type_counts = Counter(r.get("type", "unknown") for r in results)
    hash_match_count = sum(1 for r in results if r.get("hash_match", False))
    
    return {
        "final_score": final_score,
        "n_samples": n,
        "avg_score": total_score / n if n > 0 else 0.0,
        "hash_match_rate": hash_match_count / n if n > 0 else 0.0,
        "type_counts": dict(type_counts),
        "results": results
    }


def compute_gold_hashes_via_replay(
    samples: list[dict],
    split: str = "train",
    data_root: str | Path | None = None
) -> dict[str, str]:
    """Replay gold tool_calls through ToolEnv to get gold state hashes.
    
    This is the most reliable way to get gold hashes for evaluation.
    """
    from pathlib import Path as _Path
    
    if data_root is None:
        from config import DATA_ROOT
        data_root = DATA_ROOT
    
    data_root = _Path(data_root)
    
    # Add zaic_tools to path
    zaic_path = str(data_root / split / "zaic_tools")
    if zaic_path not in sys.path:
        sys.path.insert(0, str(data_root / split))
    
    from zaic_tools import ToolEnv
    
    gold_hashes = {}
    errors = []
    
    for sample in samples:
        sample_id = sample["id"]
        try:
            env = ToolEnv(split, sample_id, root=str(data_root))
            
            # Replay gold tool calls
            gold_calls = sample.get("label", {}).get("tool_calls", [])
            for tc in gold_calls:
                env.call(tc["name"], tc.get("arguments", {}))
            
            gold_hashes[sample_id] = env.get_state_hash()
        except Exception as e:
            errors.append({"id": sample_id, "error": str(e)})
    
    if errors:
        print(f"WARNING: {len(errors)} errors during gold hash replay:")
        for err in errors[:5]:
            print(f"  {err['id']}: {err['error']}")
    
    return gold_hashes


if __name__ == "__main__":
    print("Loading catalog for fuzzy params...")
    fuzzy_params = load_catalog_fuzzy_params()
    print(f"Found fuzzy params in {len(fuzzy_params)} tools:")
    for tool, params in sorted(fuzzy_params.items()):
        print(f"  {tool}: {params}")
    
    # Test token-F1
    print("\n--- Token-F1 Tests ---")
    tests = [
        ("mai chạy bộ dời sang 09:00 nhé", "mai chạy bộ dời sang 09:00 nhé", "exact"),
        ("nap 50 nghin dien thoai", "nạp 50 nghìn điện thoại", "no-diacritics"),
        ("hello world", "completely different", "mismatch"),
    ]
    for pred, gold, label in tests:
        f1 = token_f1(pred, gold)
        print(f"  {label}: F1={f1:.3f} (pred='{pred}', gold='{gold}')")


class Evaluator:
    """Convenience class to evaluate submissions against ground truth labels."""

    def __init__(self, split: str = "train", catalog_path: Path = TRAIN_CATALOG):
        self.split = split
        self.catalog_path = catalog_path
        self.fuzzy_params = load_catalog_fuzzy_params(catalog_path)
        from src.part_a.runner import load_samples
        self.gold_data = load_samples(split)
        self.gold_hashes = {}

    def evaluate(self, submission: list[dict]) -> dict:
        sample_ids = {s["id"] for s in submission}
        needed_samples = [s for s in self.gold_data if s["id"] in sample_ids and s["id"] not in self.gold_hashes]
        if needed_samples:
            computed = compute_gold_hashes_via_replay(needed_samples, split="train")
            self.gold_hashes.update(computed)
        res = evaluate_submission(submission, self.gold_data, self.fuzzy_params, self.gold_hashes)
        return {
            "score": res.get("final_score", 0.0),
            "hash_match_rate": res.get("hash_match_rate", 0.0),
            "details": res
        }
