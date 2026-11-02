"""Part A — State Diff Inspector: Precise diffing of ToolEnv states on Hash Mismatch.

When a predicted trajectory produces a state_hash mismatch with gold,
this tool pinpoints the exact app, table, entity ID, and attribute that diverged.

Usage:
    from src.part_a.state_diff import diff_states_for_sample
    diff = diff_states_for_sample(sample_id, pred_calls, gold_calls)
"""
import copy
import json
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import DATA_ROOT


def get_zaic_env(split: str, sample_id: str):
    """Instantiate a fresh ToolEnv session for a sample."""
    split_dir = DATA_ROOT / split
    if str(split_dir) not in sys.path:
        sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv
    return ToolEnv(split, sample_id, root=str(DATA_ROOT))


def compute_state_diff(before_state: dict, after_state: dict) -> dict[str, Any]:
    """Find changed apps and keys between two states."""
    diff = {}
    all_apps = set(before_state.keys()) | set(after_state.keys())

    for app in all_apps:
        b_app = before_state.get(app, {})
        a_app = after_state.get(app, {})
        if b_app != a_app:
            app_diff = _diff_dict(b_app, a_app)
            if app_diff:
                diff[app] = app_diff
    return diff


def _diff_dict(d1: Any, d2: Any) -> Any:
    """Recursive diff between two JSON-compatible structures."""
    if d1 == d2:
        return None

    if isinstance(d1, dict) and isinstance(d2, dict):
        keys = set(d1.keys()) | set(d2.keys())
        sub = {}
        for k in keys:
            if k not in d1:
                sub[k] = {"status": "ADDED_IN_PRED", "value": d2[k]}
            elif k not in d2:
                sub[k] = {"status": "MISSING_IN_PRED", "expected": d1[k]}
            else:
                inner = _diff_dict(d1[k], d2[k])
                if inner is not None:
                    sub[k] = inner
        return sub if sub else None

    if isinstance(d1, list) and isinstance(d2, list):
        if d1 != d2:
            return {
                "status": "LIST_MISMATCH",
                "gold_len": len(d1),
                "pred_len": len(d2),
                "gold_sample": d1[:3],
                "pred_sample": d2[:3]
            }
        return None

    return {
        "status": "VALUE_MISMATCH",
        "expected": d1,
        "got": d2
    }


def diff_states_for_sample(
    sample_id: str,
    pred_tool_calls: list[dict],
    gold_tool_calls: list[dict],
    split: str = "train"
) -> dict[str, Any]:
    """Replay both prediction and gold through fresh ToolEnvs and return exact differences.

    Returns:
        dict containing:
            - hash_match: bool
            - gold_hash: str
            - pred_hash: str
            - state_differences: dict mapping app -> field differences
            - gold_calls_executed: list of executed gold calls
            - pred_calls_executed: list of executed pred calls
    """
    # 1. Gold replay
    gold_env = get_zaic_env(split, sample_id)
    for c in gold_tool_calls:
        gold_env.call(c.get("name", ""), c.get("arguments", {}))
    gold_hash = gold_env.get_state_hash()
    gold_state = gold_env.get_state()

    # 2. Pred replay
    pred_env = get_zaic_env(split, sample_id)
    pred_results = []
    for c in pred_tool_calls:
        res = pred_env.call(c.get("name", ""), c.get("arguments", {}))
        pred_results.append({"name": c.get("name"), "res": res})
    pred_hash = pred_env.get_state_hash()
    pred_state = pred_env.get_state()

    hash_match = (gold_hash == pred_hash)
    differences = {} if hash_match else compute_state_diff(gold_state, pred_state)

    return {
        "sample_id": sample_id,
        "hash_match": hash_match,
        "gold_hash": gold_hash,
        "pred_hash": pred_hash,
        "state_differences": differences,
        "pred_results": pred_results,
    }
