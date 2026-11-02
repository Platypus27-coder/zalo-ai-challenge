"""Part A — Loss Attribution Matrix & Work Prioritizer (§4.2 of Review Plan v3).

Computes the definitive Loss Attribution Table:
- Groups every failing sample by its root cause:
    1. ASR Slot Error (name, amount, phone, date, negation)
    2. Entity Grounding / History Reference Error (wrong ID, stale ID, wrong entity)
    3. Task Dependency / Missing Write (missed action or duplicate write)
    4. Intent Classification Error (False OOS, False Ask, Premature Execute)
    5. Schema / Execution / Timeout Error
- Calculates total lost score: Sigma(1 - s_i)
- Computes Expected Recoverable Score and ROI for team experiment ranking.

Usage:
    python -m src.part_a.loss_attribution --submission outputs/submissions/pred.jsonl --split dev
"""
import json
import argparse
from pathlib import Path
from typing import Any
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import DATA_ROOT, TRAIN_JSONL, TRAIN_CATALOG, SPLITS_DIR, OUTPUTS_DIR
from src.part_a.evaluator import score_sample, load_catalog_fuzzy_params, compute_gold_hashes_via_replay
from src.part_a.state_diff import diff_states_for_sample


class LossAttributor:
    """Attributes lost points Sigma(1 - s) to mutually exclusive root causes."""

    # Pre-defined estimated recoverable percentage & estimated dev cost (person-hours)
    HEURISTICS = {
        "ASR Slot & Phonetic Errors": {
            "fix_direction": "Part B: Prompt biasing, number/date normalizer, fallback decoding",
            "recoverable_rate": 0.65,
            "cost_hours": 12,
        },
        "Entity Grounding & History Reference": {
            "fix_direction": "Part C: Evidence Ledger, canonical ID resolution, state invalidation",
            "recoverable_rate": 0.70,
            "cost_hours": 14,
        },
        "Task Dependency & Multi-Step Ordering": {
            "fix_direction": "Part C: Task State Tracker (pending/blocked/done), sequential unblocking",
            "recoverable_rate": 0.60,
            "cost_hours": 10,
        },
        "Intent / Policy Confusion (OOS vs Ask vs Exec)": {
            "fix_direction": "Part C: OOS installed_apps gatekeeper, Ask threshold calibration",
            "recoverable_rate": 0.80,
            "cost_hours": 8,
        },
        "Schema & Parameter Formatting": {
            "fix_direction": "Part A/C: Pre-call schema validator, ISO-8601 formatting, enum cast",
            "recoverable_rate": 0.90,
            "cost_hours": 6,
        },
        "Uncategorized State Divergence": {
            "fix_direction": "Deep dive with state_diff tool",
            "recoverable_rate": 0.50,
            "cost_hours": 10,
        }
    }

    def __init__(self, catalog_path: Path = TRAIN_CATALOG):
        self.fuzzy_params = load_catalog_fuzzy_params(catalog_path)

    def attribute(
        self,
        predictions: list[dict],
        gold_samples: list[dict],
        gold_hashes: dict[str, str],
        inspect_state: bool = False
    ) -> dict[str, Any]:
        """Compute attribution matrix across all samples."""
        gold_by_id = {s["id"]: s for s in gold_samples}

        # Root cause -> list of sample records
        groups = defaultdict(list)
        total_lost_score = 0.0
        n_samples = len(predictions)

        for pred in predictions:
            sample_id = pred["id"]
            gold = gold_by_id.get(sample_id)
            if not gold:
                continue

            pred_calls = pred.get("tool_calls", [])
            pred_hash = pred.get("state_hash", "")
            gold_calls = gold.get("label", {}).get("tool_calls", [])
            gold_hash = gold_hashes.get(sample_id, "")

            res = score_sample(pred_calls, pred_hash, gold_calls, gold_hash, self.fuzzy_params)
            score = res["score"]
            lost = max(0.0, 1.0 - score)
            total_lost_score += lost

            if lost < 0.001:
                continue  # Perfect sample, no loss

            # Determine primary root cause
            root_cause = self._identify_root_cause(
                sample_id=sample_id,
                pred_calls=pred_calls,
                pred_hash=pred_hash,
                gold_calls=gold_calls,
                gold_hash=gold_hash,
                gold_sample=gold,
                score_res=res
            )

            record = {
                "id": sample_id,
                "score": score,
                "lost": lost,
                "gold_calls": [c["name"] for c in gold_calls],
                "pred_calls": [c["name"] for c in pred_calls],
                "transcript": gold.get("audio_text", "")
            }
            groups[root_cause].append(record)

        # Build table
        table = []
        scale = 100.0 / max(1, n_samples)
        for category, samples in groups.items():
            count = len(samples)
            cat_lost_raw = sum(s["lost"] for s in samples)
            cat_lost_points = cat_lost_raw * scale

            heur = self.HEURISTICS.get(category, {
                "fix_direction": "General investigation",
                "recoverable_rate": 0.50,
                "cost_hours": 10
            })
            recoverable_pts = cat_lost_points * heur["recoverable_rate"]
            roi = recoverable_pts / max(1, heur["cost_hours"])

            table.append({
                "category": category,
                "sample_count": count,
                "total_lost_points": round(cat_lost_points, 2),
                "recoverable_rate": heur["recoverable_rate"],
                "expected_gain_points": round(recoverable_pts, 2),
                "estimated_hours": heur["cost_hours"],
                "roi": round(roi, 3),
                "fix_direction": heur["fix_direction"],
                "sample_ids": [s["id"] for s in samples[:5]]
            })

        # Sort table by total lost points descending
        table.sort(key=lambda x: -x["total_lost_points"])

        return {
            "total_samples": n_samples,
            "total_lost_points_scale100": round(total_lost_score * scale, 2),
            "achieved_score_scale100": round(100.0 - (total_lost_score * scale), 2),
            "attribution_table": table,
        }

    def _identify_root_cause(
        self,
        sample_id: str,
        pred_calls: list[dict],
        pred_hash: str,
        gold_calls: list[dict],
        gold_hash: str,
        gold_sample: dict,
        score_res: dict
    ) -> str:
        """Assign single mutually exclusive root cause."""
        n_gold = len(gold_calls)
        n_pred = len(pred_calls)

        # 1. Intent Mismatch (OOS / Ask)
        gold_has_ask = any(c.get("name") == "sys.ask_missing_info" for c in gold_calls)
        pred_has_ask = any(c.get("name") == "sys.ask_missing_info" for c in pred_calls)

        if n_gold == 0 and n_pred > 0:
            return "Intent / Policy Confusion (OOS vs Ask vs Exec)"
        if n_gold > 0 and n_pred == 0:
            return "Intent / Policy Confusion (OOS vs Ask vs Exec)"
        if gold_has_ask != pred_has_ask:
            return "Intent / Policy Confusion (OOS vs Ask vs Exec)"

        # 2. Schema / Format Errors
        # If any tool was rejected by schema
        pred_names = {c.get("name") for c in pred_calls}
        gold_names = {c.get("name") for c in gold_calls}

        # 3. Missing or Extra Tool calls
        if pred_names != gold_names:
            # Check if tools are related or completely missed
            return "Task Dependency & Multi-Step Ordering"

        # 4. Same tools called, but parameters differ
        # Check if difference is in ID parameters vs content parameters
        id_keys = {"device_id", "contact_id", "event_id", "reminder_id", "alarm_id", "appliance_id", "lock_id"}
        has_id_mismatch = False
        has_slot_mismatch = False

        for p_call in pred_calls:
            p_name = p_call.get("name")
            p_args = p_call.get("arguments", {})
            for g_call in gold_calls:
                if g_call.get("name") == p_name:
                    g_args = g_call.get("arguments", {})
                    for k in set(p_args) | set(g_args):
                        if p_args.get(k) != g_args.get(k):
                            if k in id_keys or k.endswith("_id"):
                                has_id_mismatch = True
                            else:
                                has_slot_mismatch = True

        if has_id_mismatch:
            return "Entity Grounding & History Reference"
        if has_slot_mismatch:
            return "ASR Slot & Phonetic Errors"

        # 5. State hash mismatch despite matched tool calls
        if pred_hash != gold_hash:
            return "Uncategorized State Divergence"

        return "Schema & Parameter Formatting"

    def print_matrix(self, result: dict[str, Any]):
        """Print formatted Markdown table directly conforming to §4.2."""
        print("=" * 100)
        print("LIFEMATE LOSS ATTRIBUTION MATRIX (§4.2)")
        print("=" * 100)
        print(f"Total Evaluated: {result['total_samples']} samples")
        print(f"Current Score:   {result['achieved_score_scale100']:.2f} / 100")
        print(f"Total Lost:      {result['total_lost_points_scale100']:.2f} points")
        print("-" * 100)
        print(f"{'Nhóm nguyên nhân gốc':<42} | {'Mẫu':<5} | {'Điểm mất':<9} | {'Cứu được':<9} | {'Giờ':<4} | {'ROI':<6}")
        print("-" * 100)
        for row in result["attribution_table"]:
            print(f"{row['category']:<42} | {row['sample_count']:<5} | {row['total_lost_points']:<9} | {row['expected_gain_points']:<9} | {row['estimated_hours']:<4} | {row['roi']:<6}")
        print("-" * 100)
        print("\n🎯 TOP 2 ĐỀ XUẤT CAN THIỆP (HIGHEST ROI):")
        sorted_by_roi = sorted(result["attribution_table"], key=lambda x: -x["roi"])
        for i, top in enumerate(sorted_by_roi[:2], start=1):
            print(f"   [{i}] {top['category']} (ROI: {top['roi']})")
            print(f"       -> Hướng sửa: {top['fix_direction']}")
            print(f"       -> Điểm dự kiến thu hồi: +{top['expected_gain_points']} điểm")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Loss Attribution Matrix")
    parser.add_argument("--submission", required=True, help="Path to submission.jsonl")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "holdout"])
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    splits_file = SPLITS_DIR / "splits.json"
    with open(splits_file, encoding="utf-8") as f:
        splits_data = json.load(f)
    target_ids = set(splits_data.get("splits", {}).get(args.split, splits_data.get(args.split, [])))

    all_samples = [json.loads(l) for l in open(TRAIN_JSONL, encoding="utf-8") if l.strip()]
    gold_samples = [s for s in all_samples if s["id"] in target_ids]

    preds = []
    with open(args.submission, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["id"] in target_ids:
                    preds.append(item)

    print(f"Computing gold state hashes for {len(gold_samples)} samples...")
    gold_hashes = compute_gold_hashes_via_replay(gold_samples, split="train", data_root=DATA_ROOT)

    attributor = LossAttributor()
    res = attributor.attribute(preds, gold_samples, gold_hashes)
    attributor.print_matrix(res)

    out_file = Path(args.output) if args.output else OUTPUTS_DIR / f"loss_attribution_{args.split}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {out_file}")


if __name__ == "__main__":
    main()
