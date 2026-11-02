"""Part A — Diagnostic Error Analyzer.

Deeply inspects prediction failures against ground truth:
1. Root cause breakdown: State Hash mismatch, OOS confusion, Missing Call, Extra Call, Parameter Error.
2. Parameter error diagnosis: Exact match failure vs Fuzzy Token-F1 failure (< 0.6).
3. State divergence inspection: Compares ToolEnv state when hashes differ.
4. Actionable recommendations for Part B (ASR) and Part C (Agent LLM).

Usage:
    python -m src.part_a.diagnostic --submission outputs/submissions/pred.jsonl --split dev
"""
import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import DATA_ROOT, TRAIN_JSONL, TRAIN_CATALOG, OUTPUTS_DIR, SPLITS_DIR, FUZZY_THRESHOLD
from src.part_a.evaluator import (
    load_catalog_fuzzy_params, tokenize, token_f1, normalize_text,
    calls_match, score_sample, compute_gold_hashes_via_replay
)


class DiagnosticAnalyzer:
    """Diagnoses prediction failures with deep granularity."""

    def __init__(self, catalog_path: Path = TRAIN_CATALOG):
        self.catalog_path = catalog_path
        self.fuzzy_params = load_catalog_fuzzy_params(catalog_path)

    def analyze(
        self,
        predictions: list[dict],
        gold_samples: list[dict],
        gold_hashes: dict[str, str],
        split_name: str = "dev"
    ) -> dict[str, Any]:
        """Perform granular error analysis."""
        gold_by_id = {s["id"]: s for s in gold_samples}

        summary = {
            "split": split_name,
            "total_evaluated": len(predictions),
            "perfect_score_count": 0,
            "zero_score_count": 0,
            "partial_score_count": 0,
            "total_score": 0.0,
            "avg_score": 0.0,
            "error_categories": Counter(),
            "decision_matrix": defaultdict(int),
            "tool_call_stats": {
                "total_gold_calls": 0,
                "total_pred_calls": 0,
                "matched_calls": 0,
                "unmatched_calls": 0,
                "missed_calls": 0,
            },
            "tool_failing_rank": Counter(),
            "param_exact_mismatches": Counter(),
            "param_fuzzy_failures": [],
            "detailed_failures": [],
        }

        for pred in predictions:
            sample_id = pred["id"]
            gold = gold_by_id.get(sample_id)
            if not gold:
                continue

            pred_calls = pred.get("tool_calls", [])
            pred_hash = pred.get("state_hash", "")
            gold_calls = gold.get("label", {}).get("tool_calls", [])
            gold_hash = gold_hashes.get(sample_id, "")

            # Determine decision intent
            gold_intent = self._classify_intent(gold_calls)
            pred_intent = self._classify_intent(pred_calls)
            summary["decision_matrix"][f"gold_{gold_intent} -> pred_{pred_intent}"] += 1

            n_gold = len(gold_calls)
            n_pred = len(pred_calls)
            summary["tool_call_stats"]["total_gold_calls"] += n_gold
            summary["tool_call_stats"]["total_pred_calls"] += n_pred

            res = score_sample(pred_calls, pred_hash, gold_calls, gold_hash, self.fuzzy_params)
            score = res["score"]
            summary["total_score"] += score

            summary["tool_call_stats"]["matched_calls"] += res.get("M", 0)
            summary["tool_call_stats"]["unmatched_calls"] += res.get("U", 0)
            summary["tool_call_stats"]["missed_calls"] += max(0, n_gold - res.get("M", 0))

            if score >= 0.999:
                summary["perfect_score_count"] += 1
                continue
            elif score <= 0.001:
                summary["zero_score_count"] += 1
            else:
                summary["partial_score_count"] += 1

            # Detailed failure diagnosis
            diag = self._diagnose_sample(
                sample_id=sample_id,
                pred_calls=pred_calls,
                pred_hash=pred_hash,
                gold_calls=gold_calls,
                gold_hash=gold_hash,
                score_res=res,
                transcript=gold.get("audio_text", ""),
                summary_stats=summary
            )
            summary["detailed_failures"].append(diag)

        n = len(predictions)
        summary["avg_score"] = round(summary["total_score"] / max(1, n), 4)
        summary["final_score_scaled"] = round(summary["avg_score"] * 100, 2)
        return summary

    def _classify_intent(self, calls: list[dict]) -> str:
        if not calls:
            return "oos"
        if any(c.get("name") == "sys.ask_missing_info" for c in calls):
            return "ask"
        return "execute"

    def _diagnose_sample(
        self,
        sample_id: str,
        pred_calls: list[dict],
        pred_hash: str,
        gold_calls: list[dict],
        gold_hash: str,
        score_res: dict,
        transcript: str,
        summary_stats: dict
    ) -> dict[str, Any]:
        """Diagnose a single failing sample."""
        causes = []
        n_gold = len(gold_calls)
        n_pred = len(pred_calls)

        # 1. Check Hash Mismatch
        hash_mismatch = (pred_hash != gold_hash)
        if hash_mismatch and n_gold > 0:
            causes.append("HASH_MISMATCH")
            summary_stats["error_categories"]["Hash Mismatch (State Diverged)"] += 1

        # 2. Check Decision Disagreement
        gold_intent = self._classify_intent(gold_calls)
        pred_intent = self._classify_intent(pred_calls)
        if gold_intent != pred_intent:
            if gold_intent == "oos" and pred_intent != "oos":
                causes.append("FALSE_POSITIVE_OOS")
                summary_stats["error_categories"]["False Positive (Executed OOS sample)"] += 1
            elif gold_intent != "oos" and pred_intent == "oos":
                causes.append("FALSE_NEGATIVE_OOS")
                summary_stats["error_categories"]["False Negative (OOS instead of Action)"] += 1
            elif gold_intent == "ask" and pred_intent == "execute":
                causes.append("CONFIDENT_OVEREXECUTE")
                summary_stats["error_categories"]["Executed without asking required info"] += 1
            elif gold_intent == "execute" and pred_intent == "ask":
                causes.append("PREMATURE_ASK")
                summary_stats["error_categories"]["Asked missing info unnecessarily"] += 1

        # 3. Match Tool Calls & Inspect Params
        matched_gold_indices = set()
        param_errors = []

        for p_idx, pred_call in enumerate(pred_calls):
            p_name = pred_call.get("name")
            matched_gold_idx = None
            best_gold_candidate = None
            best_candidate_idx = None

            for g_idx, gold_call in enumerate(gold_calls):
                if g_idx in matched_gold_indices:
                    continue
                if gold_call.get("name") == p_name:
                    best_gold_candidate = gold_call
                    best_candidate_idx = g_idx
                    if calls_match(pred_call, gold_call, self.fuzzy_params):
                        matched_gold_idx = g_idx
                        break

            if matched_gold_idx is not None:
                matched_gold_indices.add(matched_gold_idx)
            else:
                summary_stats["tool_failing_rank"][p_name] += 1

                # If same tool exists in gold, find parameter discrepancy!
                if best_gold_candidate is not None:
                    param_errs = self._diff_call_params(pred_call, best_gold_candidate)
                    param_errors.extend(param_errs)
                    for pe in param_errs:
                        if pe["type"] == "exact_mismatch":
                            summary_stats["param_exact_mismatches"][f"{p_name}.{pe['param']}"] += 1
                        elif pe["type"] == "fuzzy_low_score":
                            summary_stats["param_fuzzy_failures"].append({
                                "sample_id": sample_id,
                                "tool": p_name,
                                "param": pe["param"],
                                "pred": pe["pred"],
                                "gold": pe["gold"],
                                "f1": pe["f1"],
                            })

        missed = [gold_calls[i]["name"] for i in range(n_gold) if i not in matched_gold_indices]
        for m in missed:
            summary_stats["tool_failing_rank"][f"MISSED_{m}"] += 1

        if not causes:
            if missed:
                causes.append("MISSED_TOOLS")
                summary_stats["error_categories"]["Missed Expected Tool Call"] += 1
            if param_errors:
                causes.append("PARAMETER_MISMATCH")
                summary_stats["error_categories"]["Parameter Value Discrepancy"] += 1

        return {
            "id": sample_id,
            "score": score_res["score"],
            "causes": causes,
            "gold_intent": gold_intent,
            "pred_intent": pred_intent,
            "gold_calls": [c.get("name") for c in gold_calls],
            "pred_calls": [c.get("name") for c in pred_calls],
            "hash_mismatch": hash_mismatch,
            "param_errors": param_errors,
            "audio_transcript": transcript,
        }

    def _diff_call_params(self, pred_call: dict, gold_call: dict) -> list[dict]:
        """Find why two calls with the same tool name failed to match."""
        errs = []
        t_name = pred_call.get("name", "")
        p_args = pred_call.get("arguments", {})
        g_args = gold_call.get("arguments", {})
        fuzzy = self.fuzzy_params.get(t_name, set())

        all_keys = set(p_args.keys()) | set(g_args.keys())
        for k in all_keys:
            pv = p_args.get(k)
            gv = g_args.get(k)
            if pv == gv:
                continue

            if k in fuzzy:
                f1 = token_f1(str(pv or ""), str(gv or ""))
                if f1 <= FUZZY_THRESHOLD:
                    errs.append({
                        "type": "fuzzy_low_score",
                        "param": k,
                        "pred": str(pv),
                        "gold": str(gv),
                        "f1": round(f1, 3),
                    })
            else:
                errs.append({
                    "type": "exact_mismatch",
                    "param": k,
                    "pred": pv,
                    "gold": gv,
                })
        return errs

    def print_diagnostic_report(self, summary: dict[str, Any]):
        """Print rich diagnostics to terminal."""
        print("=" * 80)
        print("LIFEMATE DIAGNOSTIC ERROR ANALYSIS")
        print("=" * 80)
        print(f"Evaluated Samples:  {summary['total_evaluated']}")
        print(f"Final Score:        {summary['final_score_scaled']:.2f}/100 (Avg F1: {summary['avg_score']:.4f})")
        print(f"Perfect (Score=1):  {summary['perfect_score_count']} ({summary['perfect_score_count']/max(1, summary['total_evaluated'])*100:.1f}%)")
        print(f"Zero (Score=0):     {summary['zero_score_count']} ({summary['zero_score_count']/max(1, summary['total_evaluated'])*100:.1f}%)")
        print(f"Partial Score:      {summary['partial_score_count']}")
        print("-" * 80)

        print("📊 ROOT CAUSE BREAKDOWN:")
        for cat, cnt in summary["error_categories"].most_common():
            print(f"   {cat:45s}: {cnt:4d}")
        print("-" * 80)

        print("🔀 INTENT CONFUSION MATRIX:")
        for trans, count in sorted(summary["decision_matrix"].items(), key=lambda x: -x[1]):
            print(f"   {trans:40s}: {count:4d}")
        print("-" * 80)

        t_stats = summary["tool_call_stats"]
        print("🎯 TOOL-LEVEL MATCHING STATS:")
        print(f"   Gold Calls:      {t_stats['total_gold_calls']}")
        print(f"   Predicted Calls: {t_stats['total_pred_calls']}")
        print(f"   Matched Calls:   {t_stats['matched_calls']}")
        print(f"   Unmatched Calls: {t_stats['unmatched_calls']}")
        print(f"   Missed Calls:    {t_stats['missed_calls']}")
        print("-" * 80)

        if summary["tool_failing_rank"]:
            print("🚨 TOP FAILING TOOLS:")
            for tool, cnt in summary["tool_failing_rank"].most_common(10):
                print(f"   {tool:40s}: {cnt:4d} errors")
            print("-" * 80)

        if summary["param_exact_mismatches"]:
            print("⚠️  TOP PARAMETER EXACT MISMATCHES:")
            for p, cnt in summary["param_exact_mismatches"].most_common(10):
                print(f"   {p:40s}: {cnt:4d}")
            print("-" * 80)

        if summary["param_fuzzy_failures"]:
            print(f"🔤 FUZZY TOKEN-F1 FAILURES ({len(summary['param_fuzzy_failures'])} cases):")
            for ff in summary["param_fuzzy_failures"][:5]:
                print(f"   [{ff['tool']}.{ff['param']}] (F1: {ff['f1']}):")
                print(f"       Pred: '{ff['pred']}'")
                print(f"       Gold: '{ff['gold']}'")
            print("-" * 80)

        print("💡 ACTIONABLE RECOMMENDATIONS:")
        if summary["error_categories"].get("Hash Mismatch (State Diverged)", 0) > 0:
            print("   1. [Part A/C]: Strict Pre-Call Read-Check — ensure write operations use correct entity IDs.")
        if summary["error_categories"].get("False Positive (Executed OOS sample)", 0) > 0:
            print("   2. [Part C]: OOS Gatekeeper — filter out requests when required apps are not installed.")
        if summary["param_exact_mismatches"]:
            print("   3. [Part B/C]: Format & Time normalization — unify ISO 8601 timestamps and exact casing for IDs.")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="LifeMate Diagnostic Error Analyzer")
    parser.add_argument("--submission", required=True, help="Path to submission.jsonl")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "holdout"])
    parser.add_argument("--output", default=None, help="Output path for diagnostic JSON")
    args = parser.parse_args()

    # Load splits
    splits_file = SPLITS_DIR / "splits.json"
    if not splits_file.exists():
        print("Splits file not found. Running split generation first...")
        from scripts.run_eda import main as run_eda_main
        run_eda_main()

    with open(splits_file, encoding="utf-8") as f:
        splits_data = json.load(f)

    # Handle both {"splits": {"dev": ...}} and {"dev": ...} formats
    if "splits" in splits_data:
        target_ids = set(splits_data["splits"].get(args.split, []))
    else:
        target_ids = set(splits_data.get(args.split, []))

    # Load gold samples
    all_samples = []
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    gold_samples = [s for s in all_samples if s["id"] in target_ids]
    print(f"Loaded {len(gold_samples)} gold samples for split '{args.split}'")

    # Load predictions
    preds = []
    with open(args.submission, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["id"] in target_ids:
                    preds.append(item)
    print(f"Loaded {len(preds)} predictions for split '{args.split}'")

    # Compute gold hashes via replay
    print("Computing gold state hashes via ToolEnv replay...")
    gold_hashes = compute_gold_hashes_via_replay(gold_samples, split="train", data_root=DATA_ROOT)

    analyzer = DiagnosticAnalyzer()
    summary = analyzer.analyze(preds, gold_samples, gold_hashes, split_name=args.split)
    analyzer.print_diagnostic_report(summary)

    out_file = Path(args.output) if args.output else OUTPUTS_DIR / f"diagnostic_{args.split}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Diagnostic report saved to {out_file}")


if __name__ == "__main__":
    main()
