"""Part A — Master Evaluation CLI & Diagnostic Suite.

Comprehensive evaluation tool for LifeMate Agent submissions:
1. Runs Submission Linter (format, missing samples, 64-char sha256 hex, app installation)
2. Computes Official Scoring Metric (Exact state_hash, Token-F1 fuzzy > 0.6, Call F1)
3. Computes Loss Attribution Matrix (§4.2 of v3 Master Plan: ROI ranking & Top 2 actions)
4. State Diff Inspection for failing samples
5. Exports structured JSON and Markdown report to outputs/reports/

Usage:
    python scripts/evaluate.py --submission outputs/submissions/pred.jsonl --split dev
"""
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from config import DATA_ROOT, TRAIN_JSONL, TRAIN_CATALOG, OUTPUTS_DIR, SPLITS_DIR, FUZZY_THRESHOLD
from src.part_a.evaluator import (
    load_catalog_fuzzy_params, score_sample, evaluate_submission,
    compute_gold_hashes_via_replay
)
from src.part_a.submission_linter import SubmissionLinter
from src.part_a.loss_attribution import LossAttributor
from src.part_a.state_diff import diff_states_for_sample


def load_split_samples(split: str) -> list[dict]:
    """Load gold samples for the given split (train, dev, holdout)."""
    samples = []
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if split in ("dev", "holdout"):
        splits_file = SPLITS_DIR / "splits.json"
        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file not found at {splits_file}. Run scripts/run_eda.py first.")
        with open(splits_file, encoding="utf-8") as f:
            target_ids = set(json.load(f)["splits"].get(split, []))
        samples = [s for s in samples if s["id"] in target_ids]

    return samples


def run_evaluation(submission_path: Path, split: str = "dev", output_dir: Path | None = None) -> dict[str, Any]:
    print("=" * 80)
    print(f"LIFEMATE MASTER EVALUATION SUITE — SPLIT: {split.upper()}")
    print("=" * 80)
    print(f"Submission: {submission_path}")

    reports_dir = output_dir or (OUTPUTS_DIR / "reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # -------------------------------------------------------------
    # 1. RUN SUBMISSION LINTER
    # -------------------------------------------------------------
    print(f"\n[1/4] Running 7-Tier Submission Linter for '{split}'...")
    linter = SubmissionLinter(split=split)
    lint_report = linter.lint_file(submission_path)

    if not lint_report["passed"]:
        print(f"  ❌ Linter FAILED with {len(lint_report['critical_errors'])} critical error(s):")
        for err in lint_report["critical_errors"][:5]:
            print(f"     • {err}")
    else:
        print(f"  ✅ Linter PASSED! Checked {lint_report['stats']['total_samples']} samples, "
              f"{lint_report['stats']['valid_hashes']} valid SHA-256 hashes.")

    # -------------------------------------------------------------
    # 2. COMPUTE OFFICIAL EVALUATION SCORE
    # -------------------------------------------------------------
    print(f"\n[2/4] Loading gold labels & computing official metric on '{split}'...")
    gold_samples = load_split_samples(split)
    gold_by_id = {s["id"]: s for s in gold_samples}
    print(f"  Loaded {len(gold_samples)} gold samples for split '{split}'.")

    # Load submission entries for this split
    predictions = []
    with open(submission_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if entry["id"] in gold_by_id:
                    predictions.append(entry)

    print(f"  Matched {len(predictions)} / {len(gold_samples)} samples in submission file.")
    if len(predictions) < len(gold_samples):
        print(f"  ⚠️ Warning: {len(gold_samples) - len(predictions)} samples missing from submission!")

    fuzzy_params = load_catalog_fuzzy_params(TRAIN_CATALOG)
    print(f"  Computing gold state hashes via replay...")
    gold_hashes = compute_gold_hashes_via_replay(gold_samples, "train")

    eval_result = evaluate_submission(predictions, gold_samples, fuzzy_params, gold_hashes)

    final_score = eval_result["final_score"]
    hash_match_rate = eval_result["hash_match_rate"]
    type_counts = eval_result["type_counts"]

    print("\n" + "-" * 80)
    print(f"📊 EVALUATION SCORE: {final_score:.2f} / 100.00")
    print(f"   State Hash Match: {hash_match_rate:.2%}")
    print(f"   Output Breakdown: {type_counts}")
    print("-" * 80)

    # -------------------------------------------------------------
    # 3. COMPUTE LOSS ATTRIBUTION MATRIX & ROI
    # -------------------------------------------------------------
    print("\n[3/4] Computing Loss Attribution Matrix & Work Prioritizer (§4.2)...")
    attributor = LossAttributor(TRAIN_CATALOG)
    loss_result = attributor.attribute(
        predictions=predictions,
        gold_samples=gold_samples,
        gold_hashes=gold_hashes
    )

    attr_table = loss_result["attribution_table"]
    if attr_table:
        print("\n" + "-" * 80)
        print(f"{'Nhóm nguyên nhân gốc':<36} | {'Mẫu':<5} | {'Điểm mất':<9} | {'Cứu được':<9} | {'Giờ':<4} | {'ROI':<6}")
        print("-" * 80)
        for row in attr_table:
            print(f"{row['category']:<36} | {row['sample_count']:<5} | "
                  f"{row['total_lost_points']:<9.2f} | {row['expected_gain_points']:<9.2f} | "
                  f"{row['estimated_hours']:<4} | {row['roi']:<6.3f}")
        print("-" * 80)

        # Highlight Top 2 ROI actions
        sorted_by_roi = sorted(attr_table, key=lambda x: -x["roi"])
        print("\n🎯 TOP 2 ĐỀ XUẤT CAN THIỆP CÓ ROI CAO NHẤT:")
        for idx, item in enumerate(sorted_by_roi[:2], 1):
            print(f"  {idx}. {item['category']} (ROI={item['roi']:.3f})")
            print(f"     -> Hướng xử lý: {item['fix_direction']}")
    else:
        print("  🎉 Hoàn hảo! Không có mẫu nào bị mất điểm (Total Lost = 0.00 points).")

    # -------------------------------------------------------------
    # 4. STATE DIFF INSPECTION FOR FAILING SAMPLES
    # -------------------------------------------------------------
    print("\n[4/4] Inspecting State Divergences for Failing Samples...")
    failing_samples = []
    for pred in predictions:
        sid = pred["id"]
        p_hash = pred.get("state_hash", "")
        g_hash = gold_hashes.get(sid, "")
        if p_hash != g_hash and sid in gold_by_id:
            gold_calls = gold_by_id[sid].get("label", {}).get("tool_calls", [])
            # If gold is non-empty, diff state
            if gold_calls:
                diff = diff_states_for_sample(
                    split="train",
                    sample_id=sid,
                    pred_tool_calls=pred.get("tool_calls", []),
                    gold_tool_calls=gold_calls
                )
                if diff.get("has_divergence"):
                    failing_samples.append({
                        "id": sid,
                        "summary": diff.get("summary", ""),
                        "diff": diff.get("diff", {})
                    })

    if failing_samples:
        print(f"  Found {len(failing_samples)} samples with divergent app state. Sample diffs:")
        for fs in failing_samples[:3]:
            print(f"     • [{fs['id']}]: {fs['summary']}")
    else:
        print("  ✅ 100% state_hash match achieved across all evaluated samples!")

    # -------------------------------------------------------------
    # 5. EXPORT COMPREHENSIVE REPORTS (JSON + MARKDOWN)
    # -------------------------------------------------------------
    full_report = {
        "timestamp": timestamp,
        "submission_file": str(submission_path),
        "split": split,
        "linter": lint_report,
        "scoring": {
            "final_score": final_score,
            "hash_match_rate": hash_match_rate,
            "type_counts": type_counts,
            "n_samples": len(predictions),
            "n_gold": len(gold_samples)
        },
        "loss_attribution": loss_result,
        "failing_state_diffs": failing_samples[:10]
    }

    json_report_path = reports_dir / f"eval_{split}_{timestamp}.json"
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    md_report_path = reports_dir / f"eval_{split}_{timestamp}.md"
    _write_markdown_report(md_report_path, full_report)

    print("\n" + "=" * 80)
    print(f"✅ Báo cáo đánh giá đã được lưu trữ:")
    print(f"   • JSON: {json_report_path}")
    print(f"   • MD:   {md_report_path}")
    print("=" * 80)

    return full_report


def _write_markdown_report(path: Path, r: dict):
    scoring = r["scoring"]
    loss = r["loss_attribution"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# LifeMate Evaluation Report — {r['split'].upper()}\n")
        f.write(f"- **Thời gian:** {r['timestamp']}\n")
        f.write(f"- **Tệp dự đoán:** `{r['submission_file']}`\n\n")

        f.write("## 1. Kết Quả Điểm Số Chính Thức\n")
        f.write(f"- **Final Score (thang 100):** **`{scoring['final_score']:.2f}`**\n")
        f.write(f"- **Tỷ lệ khớp State Hash:** `{scoring['hash_match_rate']:.2%}`\n")
        f.write(f"- **Số mẫu đánh giá:** `{scoring['n_samples']} / {scoring['n_gold']}`\n")
        f.write(f"- **Phân loại kết quả:** `{json.dumps(scoring['type_counts'], ensure_ascii=False)}`\n\n")

        f.write("## 2. Ma Trận Điểm Mất (Loss Attribution Matrix)\n")
        f.write("| Nhóm nguyên nhân gốc | Số mẫu | Điểm mất | Cứu được | Giờ công | ROI |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for row in loss.get("attribution_table", []):
            f.write(f"| {row['category']} | {row['sample_count']} | {row['total_lost_points']:.2f} | "
                    f"{row['expected_gain_points']:.2f} | {row['estimated_hours']}h | {row['roi']:.3f} |\n")
        f.write("\n")

        f.write("## 3. Kiểm Định Định Dạng (Submission Linter)\n")
        lint = r["linter"]
        status_emoji = "✅ ĐẠT" if lint["passed"] else "❌ KHÔNG ĐẠT"
        f.write(f"- **Trạng thái:** {status_emoji}\n")
        f.write(f"- **Tổng mẫu:** `{lint['stats']['total_samples']}`\n")
        f.write(f"- **Mẫu OOS (rỗng):** `{lint['stats']['oos_count']}`\n")
        f.write(f"- **Mã hash hợp lệ:** `{lint['stats']['valid_hashes']}`\n")
        if lint["critical_errors"]:
            f.write("\n### Lỗi nghiêm trọng:\n")
            for err in lint["critical_errors"]:
                f.write(f"- ❌ {err}\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="LifeMate Master Evaluation CLI")
    parser.add_argument("--submission", required=True, help="Path to submission.jsonl")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "holdout"])
    parser.add_argument("--output-dir", default=None, help="Directory to save reports")
    args = parser.parse_args()

    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"Error: Submission file '{sub_path}' does not exist.")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None
    run_evaluation(submission_path=sub_path, split=args.split, output_dir=out_dir)


if __name__ == "__main__":
    main()
