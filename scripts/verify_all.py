"""Comprehensive verification script to ensure all components of LifeMate pipeline are working."""
import sys
import json
from pathlib import Path

# Fix Windows console UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_ROOT, TRAIN_JSONL, TRAIN_CATALOG, SPLITS_DIR, OUTPUTS_DIR


def test_part_a_components():
    print("=" * 60)
    print("VERIFYING PART A COMPONENTS (EVALUATION & DATA)")
    print("=" * 60)
    
    # 1. Check splits
    splits_file = SPLITS_DIR / "splits.json"
    assert splits_file.exists(), "Splits file missing!"
    with open(splits_file, encoding="utf-8") as f:
        splits = json.load(f)
    print(f"✅ Splits verified: Train={splits['counts']['train']}, Dev={splits['counts']['dev']}, Holdout={splits['counts']['holdout']}")
    
    # 2. Check catalog & fuzzy params
    from src.part_a.evaluator import load_catalog_fuzzy_params, token_f1
    fuzzy_params = load_catalog_fuzzy_params(TRAIN_CATALOG)
    assert len(fuzzy_params) > 0, "Fuzzy params empty!"
    print(f"✅ Catalog indexed: {len(fuzzy_params)} tools have fuzzy matching parameters")
    
    # 3. Test token-F1
    score = token_f1("mai chạy bộ 9 giờ", "mai chạy bộ 09:00")
    assert score > 0.5, f"Token F1 error: {score}"
    print(f"✅ Token-F1 algorithm functioning properly (score={score:.2f})")
    
    # 4. Check schema validator
    from src.part_a.validator import load_catalog, get_tool_specs, validate_call
    catalog = load_catalog(TRAIN_CATALOG)
    specs = get_tool_specs(catalog)
    val = validate_call("smart_home.light_set", {"device_id": "light_1", "power": "on"}, specs)
    assert val["ok"], f"Validation failed: {val}"
    print("✅ Pre-call schema validator functioning properly")

    # 5. Check Submission Linter
    from src.part_a.submission_linter import SubmissionLinter
    linter = SubmissionLinter(split="dev")
    assert len(linter.expected_samples) == 265, f"Expected 265 dev samples, got {len(linter.expected_samples)}"
    print(f"✅ Submission Linter indexed {len(linter.expected_samples)} expected dev samples")

    # 6. Check Loss Attributor
    from src.part_a.loss_attribution import LossAttributor
    attributor = LossAttributor(TRAIN_CATALOG)
    assert attributor is not None
    print("✅ Loss Attribution Matrix initialized successfully")


def test_part_b_components():
    print("\n" + "=" * 60)
    print("VERIFYING PART B COMPONENTS (ASR & NORMALIZER)")
    print("=" * 60)
    from src.part_b.normalizer import normalize_transcript
    test_text = "Nạp   50 nghìn   điện thoại ngày 12/03/2026"
    norm = normalize_transcript(test_text)
    assert "Nạp" in norm, f"Normalization error: {norm}"
    print(f"✅ Text normalizer: '{test_text}' -> '{norm}'")


def test_part_c_components():
    print("\n" + "=" * 60)
    print("VERIFYING PART C COMPONENTS (AGENT & PROMPT)")
    print("=" * 60)
    from src.part_c.tool_filter import filter_tools_for_sample, get_all_tool_specs
    from src.part_c.agent import create_agent
    
    catalog = json.load(open(TRAIN_CATALOG, encoding="utf-8"))
    specs = get_all_tool_specs(catalog)
    sample = {
        "id": "verify_sample",
        "context": {
            "installed_apps": {
                "smart_home": {"tools": ["smart_home.get_status", "smart_home.light_set"]}
            }
        }
    }
    filtered = filter_tools_for_sample(sample, specs)
    assert "smart_home.light_set" in filtered, "Filtered tools missing smart_home.light_set"
    print(f"✅ Tool filter: Reduced catalog to {len(filtered)} allowed tools")
    
    agent = create_agent(mode="rule")
    assert agent is not None
    print("✅ Rule-based agent baseline instantiated successfully")


def main():
    test_part_a_components()
    test_part_b_components()
    test_part_c_components()
    print("\n" + "=" * 60)
    print("🎉 ALL SYSTEMS GO — PIPELINE READY FOR SPECIALIZATION")
    print("=" * 60)


if __name__ == "__main__":
    main()
