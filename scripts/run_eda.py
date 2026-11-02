"""Run EDA on training data — produces comprehensive analysis report."""
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.part_a.eda import load_data, load_catalog, analyze_samples, analyze_catalog, print_report
from src.part_a.data_splitter import load_train_data, create_splits, print_split_stats, save_splits
from config import OUTPUTS_DIR


def main():
    print("=" * 80)
    print("LIFEMATE — EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Step 1: Analyze training data
    print("\n[1/3] Loading and analyzing training data...")
    samples = load_data()
    catalog = load_catalog()
    
    sample_stats = analyze_samples(samples)
    catalog_stats = analyze_catalog(catalog)
    
    print_report(sample_stats, catalog_stats)
    
    # Step 2: Create and save splits
    print("\n[2/3] Creating train/dev/holdout splits...")
    splits = create_splits(samples)
    print_split_stats(samples, splits)
    
    save_path = save_splits(splits)
    print(f"Splits saved to {save_path}")
    
    # Step 3: Save full report
    print("\n[3/3] Saving report...")
    import json
    report_path = OUTPUTS_DIR / "eda_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples": sample_stats, 
            "catalog": catalog_stats,
            "splits": {k: len(v) for k, v in splits.items()}
        }, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {report_path}")
    
    print("\n✅ EDA complete!")


if __name__ == "__main__":
    main()
