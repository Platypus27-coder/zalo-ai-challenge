"""Part A — Offline Model Weights Downloader.

Pre-downloads required model weights for 100% offline Docker packaging:
1. faster-whisper large-v3-turbo (~800M params, CTranslate2 format)
2. Qwen2.5-3B-Instruct (~3B params, HuggingFace format)

Usage:
    python scripts/download_models.py --target-dir models/
"""
import sys
import argparse
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_whisper(target_dir: Path, model_name: str = "large-v3-turbo"):
    print(f"\n[1/2] Downloading faster-whisper model '{model_name}'...")
    whisper_dir = target_dir / "whisper" / model_name
    whisper_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from faster_whisper import download_model
        downloaded_path = download_model(model_name, output_dir=str(whisper_dir))
        print(f"  ✅ faster-whisper saved to: {downloaded_path}")
    except Exception as e:
        print(f"  ❌ Error downloading faster-whisper: {e}")
        print("  💡 Tip: Install faster-whisper with `pip install faster-whisper` and ensure internet connection.")


def download_llm(target_dir: Path, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    print(f"\n[2/2] Downloading HuggingFace LLM '{model_name}'...")
    safe_name = model_name.replace("/", "_")
    llm_dir = target_dir / "llm" / safe_name
    llm_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(llm_dir),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            local_dir_use_symlinks=False
        )
        print(f"  ✅ LLM model saved to: {downloaded_path}")
    except Exception as e:
        print(f"  ❌ Error downloading LLM: {e}")
        print("  💡 Tip: Install huggingface_hub with `pip install huggingface_hub` and ensure internet connection.")


def main():
    parser = argparse.ArgumentParser(description="Download model weights for offline Docker bundling")
    parser.add_argument("--target-dir", default=str(PROJECT_ROOT / "models"),
                        help="Target directory to store downloaded models")
    parser.add_argument("--whisper-model", default="large-v3-turbo",
                        help="Whisper model name (e.g. large-v3-turbo, medium)")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct",
                        help="LLM model repo ID")
    parser.add_argument("--only-whisper", action="store_true")
    parser.add_argument("--only-llm", action="store_true")
    args = parser.parse_args()

    target_path = Path(args.target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("LIFEMATE — OFFLINE MODEL WEIGHTS DOWNLOADER")
    print(f"Target directory: {target_path.resolve()}")
    print("=" * 80)

    if not args.only_llm:
        download_whisper(target_path, args.whisper_model)

    if not args.only_whisper:
        download_llm(target_path, args.llm_model)

    print("\n" + "=" * 80)
    print("🎉 Model downloading step complete. Models ready for offline container bundling!")
    print("=" * 80)


if __name__ == "__main__":
    main()
