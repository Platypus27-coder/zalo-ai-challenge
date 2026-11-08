"""Push trained LoRA adapter to Hugging Face Hub with metadata & model card.

Supports both interactive login and explicit token.
Default repository visibility is PRIVATE to safeguard competition weights.

Usage:
    python scripts/push_to_hub.py \
        --adapter-dir /content/drive/MyDrive/LifeMateAgent/adapters/qwen3_4b_lora \
        --repo-id your_username/lifemate-qwen3-4b-lora \
        --private
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def generate_model_card(repo_id: str, base_model: str) -> str:
    """Generate a clean Hugging Face model card README.md."""
    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
- generated_from_trainer
- peft
- lora
- zaic2026
- lifemate
- voice-agent
datasets:
- zaic2026-lifemate-sft
pipeline_tag: text-generation
---

# LifeMate Voice-driven AI Agent — LoRA Adapter

This LoRA adapter was fine-tuned on **{base_model}** for the **Zalo AI Challenge 2026 (LifeMate: Voice-driven AI Agent)**.

## Model Highlights
- **Base Model:** `{base_model}` (< 5.0B parameters)
- **Method:** 4-bit QLoRA (Rank=16, Alpha=32, target modules: all linear layers)
- **Prompt Loss Masking:** Loss is strictly computed on agent actions (`CALL:` / `DONE`) with system & user prompt masked (`-100`).
- **Target Task:** Conversational multi-turn tool calling across 10 lifestyle application domains (contacts, alarm, finance, ride, home, media, flight, calendar, notes, food).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "{base_model}"
adapter_repo = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, adapter_repo)
model.eval()
```
"""


def main():
    parser = argparse.ArgumentParser(description="Upload trained LoRA adapter to Hugging Face Hub")
    parser.add_argument(
        "--adapter-dir",
        default="/content/drive/MyDrive/LifeMateAgent/adapters/qwen3_4b_lora",
        help="Path to directory containing adapter_model.safetensors and adapter_config.json",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face repo ID, e.g. 'your_username/lifemate-qwen3-4b-lora'",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model ID on Hugging Face Hub",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API write token (optional if logged in via huggingface-cli)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repo public (default: private)",
    )
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        # Fallback to local outputs
        local_cand = Path("outputs/adapters/qwen3_4b_lora")
        if local_cand.exists():
            adapter_dir = local_cand
        else:
            print(f"❌ Adapter directory not found: {args.adapter_dir}")
            sys.exit(1)

    print("=" * 60)
    print("PUSHING LORA ADAPTER TO HUGGING FACE HUB")
    print(f"Adapter Dir:   {adapter_dir}")
    print(f"Target Repo:   {args.repo_id}")
    print(f"Visibility:    {'Public' if args.public else 'Private 🔒 (Protected)'}")
    print("=" * 60)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("❌ huggingface_hub not installed. Please run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=args.token)

    # 1. Create repo if not exists
    print(f"\n1. Ensuring repo exists on Hugging Face: {args.repo_id}...")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=args.token,
            private=not args.public,
            exist_ok=True,
        )
        print("   ✅ Repo is ready.")
    except Exception as e:
        print(f"   ⚠️ Repo verification returned: {e}")

    # 2. Write/Update Model Card README.md if not already present
    readme_path = adapter_dir / "README.md"
    if not readme_path.exists():
        readme_content = generate_model_card(args.repo_id, args.base_model)
        readme_path.write_text(readme_content, encoding="utf-8")
        print("   ✅ Created professional Hugging Face Model Card (README.md)")

    # 3. Upload entire adapter folder to Hugging Face
    print(f"\n2. Uploading files from {adapter_dir} to https://huggingface.co/{args.repo_id}...")
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        token=args.token,
    )

    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Model adapter is now hosted on Hugging Face Hub:")
    print(f"👉 Link: https://huggingface.co/{args.repo_id}")
    print("=" * 60)
    print("\nTo use it in your submission or agent anywhere:")
    print(f"  --adapter-path {args.repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
