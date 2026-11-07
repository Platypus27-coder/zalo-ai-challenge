"""Prepare Supervised Fine-Tuning (SFT) trajectory dataset from train.jsonl.

Converts 1,545 ground-truth trajectories with intermediate tool_results into
multi-turn ChatML conversations for Qwen3-4B instruction tuning.

Each conversation turn mimics the exact runtime agent prompt:
- Step 0: User request -> CALL: {gold_tool_1}
- Step 1: User request + Obs 1 -> CALL: {gold_tool_2}
- ...
- Step N: User request + Obs 1..N -> DONE
- OOS: User request -> DONE
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_ROOT
from src.part_c.prompt import SYSTEM_PROMPT, build_user_prompt
from src.part_c.tool_filter import filter_tools_for_sample, format_tools_for_prompt


def main():
    train_split = "train"
    train_jsonl = DATA_ROOT / train_split / f"{train_split}.jsonl"
    catalog_path = DATA_ROOT / train_split / f"{train_split}_tools_catalog.json"
    output_dir = PROJECT_ROOT / "outputs" / "sft_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "lifemate_agent_sft.jsonl"

    print(f"Loading catalog from {catalog_path}...")
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    all_tools = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            all_tools[tool["name"]] = tool
    all_tools["sys.ask_missing_info"] = {
        "name": "sys.ask_missing_info",
        "description": "Báo thiếu tham số bắt buộc không thể suy ra từ ngữ cảnh hay observation",
        "parameters": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string"},
                "params": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["tool_name", "params"]
        }
    }

    print(f"Loading train samples from {train_jsonl}...")
    samples = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"  {len(samples)} samples loaded.")
    print("Generating multi-turn SFT dataset...")

    total_turns = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for s_idx, sample in enumerate(samples):
            allowed_tools = filter_tools_for_sample(sample, all_tools)
            tools_text = format_tools_for_prompt(allowed_tools)

            transcript = sample.get("audio_text", "")
            context = sample.get("context", {})
            history = sample.get("history", [])
            label = sample.get("label", {})
            gold_calls = label.get("tool_calls", [])
            gold_results = label.get("tool_results", [])

            # Generate individual step pairs
            observations = []
            for step in range(len(gold_calls) + 1):
                user_prompt = build_user_prompt(
                    transcript=transcript,
                    context=context,
                    history=history,
                    tools_text=tools_text,
                    observations=observations if observations else None,
                )

                if step < len(gold_calls):
                    target = f"CALL: {json.dumps(gold_calls[step], ensure_ascii=False)}"
                    # Append observation for next step
                    res = gold_results[step] if step < len(gold_results) else {"ok": True}
                    observations.append({
                        "call": gold_calls[step],
                        "result": res,
                    })
                else:
                    target = "DONE"

                # SFT sample in ChatML format
                sft_entry = {
                    "sample_id": sample.get("id"),
                    "step": step,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": target}
                    ]
                }
                fout.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
                total_turns += 1

    print(f"\n🎉 Successfully created {total_turns} training turns from {len(samples)} samples!")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
