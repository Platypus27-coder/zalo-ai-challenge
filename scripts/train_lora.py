"""Fine-tune Qwen3-4B-Instruct-2507 on LifeMate Agent SFT Trajectories with QLoRA.

Target Hardware: Google Colab GPU T4 (15 GB VRAM)
Memory Footprint: ~7.5 GB VRAM with 4-bit QLoRA + gradient checkpointing
Training Duration: ~25 - 35 minutes on 5,086 training turns

Usage:
    python scripts/train_lora.py \
        --data outputs/sft_data/lifemate_agent_sft.jsonl \
        --output-dir /content/drive/MyDrive/LifeMateAgent/adapters/qwen3_4b_lora \
        --epochs 2 \
        --batch-size 2 \
        --grad-accum 4 \
        --lr 2e-4
"""
import argparse
import json
import os
from pathlib import Path
import sys

# Ensure UTF-8 output
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Qwen3-4B Agent")
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data", default="outputs/sft_data/lifemate_agent_sft.jsonl")
    parser.add_argument("--output-dir", default="/content/drive/MyDrive/LifeMateAgent/adapters/qwen3_4b_lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ SFT data file not found: {data_path}")
        print("Please run first: python scripts/prepare_sft_data.py")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STARTING QLORA SFT TRAINING PIPELINE")
    print(f"Base Model:    {args.model_name}")
    print(f"Data Path:     {data_path}")
    print(f"Output Dir:    {output_dir}")
    print(f"Epochs:        {args.epochs}")
    print(f"Effective BS:  {args.batch_size * args.grad_accum}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    # 1. Load Tokenizer
    print("\n1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset
    print(f"\n2. Loading dataset from {data_path}...")
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"   Loaded {len(records)} training turns.")

    def format_and_tokenize(example):
        messages = example["messages"]
        # Apply Qwen ChatML template
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        enc = tokenizer(full_text, truncation=True, max_length=args.max_seq_len, padding=False)
        
        labels = list(enc["input_ids"])
        prompt_len = min(len(prompt_ids), len(labels))
        # Mask out prompt tokens with -100 (loss only on Assistant response)
        for i in range(prompt_len):
            labels[i] = -100
        # Guard: Ensure at least one label token is active if truncation occurred
        if all(x == -100 for x in labels):
            labels[-1] = enc["input_ids"][-1]
        enc["labels"] = labels
        return enc

    raw_dataset = Dataset.from_list(records)
    print("   Tokenizing dataset with ChatML template...")
    tokenized_dataset = raw_dataset.map(format_and_tokenize, remove_columns=raw_dataset.column_names, num_proc=1)

    # 3. Load Model in 4-bit NF4
    print(f"\n3. Loading 4-bit Quantized Base Model: {args.model_name}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    # 4. Attach LoRA Adapter
    print(f"\n4. Attaching LoRA Adapter (r={args.lora_r}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )

    # 6. Train
    print("\n5. Training in progress on GPU...")
    trainer.train()

    # 7. Save Final Adapter
    print(f"\n6. Saving final LoRA adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n🎉 QLoRA training finished successfully! Adapter saved at: {output_dir}")


if __name__ == "__main__":
    main()
