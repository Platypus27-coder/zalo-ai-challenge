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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Maximum training steps to run (e.g. 250 for ~5.5h on T4). If > 0, overrides epochs.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--precision", choices=["4bit", "fp16", "auto"], default="auto",
                        help="Model precision: 'auto' (default: fp16 on Pascal/P100, 4bit on Ampere/Turing), 'fp16' (native fast float16), '4bit' (bitsandbytes QLoRA)")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Force disable 4-bit quantization and use native float16")
    parser.add_argument("--save-steps", type=int, default=200,
                        help="Save checkpoint every N steps (default: 200). If 0, save per epoch.")
    parser.add_argument("--device-map", default=None,
                        help="Device map placement (default: auto. Pins to GPU 0 if multi-GPU without P2P like Kaggle T4x2)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint in output_dir/checkpoints if available")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Automatically upload trained adapter to Hugging Face Hub")
    parser.add_argument("--hub-model-id", default=None,
                        help="Hugging Face repo ID, e.g. 'username/lifemate-qwen3-4b-lora'")
    parser.add_argument("--hub-token", default=None,
                        help="Hugging Face API write token (optional if already logged in via huggingface-cli)")
    parser.add_argument("--hub-private", action="store_true", default=True,
                        help="Make the Hugging Face repository private (default: True)")
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

    # Determine precision: 4-bit vs fp16
    if args.no_4bit or args.precision == "fp16":
        use_4bit = False
    elif args.precision == "4bit":
        use_4bit = True
    else:  # auto
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            # Pascal (SM 6.0, e.g. Tesla P100) or older lacks fast 4-bit tensor kernels in bitsandbytes
            if major <= 6:
                print("⚡ [Auto-Detect] Detected Pascal GPU (Compute Capability <= 6.x, e.g. Tesla P100).")
                print("   bitsandbytes 4-bit NF4 kernels lack hardware acceleration on Pascal, causing ~70s/step slowdown.")
                print("   Automatically switching to native FP16 (full speed ~0.8-1.5s/step, fits within 11.5 / 16 GB VRAM).")
                use_4bit = False
            else:
                use_4bit = True
        else:
            use_4bit = False

    # Determine device map placement
    # On multi-GPU setups without NVLink/P2P (e.g. Kaggle T4 x2), splitting a 4B model across GPUs
    # causes severe PCIe bottleneck (~72s/step) via CPU host memory transfers.
    # The 4B model easily fits into 1 single T4 GPU (~4.5 GB in 4-bit, ~11 GB in FP16 out of 15 GB VRAM).
    target_device_map = args.device_map or "auto"
    if args.device_map is None and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        can_p2p = False
        try:
            can_p2p = torch.cuda.can_device_access_peer(0, 1)
        except Exception:
            can_p2p = False
        
        if not can_p2p:
            print(f"⚡ [Multi-GPU Auto-Optimize] Detected {torch.cuda.device_count()} GPUs without Peer-to-Peer PCIe access (e.g. Kaggle T4 x2).")
            print("   Sharding a 4B model across GPUs without P2P causes ~70s/step bottleneck due to CPU RAM round-tripping.")
            print("   The model fits completely on GPU 0 (~4.5 GB in 4-bit, ~11 GB in FP16 out of 15 GB).")
            print("   Automatically pinning model to GPU 0 (cuda:0) -> 120x speedup (~0.4-0.6s/step)!")
            target_device_map = {"": 0}

    # 3. Load Model
    if use_4bit:
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
            device_map=target_device_map,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"\n3. Loading Base Model in Native FP16: {args.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map=target_device_map,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        model.enable_input_require_grads()

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
    effective_bs = max(1, args.batch_size * args.grad_accum)
    if args.max_steps > 0:
        total_steps = args.max_steps
        epochs = max(1, (total_steps * effective_bs) // len(tokenized_dataset) + 1)
        save_steps = min(args.save_steps, 100) if args.save_steps > 0 else 100
        print(f"🎯 Target Max Steps: {args.max_steps} (~{args.max_steps * 82 / 3600:.1f} hours on T4)")
    else:
        total_steps = (len(tokenized_dataset) // effective_bs) * args.epochs
        epochs = args.epochs
        save_steps = args.save_steps if args.save_steps > 0 else None

    warmup_steps = max(10, int(total_steps * 0.05))

    import inspect
    extra_training_args = {}
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "gradient_checkpointing_kwargs" in sig:
        extra_training_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    save_strategy = "steps" if save_steps else "epoch"
    optim = "adamw_torch" if not use_4bit else "paged_adamw_8bit"

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=True,
        logging_steps=25,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        optim=optim,
        gradient_checkpointing=True,
        **extra_training_args,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
    )

    # 6. Train
    checkpoints_dir = output_dir / "checkpoints"
    resume_checkpoint = None
    if args.resume and checkpoints_dir.exists():
        ckpts = sorted(list(checkpoints_dir.glob("checkpoint-*")), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
        if ckpts:
            resume_checkpoint = str(ckpts[-1])
            print(f"🔄 Resuming from latest checkpoint: {resume_checkpoint}")

    print("\n5. Training in progress on GPU...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 7. Save Final Adapter
    print(f"\n6. Saving final LoRA adapter to {output_dir}...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\n🎉 QLoRA training finished successfully! Adapter saved at: {output_dir}")

    # 8. Push to Hugging Face Hub (Optional)
    if args.push_to_hub or args.hub_model_id:
        hub_id = args.hub_model_id
        if not hub_id:
            print("⚠️ --push-to-hub requested but --hub-model-id not provided. Skipping push.")
        else:
            print(f"\n7. Pushing LoRA adapter to Hugging Face Hub: {hub_id}...")
            try:
                model.push_to_hub(
                    hub_id,
                    token=args.hub_token,
                    private=args.hub_private,
                )
                tokenizer.push_to_hub(
                    hub_id,
                    token=args.hub_token,
                    private=args.hub_private,
                )
                print(f"🚀 Successfully uploaded to Hugging Face Hub: https://huggingface.co/{hub_id}")
            except Exception as e:
                print(f"⚠️ Failed to push to Hugging Face Hub: {e}")
                print("   You can still upload later using: python scripts/push_to_hub.py")


if __name__ == "__main__":
    main()
