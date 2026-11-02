# LifeMate Agent — Zalo AI Challenge 2026
# Voice-driven AI Agent: Vietnamese speech → tool calls

## Project Structure

```
LifeMateAgent/
├── environment.yml              # Conda environment
├── config.py                    # Global config (paths, constants)
├── train+public_test/           # Official data (DO NOT MODIFY)
│   ├── train/
│   │   ├── train.jsonl
│   │   ├── train_tools_catalog.json
│   │   ├── train_environment/
│   │   ├── audio/
│   │   └── zaic_tools/
│   └── public_test/
│       ├── public_test.jsonl
│       ├── public_test_tools_catalog.json
│       ├── public_test_environment/
│       ├── audio/
│       └── zaic_tools/
│
├── src/
│   ├── __init__.py
│   ├── part_a/                  # Evaluation, Execution, Integration (Owner: A)
│   │   ├── __init__.py
│   │   ├── data_splitter.py     # Train/Dev/Holdout split
│   │   ├── evaluator.py         # Local scoring (hash, F1, token-F1)
│   │   ├── runner.py            # Pipeline runner (ASR → Agent → ToolEnv)
│   │   ├── validator.py         # Pre-call schema validator
│   │   └── eda.py               # Exploratory data analysis
│   │
│   ├── part_b/                  # ASR, Input, Error Analysis (Owner: B)
│   │   ├── __init__.py
│   │   ├── asr_module.py        # ASR inference (faster-whisper)
│   │   ├── normalizer.py        # Text normalization (NFC, preserve IDs)
│   │   └── asr_benchmark.py     # WER/CER evaluation
│   │
│   └── part_c/                  # Agent, Reasoning, SFT (Owner: C)
│       ├── __init__.py
│       ├── agent.py             # LLM Agent (tool-calling loop)
│       ├── prompt.py            # Prompt templates
│       └── tool_filter.py       # Filter tools by installed_apps
│
├── scripts/
│   ├── run_baseline.py          # End-to-end baseline
│   ├── run_eda.py               # Data analysis script
│   └── generate_submission.py   # Generate submission.jsonl
│
├── outputs/                     # Experiment outputs
│   ├── splits/                  # Saved train/dev/holdout IDs
│   ├── submissions/             # Generated submission files
│   └── checkpoints/             # Model checkpoints / LoRA adapters
│
└── notebooks/                   # For Colab/Kaggle experiments
    ├── 00_colab_drive_master.ipynb   # Master Colab setup + Google Drive auto-sync
    ├── 01_asr_colab_kaggle.ipynb     # Part B: ASR & Whisper benchmark
    └── 02_agent_colab_kaggle.ipynb   # Part C: Qwen Agent & sequential tool-calling
```

## Quick Start (Local)

```bash
conda env create -f environment.yml
conda activate lifemate
python scripts/run_eda.py          # Analyze train data & create splits
python scripts/run_baseline.py     # Run baseline gold replay (100% check)
python scripts/verify_all.py       # Master sanity test
```

## Running on Google Colab (GPU T4) + Google Drive Checkpoints

Open [notebooks/00_colab_drive_master.ipynb](file:///D:/code/.vscode/LifeMateAgent/notebooks/00_colab_drive_master.ipynb):
1. Mount Google Drive (`/content/drive`).
2. Run code & data on high-speed Colab NVMe SSD (`/content/LifeMateAgent`).
3. Auto-symlink `outputs/checkpoints` and `outputs/submissions` to Google Drive (`/content/drive/MyDrive/LifeMate_Artifacts/`).
4. All checkpoints and submission files persist permanently on Drive even if Colab disconnects.

## Detailed Member Action Plans
- **Lead / Member A**: Evaluation, Loss Attribution Matrix, State Diff, Submission Linter.
- **Member B (ASR Specialist)**: [Plan_Member_B_ASR.md](file:///D:/code/.vscode/LifeMateAgent/Plan_Member_B_ASR.md)
- **Member C (Agent LLM Specialist)**: [Plan_Member_C_Agent_LLM.md](file:///D:/code/.vscode/LifeMateAgent/Plan_Member_C_Agent_LLM.md)
