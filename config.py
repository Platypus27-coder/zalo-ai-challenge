"""Global configuration for LifeMate Agent project."""
from pathlib import Path

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "train+public_test"
TRAIN_DIR = DATA_ROOT / "train"
PUBLIC_TEST_DIR = DATA_ROOT / "public_test"

# Data files
TRAIN_JSONL = TRAIN_DIR / "train.jsonl"
TRAIN_CATALOG = TRAIN_DIR / "train_tools_catalog.json"
TRAIN_ENV_DIR = TRAIN_DIR / "train_environment"
TRAIN_AUDIO_DIR = TRAIN_DIR / "audio"

PUBLIC_TEST_JSONL = PUBLIC_TEST_DIR / "public_test.jsonl"
PUBLIC_TEST_CATALOG = PUBLIC_TEST_DIR / "public_test_tools_catalog.json"
PUBLIC_TEST_ENV_DIR = PUBLIC_TEST_DIR / "public_test_environment"
PUBLIC_TEST_AUDIO_DIR = PUBLIC_TEST_DIR / "audio"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "splits"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"

# Ensure output dirs exist
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# === Constants ===
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
DEV_RATIO = 0.15
HOLDOUT_RATIO = 0.15

# Scoring
FUZZY_THRESHOLD = 0.6  # token-F1 threshold for x-match: fuzzy

# Runtime budget (seconds)
MAX_TIME_PER_SAMPLE = 30
TARGET_P95_TIME = 25
ASR_BUDGET = 5
LLM_BUDGET = 16
TOOL_BUDGET = 2
RESERVE_BUDGET = 7

# Model constraints
MAX_TOTAL_PARAMS = 6_000_000_000   # 6B total
MAX_LLM_PARAMS = 5_000_000_000    # 5B for LLM
MAX_CONTEXT_TOKENS = 16384         # 16K context window
