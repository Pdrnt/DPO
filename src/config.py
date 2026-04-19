from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "preferences.jsonl"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_NAME = "sshleifer/tiny-gpt2"
BETA = 0.1

MAX_LENGTH = 256
MAX_PROMPT_LENGTH = 128