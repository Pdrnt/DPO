from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_utils import load_jsonl, summarize_dataset, validate_dataset


DATASET_PATH = PROJECT_ROOT / "data" / "preferences.jsonl"


def main() -> None:
    print("==> Inspecionando dataset de preferências...\n")
    print(f"Arquivo: {DATASET_PATH}\n")

    records = load_jsonl(DATASET_PATH)
    summary = summarize_dataset(records)
    errors = validate_dataset(records)

    print("Resumo do dataset:")
    print(f"- Total de exemplos: {summary['total_examples']}")
    print(f"- Chaves obrigatórias: {', '.join(summary['required_keys'])}")

    if errors:
        print("\nForam encontrados problemas no dataset:")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)

    print("\nDataset válido.")
    print("\nAmostra do primeiro exemplo:")
    first = records[0]
    print(f"prompt   : {first['prompt']}")
    print(f"chosen   : {first['chosen']}")
    print(f"rejected : {first['rejected']}")


if __name__ == "__main__":
    main()