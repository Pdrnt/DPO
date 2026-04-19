import json
from pathlib import Path

from datasets import Dataset


REQUIRED_KEYS = {"prompt", "chosen", "rejected"}


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Linha {line_number} com JSON inválido: {exc}"
                ) from exc

            records.append(record)

    return records


def validate_record(record: dict, index: int) -> list[str]:
    errors = []

    missing_keys = REQUIRED_KEYS - record.keys()
    if missing_keys:
        errors.append(
            f"Registro {index}: faltam as chaves obrigatórias: {sorted(missing_keys)}"
        )

    for key in REQUIRED_KEYS:
        if key in record and not isinstance(record[key], str):
            errors.append(f"Registro {index}: o campo '{key}' deve ser string.")
        elif key in record and not record[key].strip():
            errors.append(f"Registro {index}: o campo '{key}' está vazio.")

    return errors


def validate_dataset(records: list[dict]) -> list[str]:
    errors = []

    if len(records) < 30:
        errors.append(
            f"Dataset possui {len(records)} exemplos. O mínimo exigido é 30."
        )

    for index, record in enumerate(records, start=1):
        errors.extend(validate_record(record, index))

    return errors


def summarize_dataset(records: list[dict]) -> dict:
    return {
        "total_examples": len(records),
        "required_keys": sorted(REQUIRED_KEYS),
    }


def load_preferences_dataset(path: str | Path) -> Dataset:
    records = load_jsonl(path)
    errors = validate_dataset(records)

    if errors:
        raise ValueError(
            "Dataset inválido:\n" + "\n".join(f"- {error}" for error in errors)
        )

    return Dataset.from_list(records)