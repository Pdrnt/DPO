from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, TRAIN_OUTPUT_DIR
from src.dpo_pipeline import build_dpo_trainer


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==> Iniciando treinamento DPO...")
    print(f"Diretório de saída: {TRAIN_OUTPUT_DIR}")

    trainer = build_dpo_trainer()

    print("==> Trainer construído com sucesso.")
    print(f"Total de exemplos de treino: {len(trainer.train_dataset)}")

    trainer.train()

    print("==> Treinamento finalizado.")
    print("==> Salvando modelo e tokenizer...")

    trainer.save_model(str(TRAIN_OUTPUT_DIR))
    trainer.processing_class.save_pretrained(str(TRAIN_OUTPUT_DIR))

    print(f"==> Artefatos salvos em: {TRAIN_OUTPUT_DIR}")


if __name__ == "__main__":
    main()