from pathlib import Path
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    MODEL_NAME,
    OUTPUTS_DIR,
    TRAIN_OUTPUT_DIR,
    VALIDATION_CHOSEN,
    VALIDATION_PROMPT,
    VALIDATION_REJECTED,
)


def load_model_and_tokenizer():
    model_path = str(TRAIN_OUTPUT_DIR) if TRAIN_OUTPUT_DIR.exists() else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, model_path


def compute_sequence_logprob(model, tokenizer, prompt: str, answer: str) -> float:
    full_text = prompt + "\n" + answer
    prompt_text = prompt + "\n"

    full_ids = tokenizer(full_text, return_tensors="pt")
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")

    input_ids = full_ids["input_ids"]
    attention_mask = full_ids["attention_mask"]

    labels = input_ids.clone()
    prompt_length = prompt_ids["input_ids"].shape[1]
    labels[:, :prompt_length] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    valid_tokens = (labels != -100).sum().item()
    loss = outputs.loss.item()

    return -loss * valid_tokens


def pretty_score(logprob: float) -> str:
    return f"{logprob:.4f}"


def build_report(
    model_path: str,
    chosen_logprob: float,
    rejected_logprob: float,
    chosen_prob_proxy: float,
    rejected_prob_proxy: float,
) -> str:
    if chosen_logprob > rejected_logprob:
        result_line = "Resultado: a resposta segura foi priorizada sobre a rejeitada."
    else:
        result_line = "Resultado: a resposta rejeitada ainda não foi totalmente suprimida."

    report = f"""Relatório de validação DPO

Modelo carregado de: {model_path}
Prompt de validação: {VALIDATION_PROMPT}

Comparação das respostas candidatas:
- chosen   logprob: {pretty_score(chosen_logprob)}
- rejected logprob: {pretty_score(rejected_logprob)}
- chosen   proxy score: {chosen_prob_proxy:.6f}
- rejected proxy score: {rejected_prob_proxy:.6f}

{result_line}

Resposta segura esperada:
{VALIDATION_CHOSEN}

Resposta rejeitada comparada:
{VALIDATION_REJECTED}
"""
    return report


def main() -> None:
    print("==> Iniciando validação por inferência...\n")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_path = load_model_and_tokenizer()

    chosen_logprob = compute_sequence_logprob(
        model, tokenizer, VALIDATION_PROMPT, VALIDATION_CHOSEN
    )
    rejected_logprob = compute_sequence_logprob(
        model, tokenizer, VALIDATION_PROMPT, VALIDATION_REJECTED
    )

    chosen_prob_proxy = math.exp(chosen_logprob / 100)
    rejected_prob_proxy = math.exp(rejected_logprob / 100)

    report = build_report(
        model_path=model_path,
        chosen_logprob=chosen_logprob,
        rejected_logprob=rejected_logprob,
        chosen_prob_proxy=chosen_prob_proxy,
        rejected_prob_proxy=rejected_prob_proxy,
    )

    print(report)

    report_path = OUTPUTS_DIR / "validation_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()