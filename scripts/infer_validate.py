from pathlib import Path
import math
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    MODEL_NAME,
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


def main() -> None:
    print("==> Iniciando validação por inferência...\n")

    model, tokenizer, model_path = load_model_and_tokenizer()

    print(f"Modelo carregado de: {model_path}")
    print(f"Prompt de validação: {VALIDATION_PROMPT}\n")

    chosen_logprob = compute_sequence_logprob(
        model, tokenizer, VALIDATION_PROMPT, VALIDATION_CHOSEN
    )
    rejected_logprob = compute_sequence_logprob(
        model, tokenizer, VALIDATION_PROMPT, VALIDATION_REJECTED
    )

    chosen_prob_proxy = math.exp(chosen_logprob / 100)
    rejected_prob_proxy = math.exp(rejected_logprob / 100)

    print("Comparação das respostas candidatas:")
    print(f"- chosen   logprob: {pretty_score(chosen_logprob)}")
    print(f"- rejected logprob: {pretty_score(rejected_logprob)}")
    print(f"- chosen   proxy score: {chosen_prob_proxy:.6f}")
    print(f"- rejected proxy score: {rejected_prob_proxy:.6f}\n")

    if chosen_logprob > rejected_logprob:
        print("Resultado: a resposta segura foi priorizada sobre a rejeitada.")
    else:
        print("Resultado: a resposta rejeitada ainda não foi totalmente suprimida.")

    print("\nResposta segura esperada:")
    print(VALIDATION_CHOSEN)

    print("\nResposta rejeitada comparada:")
    print(VALIDATION_REJECTED)


if __name__ == "__main__":
    main()