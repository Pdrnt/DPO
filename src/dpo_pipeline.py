from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DATASET_PATH, MODEL_NAME
from src.data_utils import load_preferences_dataset


def load_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_models(model_name: str = MODEL_NAME):
    actor_model = AutoModelForCausalLM.from_pretrained(model_name)
    reference_model = AutoModelForCausalLM.from_pretrained(model_name)

    return actor_model, reference_model


def load_dataset(dataset_path=DATASET_PATH):
    return load_preferences_dataset(dataset_path)


def build_base_pipeline():
    dataset = load_dataset()
    tokenizer = load_tokenizer()
    actor_model, reference_model = load_models()

    return {
        "dataset": dataset,
        "tokenizer": tokenizer,
        "actor_model": actor_model,
        "reference_model": reference_model,
    }