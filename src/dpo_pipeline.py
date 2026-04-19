import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from src.config import (
    BETA,
    DATASET_PATH,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    MODEL_NAME,
    NUM_TRAIN_EPOCHS,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    SAVE_STEPS,
    TRAIN_OUTPUT_DIR,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
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


def detect_optimizer() -> str:
    if torch.cuda.is_available():
        return "paged_adamw_32bit"
    return "adamw_torch"


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


def build_training_arguments(output_dir: str | None = None) -> DPOConfig:
    final_output_dir = str(output_dir or TRAIN_OUTPUT_DIR)
    optimizer_name = detect_optimizer()

    return DPOConfig(
        output_dir=final_output_dir,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        remove_unused_columns=False,
        report_to="none",
        fp16=False,
        bf16=False,
        optim=optimizer_name,
        beta=BETA,
    )


def build_dpo_trainer(training_args: DPOConfig | None = None):
    pipeline = build_base_pipeline()

    if training_args is None:
        training_args = build_training_arguments()

    trainer = DPOTrainer(
        model=pipeline["actor_model"],
        ref_model=pipeline["reference_model"],
        args=training_args,
        train_dataset=pipeline["dataset"],
        processing_class=pipeline["tokenizer"],
    )

    return trainer