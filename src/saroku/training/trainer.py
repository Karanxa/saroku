"""
saroku safety model trainer.

Fine-tunes Qwen2.5-0.5B-Instruct with LoRA to produce a specialized
safety classifier for agent actions.

Usage:
    python -m saroku.training.trainer \
        --output-dir ./models/saroku-safety-0.5b \
        --epochs 3

The output is a merged model ready for inference via saroku's local_judge.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)

from saroku.training.data_generator import generate_dataset

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA config — target the attention and MLP projection layers
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # rank — higher = more capacity, more VRAM
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)


def _format_for_training(example: dict, tokenizer) -> dict:
    """
    Format a prompt+completion pair into input_ids + labels.
    Labels are -100 for prompt tokens (not trained on) and real ids for completion.
    """
    prompt = example["prompt"]
    completion = example["completion"]

    # Use chat template for Qwen instruct models
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize full sequence
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding=False,
        return_tensors=None,
    )

    # Find where the assistant response starts and mask prompt tokens
    prompt_only = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_only, return_tensors=None)["input_ids"]
    prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels

    return tokenized


def train(
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    augment_factor: int = 4,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[saroku trainer] Base model : {BASE_MODEL}")
    print(f"[saroku trainer] Output dir : {output_dir}")
    print(f"[saroku trainer] Device     : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # ── Load tokenizer ──────────────────────────────────────────────────────────
    print("[saroku trainer] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Generate + tokenize dataset ────────────────────────────────────────────
    print(f"[saroku trainer] Generating dataset (augment_factor={augment_factor})...")
    raw_data = generate_dataset(augment_factor=augment_factor)
    print(f"[saroku trainer] Dataset size: {len(raw_data)} examples")

    # Split 90/10 train/eval
    split = int(len(raw_data) * 0.9)
    train_data = raw_data[:split]
    eval_data  = raw_data[split:]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset  = Dataset.from_list(eval_data)

    print(f"[saroku trainer] Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: _format_for_training(x, tokenizer),
        remove_columns=train_dataset.column_names,
        num_proc=1,
    )
    eval_dataset = eval_dataset.map(
        lambda x: _format_for_training(x, tokenizer),
        remove_columns=eval_dataset.column_names,
        num_proc=1,
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    print("[saroku trainer] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    # ── Training arguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("[saroku trainer] Starting training...")
    trainer.train()

    # ── Save merged model ──────────────────────────────────────────────────────
    print("[saroku trainer] Merging LoRA weights and saving...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output_path / "model"))
    tokenizer.save_pretrained(str(output_path / "model"))

    # Save metadata
    meta = {
        "base_model": BASE_MODEL,
        "lora_rank": LORA_CONFIG.r,
        "training_examples": len(raw_data),
        "epochs": epochs,
        "saroku_version": "0.3.0",
        "model_type": "safety_classifier",
    }
    with open(output_path / "model" / "saroku_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[saroku trainer] Done. Model saved to {output_path / 'model'}")
    return str(output_path / "model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train saroku safety classifier")
    parser.add_argument("--output-dir", default="./models/saroku-safety-0.5b")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--augment", type=int, default=4,
                        help="Augmentation factor for training data")
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        augment_factor=args.augment,
    )
