"""
sft_lora.py
===========
M-step: supervised fine-tune on the **accepted** self-edits using LoRA adapters.

Lightweight PEFT keeps the memory footprint small enough for 4×15 GB A16s.
"""

from __future__ import annotations

from typing import List, Dict, Any

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger


def fine_tune_lora(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    winners: List[Dict[str, Any]],
    cfg,
):
    """
    Fine-tune the *base_model* on the list of winner contexts, each containing
    an `improved_answer`.  Returns the **LoRA-adapted** model.

    The full LoRA config is pulled from `cfg.lora`.
    """
    if not winners:
        logger.warning("No winners to fine-tune on → skipping SFT.")
        return base_model

    # Build a tiny HF Dataset on-the-fly
    def format_example(ex):
        if cfg.task == "knowledge":
            text = f"{ex['question']}\n{ex['improved_answer']}"
        else:  # ARC
            text = f"{ex['question']}\nAnswer: {ex['improved_answer']}"
        return tokenizer(text, truncation=True)

    ds = Dataset.from_list(winners).map(
        format_example,
        remove_columns=winners[0].keys(),
        batched=False,
    )

    # Configure LoRA
    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
    )
    model = get_peft_model(base_model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="ft_out",
        per_device_train_batch_size=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        report_to=[],  # silence WandB/etc.
    )

    logger.info(f"Starting LoRA fine-tune on {len(winners)} winners…")
    Trainer(model=model, args=args, train_dataset=ds).train()

    return model
