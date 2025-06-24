from __future__ import annotations

from typing import List, Dict, Any  # <-- this line is required

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from loguru import logger

def fine_tune_lora(
    base_model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    winners: List[Dict[str, Any]],
    cfg,
):

    if not winners:
        logger.warning("No winners to fine-tune on → skipping SFT.")
        return base_model

    # 👁️ Visual log of accepted edits
    print("\n📘 Accepted Self-Edits:\n")
    for i, ex in enumerate(winners):
        print(f"[{i+1}] Question: {ex['question']}")
        print(f"↪ Original: {ex.get('original_answer', 'N/A')}")
        print(f"✅ Improved: {ex['improved_answer']}")
        print("-" * 60)

    def format_example(ex):
        if cfg.task == "knowledge":
            text = f"{ex['question']}\n{ex['improved_answer']}"
        else:
            text = f"{ex['question']}\nAnswer: {ex['improved_answer']}"
        encoded = tokenizer(text, truncation=True, padding="max_length", max_length=512)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    ds = Dataset.from_list(winners).map(
        format_example,
        remove_columns=winners[0].keys(),
        batched=False,
    )

    # Convert ListConfig to plain list to avoid JSON serialization issues
    target_modules = list(cfg.lora.target_modules)

    lcfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=target_modules,
    )


    model = get_peft_model(base_model, lcfg)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="ft_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=1,
        logging_steps=10,
        report_to=[],
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    logger.info(f"Starting LoRA fine-tune on {len(winners)} winners…")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
    )
    trainer.train()

    return model
