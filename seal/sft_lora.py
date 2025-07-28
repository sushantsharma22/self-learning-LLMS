# FILE: seal/sft_lora.py

from __future__ import annotations
import gc, os, torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class _Args:
    output_dir: str = "tmp_lora"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-5
    logging_steps: int = 1
    save_strategy: str = "no"
    report_to: list = None
    remove_unused_columns: bool = False
    bf16: bool = False
    fp16: bool = False
    optim: str = "paged_adamw_8bit"
    dataloader_pin_memory: bool = False
    dataloader_num_workers: int = 0

def _format_prompt(ex: Dict[str, Any]) -> str:
    return f"<|user|>\n{ex['question']}\n<|assistant|>\n{ex['improved_answer']}"

def _tokenise(batch: Dict[str, List[str]], tokenizer):
    tok = tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    tok["labels"] = tok["input_ids"].clone()
    return tok

def fine_tune_lora(model, tokenizer, winners: List[Dict[str, Any]], cfg):
    """Original single-GPU LoRA fine-tuning function"""
    device = model.device
    logger.info(f"LoRA fine-tuning will operate on device: {device}")
    logger.info(f"Starting LoRA fine-tune on {len(winners)} winners…")

    # Ensure all model parameters are on the same device
    model_device = str(device)
    for name, param in model.named_parameters():
        if str(param.device) != model_device:
            logger.warning(f"Moving parameter {name} from {param.device} to {device}")
            param.data = param.data.to(device)
    
    for name, buffer in model.named_buffers():
        if str(buffer.device) != model_device:
            logger.warning(f"Moving buffer {name} from {buffer.device} to {device}")
            buffer.data = buffer.data.to(device)

    for ex in winners:
        logger.info(f"[✓ WIN] Q: {ex['question']}\n↪ Original: {ex['original_answer']}\n✅ Improved: {ex['improved_answer']}\n" + "-" * 60)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ds = Dataset.from_list(winners).map(
        lambda e: {"prompt": _format_prompt(e)}, 
        remove_columns=list(winners[0].keys())
    ).map(
        lambda b: _tokenise(b, tokenizer), 
        batched=True, 
        remove_columns=["prompt"]
    )

    if cfg["model"].get("load_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # Fix API compatibility issue
    try:
        model.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        model.gradient_checkpointing_enable()
    
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    args_dict = {**_Args().__dict__}
    args_dict["learning_rate"] = cfg["lora"].get("learning_rate", args_dict["learning_rate"])
    args_dict["num_train_epochs"] = cfg["lora"].get("num_train_epochs", args_dict["num_train_epochs"])

    if torch.cuda.is_bf16_supported():
        args_dict["bf16"] = True
    else:
        args_dict["fp16"] = True

    training_args = TrainingArguments(**args_dict, local_rank=-1)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    if hasattr(trainer.model, 'device'):
        logger.info(f"Trainer model device: {trainer.model.device}")

    trainer.train()

    logger.info("LoRA fine-tuning complete. Cleaning up trainer objects.")
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return model

def fine_tune_lora_ddp(model, tokenizer, winners: List[Dict[str, Any]], cfg):
    """Multi-GPU LoRA fine-tuning with proper DDP synchronization"""
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    device = torch.device(f'cuda:{local_rank}')
    
    # Ensure model is on correct device
    model = model.to(device)
    
    if local_rank == 0:
        logger.info(f"[Rank {local_rank}/{world_size}] LoRA fine-tuning on device: {device}")
        logger.info(f"[Rank {local_rank}] Fine-tuning {len(winners)} winners...")
        
        # Log winners only on rank 0
        for ex in winners:
            logger.info(f"[✓ WIN] Q: {ex['question']}\n↪ Original: {ex['original_answer']}\n✅ Improved: {ex['improved_answer']}\n" + "-" * 60)

    # CRITICAL: Ensure tokenizer setup is identical across all ranks
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # CRITICAL: Prepare dataset on all ranks (not just rank 0)
    ds = Dataset.from_list(winners).map(
        lambda e: {"prompt": _format_prompt(e)}, 
        remove_columns=list(winners[0].keys())
    ).map(
        lambda b: _tokenise(b, tokenizer), 
        batched=True, 
        remove_columns=["prompt"]
    )

    # CRITICAL: Prepare model for LoRA on ALL ranks
    if cfg["model"].get("load_4bit", False):
        model = prepare_model_for_kbit_training(model)

    # Fix API compatibility on all ranks
    try:
        model.gradient_checkpointing_enable(use_reentrant=False)
    except TypeError:
        model.gradient_checkpointing_enable()
    
    model.config.use_cache = False

    # CRITICAL: Apply LoRA configuration identically on ALL ranks
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    
    # Print trainable parameters only on rank 0
    if local_rank == 0:
        model.print_trainable_parameters()

    # CRITICAL: Synchronize all ranks before DDP wrapping
    if dist.is_initialized():
        dist.barrier()

    # Wrap with DDP AFTER all ranks have identical models
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Training arguments
    args_dict = {**_Args().__dict__}
    args_dict["learning_rate"] = cfg["lora"].get("learning_rate", args_dict["learning_rate"])
    args_dict["num_train_epochs"] = cfg["lora"].get("num_train_epochs", args_dict["num_train_epochs"])
    
    # Optimize for DDP
    args_dict["per_device_train_batch_size"] = 1  # Keep conservative
    args_dict["gradient_accumulation_steps"] = 2  # Increase for effective batch size
    
    if torch.cuda.is_bf16_supported():
        args_dict["bf16"] = True
    else:
        args_dict["fp16"] = True

    training_args = TrainingArguments(
        **args_dict, 
        local_rank=local_rank,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=ddp_model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        data_collator=data_collator
    )

    if local_rank == 0:
        logger.info(f"[Rank {local_rank}] Starting DDP training...")
    
    trainer.train()

    # Extract base model from DDP wrapper
    if hasattr(trainer.model, 'module'):
        trained_model = trainer.model.module
    else:
        trained_model = trainer.model

    if local_rank == 0:
        logger.info(f"[Rank {local_rank}] LoRA fine-tuning complete.")
    
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return trained_model
