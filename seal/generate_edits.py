# FILE: seal/generate_edits.py

from __future__ import annotations
import gc, torch
from typing import Any, Dict, List
from loguru import logger
from transformers import GenerationConfig

def clean_llama_answer(txt: str) -> str:
    """Clean up generated text from Llama models"""
    lines = [l for l in txt.splitlines() if l.strip() and not l.strip().startswith("<|")]
    return (lines[0] if lines else txt).lstrip("> ").lstrip("<|assistant|>").strip().rstrip(".")

def sample_edits(
    model: Any,
    tokenizer: Any,
    batch: List[Dict[str, Any]], 
    cfg: Dict[str, Any]
) -> List[str]:
    """
    Generate self-edits for the given batch of contexts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        batch: List of contexts with 'question' keys
        cfg: Configuration dictionary
        
    Returns:
        List of generated candidate answers
    """
    device = model.device
    k = cfg['self_edits']['samples_per_ctx']
    
    logger.info(f"Sampling {k * len(batch)} self-edits (k={k} × {len(batch)} contexts)…")
    
    # Prepare prompts for generation
    prompts = []
    for ex in batch:
        base_prompt = f"<|user|>\n{ex['question']}\n<|assistant|>\n"
        prompts.extend([base_prompt] * k)
    
    # Generate in batches to manage memory
    batch_size = 32  # Adjust based on your GPU memory
    all_edits = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize inputs
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        # Generation configuration
        gen_cfg = GenerationConfig(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=32,
            pad_token_id=tokenizer.pad_token_id
        )
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_cfg)
        
        # Extract generated text (remove input prompt)
        generated = outputs[:, inputs["input_ids"].shape[-1]:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # Clean up generated text
        cleaned = [clean_llama_answer(text) for text in decoded]
        all_edits.extend(cleaned)
        
        # Clear memory
        del inputs, outputs, generated
        torch.cuda.empty_cache()
    
    logger.info(f"Sampled {len(all_edits)} candidate answers for {len(batch)} contexts.")
    
    return all_edits

def sample_edits_ddp(
    model: Any,
    tokenizer: Any,
    batch: List[Dict[str, Any]], 
    cfg: Dict[str, Any]
) -> List[str]:
    """
    Distributed version of sample_edits for multi-GPU training
    """
    import os
    import torch.distributed as dist
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    device = f"cuda:{local_rank}"
    k = cfg['self_edits']['samples_per_ctx']
    
    # Distribute contexts across ranks
    contexts_per_rank = len(batch) // world_size
    start_idx = local_rank * contexts_per_rank
    end_idx = start_idx + contexts_per_rank if local_rank < world_size - 1 else len(batch)
    
    local_batch = batch[start_idx:end_idx]
    
    logger.info(f"[Rank {local_rank}] Sampling {k * len(local_batch)} self-edits...")
    
    # Generate prompts for local batch
    prompts = []
    for ex in local_batch:
        base_prompt = f"<|user|>\n{ex['question']}\n<|assistant|>\n"
        prompts.extend([base_prompt] * k)
    
    # Generate on local GPU
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)
    
    gen_cfg = GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=32,
        pad_token_id=tokenizer.pad_token_id
    )
    
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_cfg)
    
    generated = outputs[:, inputs["input_ids"].shape[-1]:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    local_edits = [clean_llama_answer(text) for text in decoded]
    
    # Gather results from all ranks
    if dist.is_initialized():
        all_edits = [None] * world_size
        dist.all_gather_object(all_edits, local_edits)
        # Flatten results
        return [edit for rank_edits in all_edits for edit in rank_edits]
    else:
        return local_edits
