"""
generate_edits.py
=================
E-step: sample self-edits (aka “improved answers”) from the current policy.

The paper draws *M* samples for each of *N* contexts in a batch using
temperature 1.0 + nucleus (top-p 0.95) sampling.
"""

from __future__ import annotations
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from .prompts import build_knowledge_prompt, build_arc_prompt


def sample_edits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: List[Dict[str, Any]],
    cfg,
) -> List[str]:
    """
    Return a **flat list** of length `len(batch) * k`, where
    `k = cfg.self_edits.samples_per_ctx`.

    Each input example spawns `k` improved answers; caller is responsible for
    regrouping them per-context if needed.
    """
    k = cfg.self_edits.samples_per_ctx
    prompts: list[str] = []

    # Build the prompt for every (context, sample) pair
    for ex in batch:
        if cfg.task == "knowledge":
            base_prompt = build_knowledge_prompt(
                question=ex["question"],
                answer=ex["answer"],
                ground_truth=ex["ground_truth"],
            )
        else:  # ARC
            base_prompt = build_arc_prompt(
                question=ex["question"],
                options=ex["options"],
                answer=ex["answer"],
                explanation=ex.get("explanation", ""),
            )

        prompts.extend([base_prompt] * k)  # duplicate for k samples

    # Tokenise en-masse (padding for efficiency)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    logger.info(f"Sampling {len(prompts)} self-edits "
                f"(k={k} × {len(batch)} contexts)…")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=128,
        )

    edits = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return edits
