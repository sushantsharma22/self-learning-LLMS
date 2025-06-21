"""
evaluate.py
===========
Task-level metrics used as the SEAL reward signal.

* Knowledge task  → SQuAD F1 + EM   (we track F1 for reward)
* ARC task        → multiple-choice accuracy
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import torch
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


# ---------- load HF metrics once ----------
_squad = load("squad")
_accuracy = load("accuracy")


# ---------- knowledge evaluation ----------
def eval_knowledge(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: List[Dict],
) -> Tuple[float, float]:
    """
    Compute (F1, EM) on a SQuAD-style `batch`.
    Each item must have:
        question, ground_truth
    """
    preds, refs = [], []

    for i, ex in enumerate(batch):
        # Zero-shot generate an answer
        prompt = f"{ex['question']}\nAnswer:"
        inps = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inps, max_new_tokens=32)

        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        ans = ans.split("Answer:")[-1].strip()  # keep text after the cue

        preds.append({"id": str(i), "prediction_text": ans})
        refs.append(
            {
                "id": str(i),
                "answers": {
                    "text": [ex["ground_truth"]],
                    "answer_start": [0],
                },
            }
        )

    scores = _squad.compute(predictions=preds, references=refs)
    logger.debug(scores)
    return scores["f1"], scores["exact_match"]


# ---------- ARC evaluation ----------
def eval_arc(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: List[Dict],
) -> float:
    """
    Compute accuracy on ARC multiple-choice.
    Each item must have:
        question, options (multi-line A–E), label (gold letter)
    """
    preds, labels = [], []

    for ex in batch:
        prompt = (
            f"{ex['question']}\nOPTIONS:\n{ex['options']}\n"
            "Answer (A/B/C/D/E):"
        )
        inps = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inps, max_new_tokens=4)

        # last non-space char is usually the letter
        letter = tokenizer.decode(out[0], skip_special_tokens=True).strip()[-1].upper()
        preds.append(letter)
        labels.append(ex["label"])

    acc = _accuracy.compute(predictions=preds, references=labels)["accuracy"]
    logger.debug({"accuracy": acc})
    return acc
