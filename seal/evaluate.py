'''  
seal/evaluate.py
================
Evaluation utilities for QA and MC tasks: SQuAD F1/EM and semantic similarity.
'''  
from __future__ import annotations
from typing import List, Dict, Tuple

import torch
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# Load metrics and semantic model only once
_squad = load("squad")  # returns dict with 'exact' and 'f1'
_accuracy = load("accuracy")
_sbert = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(ans1: str, ans2: str) -> float:
    """
    Return cosine similarity between two answers (0..1, higher=more similar)
    """
    emb1 = _sbert.encode(ans1, convert_to_tensor=True)
    emb2 = _sbert.encode(ans2, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2).item())

def eval_knowledge(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: List[Dict[str, str]],
) -> Tuple[float, float, float, List[float]]:
    """
    Compute (F1, EM, avg_semantic_similarity, list_of_semantic_scores) on a SQuAD-style batch.
    Each item must have: 'question' and 'ground_truth'.
    Returns floats + per-example semantic similarity.
    """
    device = model.device
    preds: List[Dict[str, str]] = []
    refs: List[Dict[str, Dict[str, List[str]]]] = []
    semantic_scores: List[float] = []

    for i, ex in enumerate(batch):
        prompt = f"Question: {ex['question']}\nAnswer (respond concisely):"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=32,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        if "Answer:" in decoded:
            answer_text = decoded.split("Answer:")[-1].strip()
        else:
            answer_text = decoded[len(prompt):].strip()

        preds.append({"id": str(i), "prediction_text": answer_text})
        refs.append({
            "id": str(i),
            "answers": {"text": [ex["ground_truth"]], "answer_start": [0]},
        })
        sem_sim = semantic_similarity(answer_text, ex["ground_truth"])
        semantic_scores.append(sem_sim)
        logger.debug(f"Eval_knowledge | Ex {i}: Answer='{answer_text}' vs Truth='{ex['ground_truth']}' | Sim={sem_sim:.3f}")

    results = _squad.compute(predictions=preds, references=refs)
    f1 = float(results.get("f1", 0.0))
    em = float(results.get("exact", results.get("exact_match", 0.0)))
    avg_sem_sim = float(sum(semantic_scores)) / len(semantic_scores) if semantic_scores else 0.0
    logger.info(f"Eval_knowledge | F1: {f1:.3f}, EM: {em:.3f}, SemSim: {avg_sem_sim:.3f}")
    return f1, em, avg_sem_sim, semantic_scores

def eval_arc(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch: List[Dict[str, str]],
) -> float:
    """
    Compute accuracy on ARC multiple-choice.
    Each item must have: 'question', 'options', 'label'.
    Returns float.
    """
    device = model.device
    preds: List[str] = []
    labels: List[str] = []

    for ex in batch:
        prompt = (
            f"{ex['question']}\nOPTIONS:\n{ex['options']}\nAnswer (A/B/C/D/E):"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=4,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # last character should be the letter
        letter = decoded[-1].upper() if decoded else ''
        preds.append(letter)
        labels.append(ex['label'])
        logger.debug(f"Eval_arc | Prompt='{prompt}' → Pred='{letter}', True='{ex['label']}'")

    acc = _accuracy.compute(predictions=preds, references=labels).get('accuracy', 0.0)
    acc_f = float(acc)
    logger.info(f"Eval_arc | Accuracy: {acc_f:.3f}")
    return acc_f
