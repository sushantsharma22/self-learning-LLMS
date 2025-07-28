'''
seal/evaluate.py
================
Upgraded evaluation utilities for LoRA-adapted QA models: SQuAD F1/EM and semantic similarity.
Runs on the full dev set efficiently, with batch size control and full logging.
'''

from __future__ import annotations
from typing import List, Dict, Tuple
import argparse
import torch
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import logging
import numpy as np
import random
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

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
    batch_size: int = 8,
    max_new_tokens: int = 32
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

    for i in range(0, len(batch), batch_size):
        sub_batch = batch[i:i+batch_size]
        prompts = [f"Question: {ex['question']}\nAnswer (respond concisely):" for ex in sub_batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False
            )
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for j, (ex, d) in enumerate(zip(sub_batch, decoded)):
            if "Answer:" in d:
                answer_text = d.split("Answer:")[-1].strip()
            else:
                answer_text = d[len(prompts[j]):].strip()
            preds.append({"id": str(i+j), "prediction_text": answer_text})
            refs.append({
                "id": str(i+j),
                "answers": {"text": [ex["ground_truth"]], "answer_start": [0]},
            })
            sem_sim = semantic_similarity(answer_text, ex["ground_truth"])
            semantic_scores.append(sem_sim)
            logger.debug(f"Eval_knowledge | Ex {i+j}: Answer='{answer_text}' vs Truth='{ex['ground_truth']}' | Sim={sem_sim:.3f}")

    results = _squad.compute(predictions=preds, references=refs)
    f1 = float(results.get("f1", 0.0))
    em = float(results.get("exact", results.get("exact_match", 0.0)))
    avg_sem_sim = float(sum(semantic_scores)) / len(semantic_scores) if semantic_scores else 0.0
    logger.info(f"Eval_knowledge | F1: {f1:.3f}, EM: {em:.3f}, SemSim: {avg_sem_sim:.3f}")
    return f1, em, avg_sem_sim, semantic_scores

def main():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA-PEFT QA model on SQuAD dev set (full set, efficient batch eval).")
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to LoRA adapter checkpoint dir (e.g. ckpt_run1/policy)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--max_new_tokens', type=int, default=32, help='Max tokens to generate per answer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit of eval samples (debugging)')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LoRA adapter from: {args.model_ckpt}")

    # Load PEFT config to get base model name
    peft_config = PeftConfig.from_pretrained(args.model_ckpt)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model: {base_model_name}")

    # Load base model & tokenizer from Hugging Face Hub
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(model, args.model_ckpt)
    model = model.merge_and_unload()  # For pure inference, optional
#    model.to(device)

    print("Loading SQuAD validation set...")
    ds = load_dataset("squad")
    eval_examples = ds["validation"]
    if args.limit:
        eval_examples = eval_examples.select(range(args.limit))

    # Prepare SQuAD-style dict format expected by eval_knowledge
    qa_batch = [{"question": q, "ground_truth": a["text"][0]} for q, a in zip(eval_examples["question"], eval_examples["answers"])]

    print(f"Evaluating {len(qa_batch)} examples, batch size {args.batch_size}...")
    f1, em, semsim, sem_scores = eval_knowledge(model, tokenizer, qa_batch, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    print("=" * 60)
    print(f"Full SQuAD dev set evaluation:")
    print(f"F1: {f1:.2f}")
    print(f"EM: {em:.2f}")
    print(f"Average Semantic Similarity: {semsim:.3f}")
    print(f"Total evaluated: {len(qa_batch)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
