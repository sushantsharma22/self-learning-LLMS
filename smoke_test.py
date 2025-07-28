#!/usr/bin/env python3
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# import the two core bits of SEAL we want to smoke-test
from seal.evaluate import eval_knowledge
from seal.generate_edits import sample_edits

def main():
    model_name = "Qwen/Qwen1.5-7B"
    print(f"→ Loading {model_name} with device_map='auto' across GPUs")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",          # auto-shard across all available GPUs
        low_cpu_mem_usage=True,     # reduce CPU RAM spikes
        torch_dtype=torch.float16,  # use fp16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # grab one SQuAD QA
    with open("data/squad/train-v1.1.json") as f:
        squad = json.load(f)
    qa = squad["data"][0]["paragraphs"][0]["qas"][0]
    example = {
        "question":       qa["question"],
        "ground_truth":   qa["answers"][0]["text"],
        "answer":         "",
        "metric":         0.0,
    }

    # run eval_knowledge
    print("→ Running eval_knowledge …")
    f1, em = eval_knowledge(model, tokenizer, [example])
    print(f"   ↳ F1 = {f1:.2f}, EM = {em:.2f}")

    # sample_edits with k=1
    class C: pass
    cfg = C()
    cfg.task = "knowledge"
    cfg.self_edits = C(); cfg.self_edits.samples_per_ctx = 1
    cfg.dataset   = C(); cfg.dataset.ctx_per_round    = 1

    print("→ Running sample_edits (k=1) …")
    edits = sample_edits(model, tokenizer, [example], cfg)
    print(f"   ↳ Got {len(edits)} edit(s):", edits)

if __name__ == "__main__":
    main()
