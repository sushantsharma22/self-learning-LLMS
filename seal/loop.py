"""
loop.py
=======
Top-level script that ties everything together:

* parses Hydra config (`conf/*.yaml`)
* loads / quantises the base model
* slices the dataset
* runs `rl_restem.restem_round` for R rounds
* prints metric deltas after each round

Launch via:

    accelerate launch --deepspeed ds_z3_A16.json -m seal.loop task=knowledge
"""

from __future__ import annotations
import random
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --- local imports ---
from .generate_edits import sample_edits
from .sft_lora import fine_tune_lora
from .evaluate import eval_knowledge, eval_arc
from .rl_restem import restem_round

# Hydra looks for configs relative to this dir → ../../conf
CONF_DIR = str(Path(__file__).resolve().parent.parent / "conf")


@hydra.main(config_path=CONF_DIR, config_name="knowledge", version_base=None)
def main(cfg: DictConfig):
    logger.info("Loaded config:\n" + OmegaConf.to_yaml(cfg))

    # --------------------------------------------------
    # 1. Load / quantise model
    # --------------------------------------------------
    if cfg.model.load_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        policy = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            device_map="auto",
            quantization_config=bnb_cfg,
        )
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            device_map="auto",
            torch_dtype="auto",
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # 2. Prepare dataset slice
    # --------------------------------------------------
    if cfg.task == "knowledge":
        with open("data/squad/train-v1.1.json") as f:
            train_json = json.load(f)
        with open("data/squad/dev-v1.1.json") as f:
            val_json = json.load(f)

        def extract_qas(data):
            return [
                qa
                for article in data["data"]
                for para in article["paragraphs"]
                for qa in para["qas"]
                if qa["answers"]
            ]

        train_raw = extract_qas(train_json)
        eval_raw = extract_qas(val_json)

        print("✅ SQuAD loaded with", len(train_raw), "train examples")

        train = train_raw[:int(len(train_raw) * 0.01)] \
            if str(cfg.dataset.train_slice).endswith("%") else train_raw
        eval_split = eval_raw[:200]

        train_examples = [
            {
                "question": qa["question"],
                "ground_truth": qa["answers"][0]["text"],
                "answer": "",
                "metric": 0.0,
            }
            for qa in train
        ]

        eval_fn = eval_knowledge

    else:  # ARC
        from datasets import load_dataset  # only import if needed
        raw = load_dataset(cfg.dataset.name, cfg.dataset.subset)
        train = raw["train"]

        train_examples = [
            {
                "question": q,
                "options": "\n".join(opts),
                "label": correct,
                "answer": correct,
                "metric": 0.0,
            }
            for q, opts, correct in zip(
                train["question"],
                train["choices"]["text"],
                train["answerKey"],
            )
        ]
        eval_fn = eval_arc

    # --------------------------------------------------
    # 3. Baseline metric
    # --------------------------------------------------
    baseline_metric = (
        eval_fn(policy, tokenizer, train_examples[:10])[0]
        if cfg.task == "knowledge"
        else eval_fn(policy, tokenizer, train_examples[:10])
    )
    logger.success(f"Baseline metric = {baseline_metric:.3f}")
    current_metric = baseline_metric

    # --------------------------------------------------
    # 4. SEAL / ReSTEM rounds
    # --------------------------------------------------
    for r in range(cfg.rl.max_rounds):
        logger.info(f"========== SEAL Round {r + 1}/{cfg.rl.max_rounds} ==========")

        batch = random.sample(train_examples, cfg.dataset.ctx_per_round)
        for ctx in batch:
            ctx["metric"] = current_metric

        policy, new_metric, winners = restem_round(
            policy,
            tokenizer,
            batch,
            cfg,
            generate_edits_fn=sample_edits,
            evaluate_fn=eval_fn,
            fine_tune_fn=fine_tune_lora,
        )

        if new_metric is not None:
            logger.success(f"✓ metric improved → {new_metric:.3f}")
            current_metric = new_metric
        else:
            logger.warning("✗ no improvement this round")

    logger.info("SEAL loop completed.")


if __name__ == "__main__":
    main()
