'''
seal/rl_restem.py
=================
ReSTEM outer loop with combined F1 + Semantic Similarity filtering.
Logs ALL candidates per context for analytics.
'''
from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple
import os
import csv

from loguru import logger
import evaluate
from sentence_transformers import SentenceTransformer, util

# initialize the sentence‐transformer model for semantic similarity
st_model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_sim(a: str, b: str) -> float:
    emb1 = st_model.encode(a, convert_to_tensor=True)
    emb2 = st_model.encode(b, convert_to_tensor=True)
    return float(util.pytorch_cos_sim(emb1, emb2)[0, 0])

# load SQuAD F1 metric
_squad_metric = evaluate.load("squad")
def _f1(pred: str, gold: str) -> float:
    example_id = "0"
    result = _squad_metric.compute(
        predictions=[{"id": example_id, "prediction_text": pred}],
        references=[{"id": example_id, "answers": {"text": [gold], "answer_start": [0]}}],
    )
    f1 = float(result.get("f1", 0.0))
    # if gold string appears in pred but metric reports 0, boost to 100
    if f1 == 0.0 and gold.strip().lower() in pred.strip().lower():
        return 100.0
    return f1

# CSV logging of all candidates for analysis
CANDIDATE_LOG = "logs/all_candidates.csv"
CANDIDATE_FIELDS = [
    "round", "context_idx", "question", "original_answer", "candidate",
    "ground_truth", "f1", "semantic_sim", "accepted"
]

def maybe_init_candidate_log():
    if not os.path.exists(CANDIDATE_LOG):
        os.makedirs(os.path.dirname(CANDIDATE_LOG), exist_ok=True)
        with open(CANDIDATE_LOG, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
            writer.writeheader()

def restem_round(
    policy: Any,
    tokenizer: Any,
    ctx_batch: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    *,
    generate_edits_fn: Callable,
    evaluate_fn: Callable,
    fine_tune_fn: Callable,
    round_num: int = 0,
) -> Tuple[Any, float | None, List[Dict[str, Any]]]:
    if hasattr(policy, 'device'):
        policy_device = policy.device
        logger.debug(f"Policy model on device: {policy_device}")
        
        # Check if all parameters are on the same device
        device_mismatches = []
        for name, param in policy.named_parameters():
            if param.device != policy_device:
                device_mismatches.append(f"{name}: {param.device}")
        
        if device_mismatches:
            logger.warning(f"Device mismatches detected: {device_mismatches}")


    # hyperparameters from config
    k                = cfg['self_edits']['samples_per_ctx']
    rl_cfg           = cfg['rl']
    reward_threshold = rl_cfg['reward_threshold']
    sim_threshold    = rl_cfg['sim_threshold']
    f1_threshold     = rl_cfg['f1_threshold']
    alpha            = rl_cfg['alpha']

    maybe_init_candidate_log()

    # generate self‐edits
    edits = generate_edits_fn(policy, tokenizer, ctx_batch, cfg)
    expected = len(ctx_batch) * k
    if len(edits) < expected:
        logger.warning(f"⚠️ Only {len(edits)} edits generated (expected {expected}).")

    # group edits per context
    grouped: List[List[str]] = []
    for i in range(len(ctx_batch)):
        start, end = i * k, (i + 1) * k
        if end <= len(edits):
            grouped.append(edits[start:end])
    if len(grouped) < len(ctx_batch):
        skipped = len(ctx_batch) - len(grouped)
        logger.warning(f"⚠️ Skipping {skipped} contexts due to insufficient edits.")
        ctx_batch = ctx_batch[:len(grouped)]

    winners: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    # evaluate each context
    for idx, (ctx, cand_list) in enumerate(zip(ctx_batch, grouped)):
        original_answer = ctx.get('answer', '')
        ground_truth    = ctx['ground_truth']
        orig_sim        = semantic_sim(original_answer, ground_truth)
        orig_f1         = float(ctx['metric'])

        # track best candidate by combined reward
        best_reward = -float('inf')
        best_edit   = None
        best_sim    = orig_sim
        best_f1     = orig_f1

        # evaluate each candidate
        for cand in cand_list:
            sim       = semantic_sim(cand, ground_truth)
            f1_score  = _f1(cand, ground_truth)
            delta_sim = sim - orig_sim
            delta_f1  = (f1_score - orig_f1) / 100.0
            reward    = alpha * delta_sim + (1 - alpha) * delta_f1

    # ✅ Add diagnostic print here:
            logger.debug(
                f"ctx={idx} cand={cand_list.index(cand):02d} "
                f"F1={f1_score:5.3f} ΔF1={delta_f1:+5.3f} | "
                f"Sim={sim:5.3f} ΔSim={delta_sim:+5.3f} | "
                f"Reward={reward:+5.3f}"
            )

            # record for logging
            candidate_rows.append({
                "round":        round_num,
                "context_idx":  idx,
                "question":     ctx["question"],
                "original_answer": original_answer,
                "candidate":    cand,
                "ground_truth": ground_truth,
                "f1":           f1_score,
                "semantic_sim": sim,
                "accepted":     0,
            })

            # select top candidate that meets all thresholds
            if (
                reward        >= reward_threshold
                and delta_sim >= sim_threshold
                and delta_f1  >= f1_threshold
                and reward    >  best_reward
            ):
                best_reward = reward
                best_edit   = cand
                best_sim    = sim
                best_f1     = f1_score

        # mark the accepted candidate in logs
        for row in candidate_rows[-len(cand_list):]:
            if best_edit is not None and row["candidate"] == best_edit:
                row["accepted"] = 1

        # decide whether to include this edit
        if best_edit is not None and best_reward >= reward_threshold:
            winners.append({
                **ctx,
                'original_answer': original_answer,
                'improved_answer': best_edit,
                'metric':          best_f1,
                'semantic_sim':    best_sim,
            })
            logger.info(
                f"Ctx {idx}: accepted — reward={best_reward:.3f}, "
                f"Δsim={best_sim-orig_sim:.3f}, Δf1={(best_f1-orig_f1):.2f}"
            )
        else:
            logger.debug(
                f"Ctx {idx}: no candidate passed (best_reward={best_reward:.3f})"
            )

    # append all candidate stats to CSV
    with open(CANDIDATE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS)
        for row in candidate_rows:
            writer.writerow(row)

    # if no winners, skip fine-tune
    if not winners:
        logger.warning("No edits accepted → policy unchanged.")
        return policy, None, []

    # fine-tune on winners
    new_policy = fine_tune_fn(policy, tokenizer, winners, cfg)

    # re-evaluate on the dev batch
    if cfg.get('task') == 'knowledge':
        eval_out   = evaluate_fn(new_policy, tokenizer, ctx_batch)
        new_metric = float(eval_out[0]) if isinstance(eval_out, (list, tuple)) else float(eval_out)
    else:
        new_metric = float(evaluate_fn(new_policy, tokenizer, ctx_batch))

    logger.info(f"Re-eval after fine-tune → new_metric = {new_metric:.3f}")
    return new_policy, new_metric, winners
