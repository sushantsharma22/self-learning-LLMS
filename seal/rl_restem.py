"""
rl_restem.py
============
Outer loop = ReSTEM (§3.4 in the SEAL paper).

For each round:
1. Draw N contexts.
2. Sample k self-edits per context  (E-step).
3. Keep the best edit if it beats the existing metric by ≥ threshold.
4. Supervised-fine-tune (LoRA) on all accepted winners  (M-step).

This file does **not** touch HF/DeepSpeed directly; that’s handled by
the caller (`loop.py`).  It’s pure Python logic so you could unit-test it.
"""

from __future__ import annotations
from typing import List, Dict, Callable, Any, Tuple

from loguru import logger
from .reward import should_accept


def restem_round(
    policy,
    tokenizer,
    ctx_batch: List[Dict[str, Any]],
    cfg,
    *,
    generate_edits_fn: Callable,
    evaluate_fn: Callable,
    fine_tune_fn: Callable,
) -> Tuple[Any, float | None, List[Dict[str, Any]]]:
    """
    Execute **one** SEAL/ReSTEM round.

    Args
    ----
    policy             : current language-model policy (HF model)
    tokenizer          : matching tokenizer
    ctx_batch          : list of N context dicts (each already contains
                         the *current* metric in ctx["metric"])
    cfg                : Hydra config
    generate_edits_fn  : function(model, tok, batch, cfg) → flat list[str]
    evaluate_fn        : function(model, tok, batch) → metric (float or tuple)
    fine_tune_fn       : function(base_model, tok, winners, cfg) → new_model

    Returns
    -------
    new_policy         : HF model (either LoRA-updated or original)
    new_metric         : metric on the batch *after* fine-tune, or None
    winners            : list of accepted winner dicts
    """
    k = cfg.self_edits.samples_per_ctx

    # ---------- 1. sample self-edits ----------
    edits = generate_edits_fn(policy, tokenizer, ctx_batch, cfg)

    # regroup: k edits per context
    assert len(edits) == len(ctx_batch) * k, "edit count mismatch"
    grouped = [
        edits[i * k : (i + 1) * k] for i in range(len(ctx_batch))
    ]

    winners: list[dict] = []

    # ---------- 2. per-context selection ----------
    for ctx, cand_list in zip(ctx_batch, grouped):
        old_m = ctx["metric"]
        best_m = old_m
        best_edit = None

        for cand in cand_list:
            probe_ctx = ctx.copy()
            probe_ctx["improved_answer"] = cand

            # metric: F1 (knowledge) or accuracy (ARC)
            new_m = (
                evaluate_fn(policy, tokenizer, [probe_ctx])[0]
                if cfg.task == "knowledge"
                else evaluate_fn(policy, tokenizer, [probe_ctx])
            )

            if new_m > best_m:
                best_m, best_edit = new_m, cand

        # threshold check (binary reward)
        if best_edit and should_accept(old_m, best_m, cfg.rl.reward_threshold):
            winners.append({**ctx, "improved_answer": best_edit, "metric": best_m})
            logger.debug(
                f"✓ accepted edit — Δmetric = {best_m - old_m:.3f}"
            )

    # ---------- 3. fine-tune on winners ----------
    if winners:
        new_policy = fine_tune_fn(policy, tokenizer, winners, cfg)
        # recompute metric on whole batch after update
        batch_metric = (
            evaluate_fn(new_policy, tokenizer, ctx_batch)[0]
            if cfg.task == "knowledge"
            else evaluate_fn(new_policy, tokenizer, ctx_batch)
        )
        return new_policy, batch_metric, winners

    logger.warning("No edits met the reward threshold → policy unchanged.")
    return policy, None, winners
