# training_utils/log_helpers.py
from __future__ import annotations

import logging
from typing import Sequence


logger = logging.getLogger(__name__)


def _avg(values: Sequence[float]) -> float:
    """Safe average (returns 0.0 if empty)."""
    return sum(values) / max(len(values), 1)


def log_epoch_metrics(
    epoch: int,
    num_epochs: int,
    *,
    train_ctc: float,
    eval_ctc_clean: float,
    eval_ctc_perturbed: float,
    train_wer: float,
    eval_wer_clean: float,
    eval_wer_perturbed: float,
) -> None:
    """
    Log metrics for a single epoch in a structured table.
    """
    lines = [
        "=" * 70,
        f"Epoch {epoch}/{num_epochs} summary:",
        f"{'Metric':<10} | {'Train':>10} | {'Eval Clean':>12} | {'Eval Perturbed':>16}",
        "-" * 70,
        f"{'CTC':<10} | {train_ctc:>10.0f} | {eval_ctc_clean:>12.0f} | {eval_ctc_perturbed:>16.0f}",
        f"{'WER':<10} | {train_wer:>10.2f} | {eval_wer_clean:>12.2f} | {eval_wer_perturbed:>16.2f}",
        "=" * 70,
    ]
    for line in lines:
        logger.info(line)


def log_summary_metrics(
    *,
    args,
    clean_ctc_test: float,
    clean_wer_test: float,
    pert_ctc_test: float,
    pert_wer_test: float,
    best_epoch: int,
) -> None:
    """
    Log final summary after training is complete.
    """
    lines = [
        "=" * 70,
        "Summary",
        "=" * 70,
        f"{'Perturbation norm type:':<30} {args.norm_type}",
        f"{'Perturbation size:':<30} {args.attack_size_string}",
        "-" * 70,
        f"{'Metric':<20} | {'Clean Test':>15} | {'Perturbed Test':>15}",
        "-" * 70,
        f"Best epoch: {best_epoch}",
        f"{'CTC':<20} | {clean_ctc_test:>15.2f} | {pert_ctc_test:>15.2f}",
        f"{'WER':<20} | {clean_wer_test:>15.3f} | {pert_wer_test:>15.3f}",
        "=" * 70,
    ]
    for line in lines:
        logger.info(line)


def log_train_progress(
    batch_idx: int,
    total_batches: int,
    ctc_scores: Sequence[float],
    wer_scores: Sequence[float],
    times: Sequence[float],
) -> None:
    """
    Log progress during training at intermediate checkpoints.
    """
    msg = (
        f"Batch {batch_idx}/{total_batches} | "
        f"avg CTC: {_avg(ctc_scores):.0f} | "
        f"avg WER: {_avg(wer_scores):.3f} | "
        f"avg time: {_avg(times):.2f}s"
    )
    logger.info(msg)
