"""Anomaly detection metrics operating on per-window scores and binary labels."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_segments(labels: np.ndarray) -> list[tuple[np.intp, np.intp]]:
    """Return (start, end) indices of contiguous runs of 1s. End is exclusive."""
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1s = np.zeros_like(denom)
    np.divide(2 * precision[:-1] * recall[:-1], denom, out=f1s, where=denom > 0)
    return float(thresholds[np.argmax(f1s)])


def eval_classification(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
) -> dict[str, float]:
    """AUROC, AUPRC, F1/precision/recall. threshold=None uses oracle best-F1."""
    if threshold is None:
        threshold = best_f1_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def eval_point_adjusted(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
    n_candidates: int = 500,
) -> dict[str, float]:
    """Point-adjusted F1: if any point in a GT segment is detected, the whole segment counts as detected."""
    gt_segments = find_segments(labels)

    if threshold is not None:
        preds = (scores >= threshold).astype(int)
        adjusted = preds.copy()
        for s, e in gt_segments:
            if preds[s:e].any():
                adjusted[s:e] = 1
        return {
            "f1": float(f1_score(labels, adjusted, zero_division=0)),
            "precision": float(precision_score(labels, adjusted, zero_division=0)),
            "recall": float(recall_score(labels, adjusted, zero_division=0)),
        }

    # Oracle: sweep candidates from PR curve
    _, _, all_thresholds = precision_recall_curve(labels, scores)
    if len(all_thresholds) > n_candidates:
        idx = np.linspace(0, len(all_thresholds) - 1, n_candidates).astype(int)
        all_thresholds = all_thresholds[idx]

    best: dict[str, float] = {"f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in all_thresholds:
        preds = (scores >= t).astype(int)
        adjusted = preds.copy()
        for s, e in gt_segments:
            if preds[s:e].any():
                adjusted[s:e] = 1
        f = float(f1_score(labels, adjusted, zero_division=0))
        if f > best["f1"]:
            best = {
                "f1": f,
                "precision": float(precision_score(labels, adjusted, zero_division=0)),
                "recall": float(recall_score(labels, adjusted, zero_division=0)),
            }
    return best


def eval_point(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
) -> dict[str, float]:
    """Point-level metrics (last-timestep-per-window convention), no PA."""
    if threshold is None:
        threshold = best_f1_threshold(scores, labels)
    preds = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def eval_event(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float | None,
) -> dict[str, float]:
    """Segment-level detection: a GT segment is a hit if any predicted point falls inside it."""
    if threshold is None:
        threshold = best_f1_threshold(scores, labels)

    preds = (scores >= threshold).astype(int)
    gt_segments = find_segments(labels)

    if not gt_segments:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}

    tp = sum(1 for s, e in gt_segments if preds[s:e].any())
    fn = len(gt_segments) - tp

    pred_segments = find_segments(preds)
    fp = sum(
        1
        for ps, pe in pred_segments
        if not any(s < pe and ps < e for s, e in gt_segments)
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
