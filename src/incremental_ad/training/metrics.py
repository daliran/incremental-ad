import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    denom = precision[:-1] + recall[:-1]
    f1s = np.zeros_like(denom)
    np.divide(2 * precision[:-1] * recall[:-1], denom, out=f1s, where=denom > 0)
    return float(thresholds[np.argmax(f1s)])


def eval_classification(
    scores: np.ndarray, labels: np.ndarray, threshold: float | None
) -> dict:
    if threshold is None:
        threshold = find_best_f1_threshold(scores, labels)

    preds = (scores >= threshold).astype(int)

    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def find_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """Returns (start, end) index pairs for each contiguous run of 1s. End is exclusive."""
    padded = np.concatenate([[0], labels, [0]])
    transitions = np.diff(padded)
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _compute_segment_hits(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[np.ndarray, list[tuple[int, int]], list[bool]]:
    """Threshold scores, find GT segments, and return which segments are hit by a prediction."""
    preds = (scores >= threshold).astype(int)
    gt_segments = find_segments(labels)
    hits = [bool(preds[s:e].any()) for s, e in gt_segments]
    return preds, gt_segments, hits


def eval_point_adjusted(
    scores: np.ndarray, labels: np.ndarray, threshold: float | None
) -> dict:
    if threshold is not None:
        # Fixed threshold: apply PA directly without sweeping.
        preds, gt_segments, hits = _compute_segment_hits(scores, labels, threshold)

        adjusted = preds.copy()

        for (s, e), hit in zip(gt_segments, hits):
            if hit:
                adjusted[s:e] = 1

        return {
            "f1": float(f1_score(labels, adjusted, zero_division=0)),
            "precision": float(precision_score(labels, adjusted, zero_division=0)),
            "recall": float(recall_score(labels, adjusted, zero_division=0)),
        }

    # Oracle: sweep all exact threshold candidates, applying PA at each step.
    _, _, all_thresholds = precision_recall_curve(labels, scores)

    # Downsample to at most 500 candidates: the PA-F1 curve is smooth so
    # evenly-spaced sampling loses negligible accuracy while avoiding O(N²).
    n_candidates = 500

    if len(all_thresholds) > n_candidates:
        idx = np.linspace(0, len(all_thresholds) - 1, n_candidates, dtype=int)
        all_thresholds = all_thresholds[idx]

    # GT segments don't change across threshold candidates — compute once.
    gt_segments = find_segments(labels)

    best: dict = {"f1": -1.0, "precision": 0.0, "recall": 0.0}

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


def eval_event(scores: np.ndarray, labels: np.ndarray, threshold: float | None) -> dict:
    if threshold is None:
        threshold = find_best_f1_threshold(scores, labels)

    preds, gt_segments, hits = _compute_segment_hits(scores, labels, threshold)

    if len(gt_segments) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    tp = sum(hits)
    fn = len(gt_segments) - tp

    # FP: contiguous detected regions that don't overlap any anomaly segment.
    pred_segments = find_segments(preds)

    fp = sum(
        1
        for ps, pe in pred_segments
        if not any(s < pe and ps < e for s, e in gt_segments)
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "n_segments": len(gt_segments),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
