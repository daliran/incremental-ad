"""Split a labelled AD test recording in time into a calibration part and an evaluation part.

Used by `remerge.py --calibration_split` (§1.42): α is picked on the calibration part (the first
c of the recording) and judged on the evaluation part (the rest). Pure functions over the
evaluator's own per-window scores, so the numbers are the published window AUROC restricted to a
subset — never a re-implementation of it.

Rules, each asserted rather than assumed:

- **The cut never falls inside an anomaly segment.** The requested cut ``round(c * n_points)`` is
  moved to the nearer boundary of the segment it lands in (ties to the segment's start, which
  keeps the event in the evaluation part), and both positions are reported.
- **A window belongs to a part only if it lies wholly inside it.** Windows straddling the cut are
  dropped and counted, so no calibration window sees evaluation points or vice versa.
- **Every event lies in exactly one part.** A consequence of the first rule, checked anyway.
- **Window labels are re-derived from the dataset and must equal the evaluator's.** That binds
  the window order the scores arrive in to the window positions the split assumes; if the test
  loader were ever shuffled or strided differently, this fails instead of mis-assigning windows.
"""

import numpy as np

from incremental_ad.framework.evaluators._metrics import eval_classification


def anomaly_segments(point_labels: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of label 1."""
    labels = np.asarray(point_labels).astype(np.int8).ravel()
    edges = np.diff(np.concatenate(([0], labels, [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def place_cut(n_points: int, fraction: float,
              segments: list[tuple[int, int]]) -> tuple[int, int]:
    """(requested, actual) cut index. `actual` is never strictly inside a segment."""
    requested = int(round(fraction * n_points))
    for start, end in segments:
        if start < requested < end:
            actual = start if (requested - start) <= (end - requested) else end
            return requested, actual
    return requested, requested


def window_starts(n_windows: int, stride: int) -> np.ndarray:
    return np.arange(n_windows, dtype=np.int64) * int(stride)


def split(scores: np.ndarray, window_labels: np.ndarray, point_labels: np.ndarray,
          window_len: int, stride: int, fractions) -> dict[str, float]:
    """Metrics keyed `c{pct}/{field}` for every calibration fraction."""
    scores = np.asarray(scores, dtype=np.float64)
    window_labels = np.asarray(window_labels).astype(np.int32)
    point_labels = np.asarray(point_labels).astype(np.int32).ravel()
    starts = window_starts(len(scores), stride)
    derived = np.array([point_labels[s:s + window_len].max() for s in starts], dtype=np.int32)
    if not np.array_equal(derived, window_labels):
        raise AssertionError(
            "window labels re-derived from the dataset (start = i * stride) do not match the "
            "evaluator's — the score order is not the window order, so the split would assign "
            "windows to the wrong part")
    segments = anomaly_segments(point_labels)
    n_points = len(point_labels)
    out: dict[str, float] = {"n_points": float(n_points), "n_events": float(len(segments)),
                             "n_windows": float(len(scores))}
    for fraction in fractions:
        key = f"c{int(round(fraction * 100)):02d}"
        requested, cut = place_cut(n_points, fraction, segments)
        calib = starts + window_len <= cut
        evaluation = starts >= cut
        ev_calib = sum(1 for s, e in segments if e <= cut)
        ev_eval = sum(1 for s, e in segments if s >= cut)
        assert ev_calib + ev_eval == len(segments), "an event straddles the cut"
        out.update({
            f"{key}/cut_requested": float(requested), f"{key}/cut": float(cut),
            f"{key}/cut_moved_by": float(cut - requested),
            f"{key}/calib_windows": float(calib.sum()),
            f"{key}/eval_windows": float(evaluation.sum()),
            f"{key}/dropped_windows": float((~calib & ~evaluation).sum()),
            f"{key}/calib_anomalous_windows": float(window_labels[calib].sum()),
            f"{key}/eval_anomalous_windows": float(window_labels[evaluation].sum()),
            f"{key}/calib_events": float(ev_calib), f"{key}/eval_events": float(ev_eval),
            f"{key}/calib_window_auroc": _auroc(scores[calib], window_labels[calib]),
            f"{key}/eval_window_auroc": _auroc(scores[evaluation], window_labels[evaluation]),
        })
    return out


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """The published window AUROC on a subset; NaN when the subset holds only one class."""
    if len(labels) == 0 or labels.min() == labels.max():
        return float("nan")
    return float(eval_classification(scores, labels, None)["auroc"])


def self_test() -> None:
    labels = np.zeros(100, dtype=np.int32)
    labels[18:25] = 1                       # a 10% cut (index 10) is clear of it
    labels[48:56] = 1                       # a 50% cut (index 50) lands inside -> moved to 48
    labels[80:83] = 1
    segs = anomaly_segments(labels)
    assert segs == [(18, 25), (48, 56), (80, 83)], segs
    assert place_cut(100, 0.10, segs) == (10, 10)
    assert place_cut(100, 0.50, segs) == (50, 48), "ties/nearest go to the segment start"
    assert place_cut(100, 0.55, segs) == (55, 56), "nearer boundary is the end"
    window_len, stride = 5, 5
    starts = window_starts(20, stride)
    wl = np.array([labels[s:s + window_len].max() for s in starts])
    rng = np.random.default_rng(0)
    scores = rng.random(20) + wl            # separable enough to have an AUROC
    got = split(scores, wl, labels, window_len, stride, [0.3, 0.5])
    assert got["c50/cut"] == 48.0 and got["c50/cut_moved_by"] == -2.0
    assert got["c50/calib_events"] == 1 and got["c50/eval_events"] == 2
    total = got["c30/calib_windows"] + got["c30/eval_windows"] + got["c30/dropped_windows"]
    assert total == 20
    full = float(eval_classification(scores, wl, None)["auroc"])
    assert 0.0 <= full <= 1.0
    try:
        split(scores[::-1], wl[::-1], labels, window_len, stride, [0.3])
    except AssertionError:
        pass
    else:
        raise AssertionError("a reordered score vector must be refused")
    print("calibration_split self-test OK")


if __name__ == "__main__":
    self_test()
