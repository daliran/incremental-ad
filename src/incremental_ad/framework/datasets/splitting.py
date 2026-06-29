"""Generic time-series partitioning for incremental experiments.

Splits a series of length ``n`` into a baseline partition and ``n_finetune_segments``
equal fine-tuning partitions, and carves a validation tail from any ``[start, end)``
range. Datasets call these to compute index ranges, then build their own window
datasets from the slices — keeping the split arithmetic in one place, independent of
the per-task window format. Validation of the SplitConfig itself lives on
``SplitConfig.validate()``.
"""

from incremental_ad.framework.contracts.dataset import SplitConfig

Range = tuple[int, int]


def baseline_range(n: int, cfg: SplitConfig) -> Range:
    """``[0, use_end)`` — the slice the base model trains on.

    ``use_end = baseline_use_fraction * (baseline_fraction * n)``. Fine-tune segments are
    anchored at ``baseline_fraction * n`` regardless of ``baseline_use_fraction``, so a
    use-fraction < 1 leaves an intentional gap that no partition uses.
    """
    baseline_end = int(n * cfg.baseline_fraction)
    use_end = int(baseline_end * cfg.baseline_use_fraction)
    return (0, use_end)


def finetune_ranges(n: int, cfg: SplitConfig) -> list[Range]:
    """Equal fine-tuning partitions tiling ``[baseline_fraction*n, ...)``.

    Empty when ``n_finetune_segments == 0``. Uses floor division, so up to
    ``n_finetune_segments - 1`` trailing timesteps may be left unused — the segment size
    matches what each fine-tune actually trains on.
    """
    if cfg.n_finetune_segments <= 0:
        return []
    anchor = int(n * cfg.baseline_fraction)
    size = (n - anchor) // cfg.n_finetune_segments
    return [
        (anchor + i * size, anchor + (i + 1) * size)
        for i in range(cfg.n_finetune_segments)
    ]


def all_segment_ranges(n: int, cfg: SplitConfig) -> list[Range]:
    """Baseline range first, then each fine-tune range."""
    return [baseline_range(n, cfg), *finetune_ranges(n, cfg)]


def val_tail_split(start: int, end: int, val_fraction: float) -> tuple[Range, Range]:
    """Split ``[start, end)`` into ``(train, val)`` where val is the last ``val_fraction``.

    Temporal order is preserved (no shuffling): validation is always the tail.
    """
    val_size = int((end - start) * val_fraction)
    split = end - val_size
    return (start, split), (split, end)
