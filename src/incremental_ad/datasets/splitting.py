"""Reusable, dataset-agnostic helpers to carve a training series into slices.

These work purely on indices over ``[0, n)``, so any time-series dataset can use
them. A dataset that knows real timestamps (e.g. split by day/week) can compute
its own slice boundaries and still reuse :func:`val_tail_split` + windowing — the
phases only ever ask for a named slice, they never compute boundaries themselves.
"""


def equal_chunks(
    n: int, partial_ratio: float, n_finetune: int
) -> dict[str, tuple[int, int]]:
    """Partition ``[0, n)`` into a ``partial`` prefix then ``n_finetune`` equal chunks.

    Returns ``{"partial": (start, end), "ft_0": (...), ..., "ft_{N-1}": (...)}``.
    The partial prefix is the first ``partial_ratio`` fraction; the remainder is
    split into ``n_finetune`` equal chunks (the last one absorbs any remainder).
    """
    if not 0.0 < partial_ratio < 1.0:
        raise ValueError(f"partial_ratio must be in (0, 1), got {partial_ratio}")
    if n_finetune < 1:
        raise ValueError(f"n_finetune must be >= 1, got {n_finetune}")

    partial_end = int(n * partial_ratio)

    remaining = n - partial_end

    if partial_end <= 0 or remaining <= 0:
        raise ValueError(
            f"partial_ratio {partial_ratio} leaves an empty partial or fine-tuning "
            f"region for n={n}"
        )

    chunk = remaining // n_finetune

    if chunk <= 0:
        raise ValueError(
            f"n_finetune {n_finetune} is too large for {remaining} remaining timesteps"
        )

    chunks = {"partial": (0, partial_end)}

    cursor = partial_end

    for i in range(n_finetune):
        # the last chunk absorbs the remainder so the chunks tile [partial_end, n).
        end = n if i == n_finetune - 1 else cursor + chunk
        chunks[f"ft_{i}"] = (cursor, end)
        cursor = end

    return chunks


def val_tail_split(
    start: int, end: int, val_ratio: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Split ``[start, end)`` into (train, val) where val is the last ``val_ratio``.

    Temporal order is preserved (no shuffling): validation is always the tail.
    """
    val_size = int((end - start) * val_ratio)
    split = end - val_size
    return (start, split), (split, end)
