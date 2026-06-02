def equal_chunks(
    n: int, partial_ratio: float, n_finetune: int, base_ratio: float = 1.0
) -> dict[str, tuple[int, int]]:
    """Partition ``[0, n)`` into a ``partial`` prefix then ``n_finetune`` equal chunks.

    Returns ``{"partial": (0, partial_end), "base": (0, base_end),
    "ft_0": (...), ..., "ft_{N-1}": (...)}``.

    ``partial`` is the full allocation for the base phase (first ``partial_ratio``
    fraction).  ``base`` is the slice the base model actually trains on — the first
    ``base_ratio`` of ``partial``.  When ``base_ratio=1.0`` (default), ``base``
    equals ``partial``.  The ft chunks are always anchored at ``partial_end``
    regardless of ``base_ratio``.
    """
    if not 0.0 < partial_ratio < 1.0:
        raise ValueError(f"partial_ratio must be in (0, 1), got {partial_ratio}")
    if n_finetune < 1:
        raise ValueError(f"n_finetune must be >= 1, got {n_finetune}")
    if not 0.0 < base_ratio <= 1.0:
        raise ValueError(f"base_ratio must be in (0, 1], got {base_ratio}")

    partial_end = int(n * partial_ratio)
    base_end = int(partial_end * base_ratio)

    remaining = n - partial_end

    if partial_end <= 0 or remaining <= 0:
        raise ValueError(
            f"partial_ratio {partial_ratio} leaves an empty partial or fine-tuning "
            f"region for n={n}"
        )

    if base_end <= 0:
        raise ValueError(
            f"base_ratio {base_ratio} leaves an empty base slice for partial_end={partial_end}"
        )

    chunk = remaining // n_finetune

    if chunk <= 0:
        raise ValueError(
            f"n_finetune {n_finetune} is too large for {remaining} remaining timesteps"
        )

    chunks = {
        "partial": (0, partial_end),
        "base": (0, base_end),
    }

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
