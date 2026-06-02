import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(requested_device: str = "auto") -> torch.device:

    if requested_device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = requested_device

    try:
        device = torch.device(device_str)
        torch.zeros(1).to(device)
    except (RuntimeError, AssertionError) as e:
        log.warning(f"'{device_str}' unavailable ({e}), falling back to cpu")
        device = torch.device("cpu")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / 1024**3 # gigabytes
        log.info(f"Using {device} — {props.name} ({vram_gb:.1f} GB VRAM)")
    else:
        log.info(f"Using {device}")

    return device
