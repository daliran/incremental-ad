from dataclasses import dataclass
from typing import Literal

# --- Constants ---
HF_DATASET_PATH = "thuml/Time-Series-Library"
HF_DATASET_NAME = "SWaT"

Normalization = Literal["standard", "none"]

@dataclass
class SWaTConfig:
    window_len: int
    stride: int
    normalization: Normalization
