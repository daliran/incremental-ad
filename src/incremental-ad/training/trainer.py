from dataclasses import dataclass
from typing import Literal

OptimizerType = Literal["adamw"]
SchedulerType = Literal["cosine", "constant"]

@dataclass
class TrainingConfig:
    epochs: int
    patience: int
    batch_size: int
    optimizer: OptimizerType
    weight_decay: float
    learning_rate: float
    grad_clip: float
    scheduler: SchedulerType
    warmup_ratio: float
