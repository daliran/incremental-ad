# Reusable task-vector machinery: constructing task vectors from fine-tuned models and
# combining them back into a base model. Operates purely on state dicts
# (dict[str, Tensor]) — no knowledge of any specific model, task, or dataset.
from .task_vectors import merge_task_arithmetic
