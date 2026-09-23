# 🧩 Model Merging Demo

This repository provides a minimal example of **task vector merging** —
a simple yet extensible framework to combine fine-tuned models in weight space.

You can easily swap between multiple merging strategies such as:

- **TaskArithmetic (TA)** — Editing models with Task Arithmetic (plain task vector addition)
- **DARE** — Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch
- **ISO** — No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces (Isotropic Model Merging)
- **TIES** — TIES-Merging: Resolving Interference When Merging Models (parameter importance based)
- **TSV** — Task Singular Vectors: Reducing Task Interference in Model Merging

---

## 🔧 Overview

The script [`main.py`](./main.py) demonstrates how to:
1. Generate synthetic task vectors (layer-wise tensors)
2. Add them to a merging pool
3. Perform model merging with a chosen algorithm
4. Optionally apply the merged task vector to a base model

---

## 🧠 Example: Task Merging (Unified Interface)

You can easily choose the desired merging strategy and scaling factor directly from the command line.

### 🔧 Available arguments

| Argument | Type | Default | Choices | Description |
|-----------|------|----------|----------|--------------|
| `--merging` | `str` | `ta` | `ta`, `dare`, `iso`, `ties`, `tsv` | Selects the merging method to apply. |
| `--alpha_merging` | `float` | `1.0` | any positive float | Global scaling factor applied to the merged task vector. |

### 💻 Example usage

```bash
# Default: Task Arithmetic (TA) with alpha = 1.0
python main.py

# Use DARE merging with alpha = 0.5
python main.py --merging dare --alpha_merging 0.5

# Use ISO (Isotropic) merging
python main.py --merging iso

# Try TIES-Merging
python main.py --merging ties

# Try Task-Specific Vector fusion
python main.py --merging tsv