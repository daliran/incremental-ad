import argparse
import torch
from typing import Dict

from merging import add_merging_args, get_merging_function


def make_task_vector(layer_specs, device) -> Dict[str, torch.Tensor]:
    """
    Create a synthetic task vector with fixed shapes on the given device.
    layer_specs: list of (name, shape) tuples
    """
    tv = {}
    for name, shape in layer_specs:
        tv[name] = torch.randn(*shape, device=device)
    return tv


def main():
    # 0) Parse command-line arguments
    parser = argparse.ArgumentParser(description="Model Merging Demo")
    add_merging_args(parser)
    command_args = parser.parse_args()

    # 1) Choose device (CPU fallback if CUDA is unavailable)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 2) Define a common set of layers (all task vectors must have identical keys/shapes)
    layer_specs = [
        ("encoder.l1.weight", (4, 4)),
        ("encoder.l1.bias", (4,)),
        ("encoder.l2.weight", (4, 4)),
        ("head.weight", (2, 4)),
        ("head.bias", (2,))
    ]

    # 3) Create three task vectors (e.g., from three fine-tuned models)
    task_vec_A = make_task_vector(layer_specs, device)
    task_vec_B = make_task_vector(layer_specs, device)
    task_vec_C = make_task_vector(layer_specs, device)

    # 4) Instantiate the merger with a global scaling alpha using parsed arguments
    merger = get_merging_function(command_args, device)
    print(f"[INFO] Initialized merger: {command_args.merging.upper()} "
          f"(alpha={command_args.alpha_merging})")

    # 5) Add the task vectors to the pool
    merger.add(task_vec_A)
    merger.add(task_vec_B)
    merger.add(task_vec_C)

    # 6) Merge: by default you get a dict {layer_name: tensor}
    # merger.set_per_task_weights([1.0, 0.5, 1.5])  # Example of setting per-task weights
    merged_dict = merger.merge()  # returns a dict with the same keys as the inputs

    # 7) (Optional) If you want tensors in a specific order, pass 'names'
    ordered_names = [name for name, _ in layer_specs]
    merged_list = merger.merge(names=ordered_names)  # returns a list of tensors in this order

    # 8) Show a quick summary
    print("\n[INFO] Merged dict summary (mean/std per param):")
    for k, v in merged_dict.items():
        print(f"  {k:20s}  shape={tuple(v.shape)}  mean={v.mean().item():+.4f}  std={v.std().item():+.4f}")

    print("\n[INFO] First tensor in ordered list matches:", ordered_names[0],
          " shape=", tuple(merged_list[0].shape))

    # 9) (Optional) Apply merged task vector to a base model's state_dict:
    # base_state = base_model.state_dict()
    # for k in base_state.keys():
    #     if k in merged_dict:
    #         base_state[k] = base_state[k] + merged_dict[k]   # TA-style addition
    # base_model.load_state_dict(base_state)


if __name__ == "__main__":
    main()
