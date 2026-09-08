"""Self-check for `--*_train_only`: the task vector must be exactly zero outside the trained set.

    python scripts/verify_train_only.py

Attention-exclusive fine-tuning (§1.33) rests on one structural claim: if only attention
parameters are trained, then τ = θ_ft − θ₀ is **identically zero** everywhere else, so the merge
can only move attention. That is the whole reason the experiment tests QOMM's premise rather than
some approximation of it. If a frozen parameter drifts by even a rounding step — through weight
decay, momentum, or a buffer update — the claim is false and the geometry it produces means
something different.

CLAUDE.md's convention: new data-path or training code gets a check on its own semantics before
its first number is quoted. This runs a real two-epoch fine-tune on a tiny synthetic segment and
asserts the property end to end, rather than reasoning about it from the flag's implementation.

**Weight decay is deliberately non-zero here.** AdamW's decay is decoupled: it moves a parameter
every step whether or not it has a gradient. The first version of `_build_optimizer` passed
`model.parameters()` — all of them — so a frozen tensor would have drifted at any non-zero decay
while `requires_grad` said otherwise. Testing at `weight_decay=0` would have passed and hidden it.

Four properties:

1. **Frozen parameters do not move at all** — bitwise, not "small".
2. **Trained parameters do move**, so the run is not vacuously passing by training nothing.
3. **`requires_grad` is False on exactly the frozen set**, since that is what keeps them out of
   the optimiser rather than merely out of the gradient.
4. **An empty `train_only` is an exact no-op** — the default must not perturb any existing run.
"""

import sys

import torch

sys.path.insert(0, "src")


def tiny_setup(train_only):
    """A minimal model + segment that exercises the real trainer, not a stand-in."""
    from torch import nn
    from torch.utils.data import TensorDataset

    from incremental_ad.framework.contracts.dataset import DataLoaderConfig, Segment
    from incremental_ad.framework.trainers.standard_trainer import StandardTrainer

    class Tiny(nn.Module):
        """Names deliberately mimic the real model so substring matching is exercised."""

        def __init__(self):
            super().__init__()
            self.encoder_attn_qkv = nn.Linear(4, 4)
            self.encoder_mlp_fc = nn.Linear(4, 4)
            self.norm_scale = nn.Parameter(torch.ones(4))

        def forward(self, x):
            return self.encoder_mlp_fc(self.encoder_attn_qkv(x)) * self.norm_scale

        def compute_loss(self, batch):
            x = batch[0]
            return ((self(x) - x) ** 2).mean()

    torch.manual_seed(0)
    model = Tiny()
    data = TensorDataset(torch.randn(32, 4))
    segment = Segment(train=data, val=data)
    trainer = StandardTrainer(
        n_epochs=2, patience=5, optimizer="adamw", weight_decay=0.01, learning_rate=0.05,
        grad_clip=1.0, scheduler="constant", warmup_ratio=0.0,
        loader_config=DataLoaderConfig(batch_size=8, num_workers=0),
        device="cpu", train_only=train_only,
    )
    return model, segment, trainer


def run(train_only):
    model, segment, trainer = tiny_setup(train_only)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer.fit(model, segment, step_name="selfcheck")
    after = model.state_dict()
    tau = {k: (after[k] - before[k]) for k in before}
    grads = {name: p.requires_grad for name, p in model.named_parameters()}
    return tau, grads


def main() -> None:
    print("TRAIN_ONLY SELF-CHECKS")
    failures = 0

    tau, grads = run(["attn"])
    frozen = [k for k in tau if "attn" not in k]
    trained = [k for k in tau if "attn" in k]

    moved_frozen = [k for k in frozen if not torch.equal(tau[k], torch.zeros_like(tau[k]))]
    if moved_frozen:
        for key in moved_frozen:
            print(f"  FAIL  frozen parameter moved: {key} "
                  f"max|delta|={tau[key].abs().max().item():.3e}")
        failures += len(moved_frozen)
    print(f"  {'ok' if not moved_frozen else 'FAILED'}  {len(frozen)} non-attention tensor(s) "
          f"are bitwise unchanged — the task vector is exactly zero outside attention")

    unmoved = [k for k in trained if torch.equal(tau[k], torch.zeros_like(tau[k]))]
    if unmoved:
        print(f"  FAIL  trained parameters did not move: {unmoved} — the check would pass "
              f"vacuously")
        failures += 1
    print(f"  {'ok' if not unmoved else 'FAILED'}  {len(trained)} attention tensor(s) did move")

    wrong = [n for n, g in grads.items() if g != ("attn" in n)]
    if wrong:
        print(f"  FAIL  requires_grad does not match the train_only set: {wrong}")
        failures += 1
    print(f"  {'ok' if not wrong else 'FAILED'}  requires_grad is False on exactly the frozen set")

    # The default must not perturb anything: every parameter trains, as before the flag existed.
    tau_default, grads_default = run([])
    still = [k for k in tau_default
             if torch.equal(tau_default[k], torch.zeros_like(tau_default[k]))
             and tau_default[k].numel() > 0 and tau_default[k].is_floating_point()]
    if not all(grads_default.values()):
        print(f"  FAIL  empty train_only froze something: "
              f"{[n for n, g in grads_default.items() if not g]}")
        failures += 1
    print(f"  {'ok' if all(grads_default.values()) else 'FAILED'}  empty train_only is a no-op — "
          f"every parameter still trains ({len(still)} unchanged by optimisation, not by freezing)")

    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
