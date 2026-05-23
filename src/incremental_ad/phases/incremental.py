import argparse
import logging
import os
from dataclasses import replace

from incremental_ad.core import checkpoint
from incremental_ad.core.device import resolve_device
from incremental_ad.core.run import setup_run_dir
from incremental_ad.core.seed import set_seed
from incremental_ad.ops.eval import run_eval
from incremental_ad.ops.merge import run_merge
from incremental_ad.ops.train import run_train
from incremental_ad.training import evaluator, trainer

log = logging.getLogger(__name__)


def run_incremental(*, datasets: dict, models: dict, num_workers: int) -> None:
    """Whole incremental pipeline in one job on a static dataset:

      1. pretrain a base model on the `partial` slice,
      2. fine-tune it on each of the `n_finetune` chunks (ft_0 .. ft_{N-1}),
      3. merge the fine-tunings into the base via task arithmetic,
      4. evaluate the merged model on the test set.

    Everything lives in one run dir (base/, ft_i/, merged/) and one wandb group, so
    no parameters/paths are passed between jobs. Changing the split = re-run this phase.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", choices=datasets, required=True)
    parser.add_argument("--model", choices=models, required=True)

    # Split + merge params (phase-level).
    parser.add_argument("--partial-ratio", type=float, required=True)
    parser.add_argument("--n-finetune", type=int, required=True)
    parser.add_argument("--merge-scale", type=float, default=1.0)
    parser.add_argument("--eval-stride", type=int, default=1)

    known, _ = parser.parse_known_args()

    # Add the args for the specific dataset and model.
    datasets[known.dataset][0](parser)
    models[known.model][0](parser)

    # Two trainer configs (base pretrain + fine-tuning) and the eval config.
    trainer.add_args(parser, "train")
    trainer.add_args(parser, "finetune")
    evaluator.add_args(parser)

    args = parser.parse_args()

    dataset_cfg = datasets[args.dataset][1](args)
    model_cfg = models[args.model][1](args)
    pretrain_cfg = trainer.make_config(args, "train")
    finetune_cfg = trainer.make_config(args, "finetune")
    eval_cfg = evaluator.make_config(args)

    run_dir, run_id = setup_run_dir(args.experiment, args.phase, args.run_tag)
    os.environ["WANDB_DIR"] = str(run_dir)
    device = resolve_device(args.device)

    log.info(f"Phase: {args.phase}  (experiment={args.experiment})")

    # 1) Pretrain the base on the partial slice.
    base_dir = run_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    set_seed(pretrain_cfg.seed)

    run_train(
        experiment=args.experiment,
        phase=args.phase,
        run_tag="base",
        device=device,
        run_dir=base_dir,
        run_id=run_id,
        dataset_name=args.dataset,
        dataset_cfg=dataset_cfg,
        model_name=args.model,
        model_cfg=model_cfg,
        train_cfg=pretrain_cfg,
        train_slice="partial",
        partial_ratio=args.partial_ratio,
        n_finetune=args.n_finetune,
        num_workers=num_workers,
    )

    base_ckpt_path = base_dir / "checkpoints" / "best.pt"
    base_state = checkpoint.load_checkpoint(base_ckpt_path)["model_state"]

    # 2) Fine-tune from the base on each chunk.
    ft_ckpt_paths = []

    for i in range(args.n_finetune):

        ft_dir = run_dir / f"ft_{i}"
        ft_dir.mkdir(parents=True, exist_ok=True)

        set_seed(finetune_cfg.seed)

        run_train(
            experiment=args.experiment,
            phase=args.phase,
            run_tag=f"ft_{i}",
            device=device,
            run_dir=ft_dir,
            run_id=run_id,
            dataset_name=args.dataset,
            dataset_cfg=dataset_cfg,
            model_name=args.model,
            model_cfg=model_cfg,
            train_cfg=finetune_cfg,
            train_slice=f"ft_{i}",
            partial_ratio=args.partial_ratio,
            n_finetune=args.n_finetune,
            init_model_state=base_state,
            num_workers=num_workers,
        )
        
        ft_ckpt_paths.append(ft_dir / "checkpoints" / "best.pt")

    # 3) Merge the fine-tunings into the base (task arithmetic).
    merged_dir = run_dir / "merged"
    merged_ckpt_path = merged_dir / "checkpoints" / "best.pt"

    merged_ckpt = run_merge(
        base_checkpoint_path=base_ckpt_path,
        ft_checkpoint_paths=ft_ckpt_paths,
        scale=args.merge_scale,
        out_path=merged_ckpt_path,
    )

    # 4) Evaluate the merged model. Eval metrics assume stride=1 (see --eval-stride).
    model_name = merged_ckpt["configs"]["model_name"]
    merged_model_cfg = models[model_name][2](**merged_ckpt["configs"]["model"])
    eval_dataset_cfg = replace(dataset_cfg, stride=args.eval_stride)

    set_seed(eval_cfg.seed)

    run_eval(
        experiment=args.experiment,
        phase=args.phase,
        run_tag="merged",
        device=device,
        run_dir=merged_dir,
        run_id=run_id,
        group_run_id=run_id,
        eval_dataset_name=args.dataset,
        eval_dataset_cfg=eval_dataset_cfg,
        eval_cfg=eval_cfg,
        ckpt=merged_ckpt,
        checkpoint_path=merged_ckpt_path,
        model_name=model_name,
        model_cfg=merged_model_cfg,
        num_workers=num_workers,
    )
