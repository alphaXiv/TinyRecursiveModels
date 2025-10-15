#!/usr/bin/env python3
"""Distributed evaluation runner that reuses pretrain.py evaluate() behavior.

This script constructs a PretrainConfig from a YAML file (default: config/cfg_pretrain.yaml),
overrides a few fields from the CLI (checkpoint path, dataset path, eval_save_outputs),
creates the test dataloader and model (loading checkpoint), and runs evaluate()
exactly like pretrain.py. It is safe to run with torchrun for multi-GPU evaluation.

Example (single-process):
  python scripts/run_eval_only.py --checkpoint /path/to/step_50000 --dataset data/maze-30x30-hard-1k

Example (distributed via torchrun):
  torchrun --nproc_per_node=8 scripts/run_eval_only.py --checkpoint /path/to/step_50000 --dataset /data/maze-30x30-hard-1k
"""

import os
import argparse
import yaml
import copy
import torch
import torch.distributed as dist
import numpy as np
# Reuse functions and classes from pretrain.py
from pretrain import (
    create_dataloader,
    create_evaluators,
    init_train_state,
    evaluate,
    PretrainConfig,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/cfg_pretrain.yaml', help='YAML config file (pydantic fields)')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint file to load')
    p.add_argument('--dataset', required=True, help='Path to dataset directory to evaluate (overrides data_paths_test)')
    p.add_argument('--outdir', default=None, help='Directory to save evaluation preds (overrides checkpoint_path in config)')
    p.add_argument('--eval-save-outputs', nargs='+', default=['inputs','labels','puzzle_identifiers','preds'], help='List of keys to save during evaluation')
    p.add_argument('--global-batch-size', type=int, default=None, help='Global batch size override for evaluation')
    p.add_argument('--apply-ema', action='store_true', help='Attempt to apply EMA weights for evaluation')
    p.add_argument('--ema-shadow', default=None, help='Path to EMA shadow state dict (optional). If provided, it will be loaded into EMAHelper before applying EMA.')
    p.add_argument('--repeats', type=int, default=1, help='Number of times to run evaluation (will save outputs per run)')
    p.add_argument('--seed-start', type=int, default=0, help='Offset added to seed for each repeat (seed + seed-start + rep)')
    p.add_argument('--eval-only', action='store_true', help='Run in eval-only mode (skip optimizer creation when initializing model)')
    return p.parse_args()


def main():
    args = parse_args()

    # Distributed init if running under torchrun
    RANK = 0
    WORLD_SIZE = 1
    CPU_GROUP = None

    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        CPU_GROUP = dist.new_group(backend='gloo')

    # Compose config using Hydra programmatic API on rank 0 and broadcast
    from hydra import initialize, compose
    from omegaconf import OmegaConf

    config_obj = None
    objects = [None]
    if RANK == 0:
        # Determine config directory and base name from args.config (e.g. 'config/cfg_pretrain.yaml')
        config_path = os.path.dirname(args.config) or 'config'
        config_name = os.path.splitext(os.path.basename(args.config))[0]

        # Compose Hydra config; do not apply CLI overrides here (we'll apply programmatic overrides below)
        with initialize(config_path=config_path, job_name="run_eval_only"):
            hydra_cfg = compose(config_name=config_name)

        # Convert to plain dict (resolve interpolations)
        cfg = OmegaConf.to_container(hydra_cfg, resolve=True)

        # Apply CLI overrides on top of the composed config
        cfg['data_paths_test'] = [args.dataset]
        cfg['load_checkpoint'] = args.checkpoint
        if args.outdir is not None:
            cfg['checkpoint_path'] = args.outdir
        if args.global_batch_size is not None:
            cfg['global_batch_size'] = args.global_batch_size
        cfg['eval_save_outputs'] = args.eval_save_outputs

        # Print the final composed config on rank 0 for debugging
        try:
            print('\nComposed config (after Hydra compose + CLI overrides):')
            print(yaml.safe_dump(cfg, sort_keys=False))
        except Exception:
            print('Warning: failed to pretty-print composed config')

        # Construct pydantic PretrainConfig from fully composed config
        config_obj = PretrainConfig(**cfg)
        objects = [config_obj]

    if WORLD_SIZE > 1:
        dist.broadcast_object_list(objects, src=0)

    config = objects[0]

    # Ensure config present
    if config is None:
        raise RuntimeError('Failed to load config via broadcast; config is None on this rank')

    # Seed RNGs
    torch.random.manual_seed(config.seed + RANK)

    # Create dataloaders
    # train_loader, train_metadata = create_dataloader(config, 'train', rank=RANK, world_size=WORLD_SIZE, test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    try:
        eval_loader, eval_metadata = create_dataloader(config, 'test', rank=RANK, world_size=WORLD_SIZE, test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size)
    except Exception:
        if RANK == 0:
            print('NO EVAL DATA FOUND')
        return

    # Evaluators
    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        if RANK == 0:
            print('No evaluator found')
        evaluators = []

    # Init model & train_state (loads checkpoint on rank 0 inside create_model).
    # Pass is_eval according to CLI flag to skip optimizer construction in evaluation-only runs.
    train_state = init_train_state(config, eval_metadata, rank=RANK, world_size=WORLD_SIZE, is_eval=bool(args.eval_only))

    # Optionally switch to EMA copy if requested by CLI or config
    train_state_eval = train_state
    if args.apply_ema or config.ema:
        # Import EMA helper
        from models.ema import EMAHelper

        if RANK == 0:
            print('Preparing EMA for evaluation...')

        ema_helper = EMAHelper(mu=config.ema_rate)
        # Register model parameters
        ema_helper.register(train_state.model)

        # If user provided an EMA shadow file, load and broadcast it to all ranks
        ema_state = None
        objects = [None]
        if args.ema_shadow is not None:
            if RANK == 0:
                ema_state = torch.load(args.ema_shadow, map_location='cpu')
                objects = [ema_state]

        if WORLD_SIZE > 1:
            dist.broadcast_object_list(objects, src=0)

        if objects[0] is not None:
            # Load shadow into helper
            ema_helper.load_state_dict(objects[0])
            if RANK == 0:
                print('Loaded EMA shadow state and applying EMA copy for evaluation.')
            train_state_eval = copy.deepcopy(train_state)
            train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
        else:
            # No explicit shadow file provided. If the checkpoint already contains EMA weights (saved by training
            # after swapping to the EMA copy), then load_checkpoint already set those weights when init_train_state ran.
            # We still create a deepcopy for safety to avoid modifying the main train_state model during eval.
            if RANK == 0:
                print('No EMA shadow provided — assuming checkpoint contains EMA weights (if training saved EMA).')
            train_state_eval = copy.deepcopy(train_state)

    # Run evaluation repeats
    original_ckpt_path = config.checkpoint_path
    # Prepare metric accumulators (only used on rank 0)
    metric_acc = {}  # {set_name: {metric_name: [vals]}}

    for rep in range(args.repeats):
        # reseed per repeat
        torch.random.manual_seed(config.seed + RANK + args.seed_start + rep)

        # create per-repeat checkpoint path so outputs don't clash
        if original_ckpt_path is None:
            rep_ckpt = os.path.join('checkpoints', f'eval_run_{rep}')
        else:
            rep_ckpt = original_ckpt_path + f'_run{rep}' if args.repeats > 1 else original_ckpt_path
        config.checkpoint_path = rep_ckpt
        if RANK == 0:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            print(f"Starting evaluation run {rep+1}/{args.repeats}, outputs -> {config.checkpoint_path}")

        # deepcopy eval state to avoid side-effects
        ts = copy.deepcopy(train_state_eval)
        ts.model.eval()

        metrics = evaluate(
            config=config,
            train_state=ts,
            eval_loader=eval_loader,
            eval_metadata=eval_metadata,
            evaluators=evaluators,
            rank=RANK,
            world_size=WORLD_SIZE,
            cpu_group=CPU_GROUP,
        )

        if RANK == 0 and metrics is not None:
            print(f'Run {rep+1} metrics:')
            print(metrics)
            # Accumulate metrics for final summary
            for set_name, m in metrics.items():
                metric_acc.setdefault(set_name, {})
                for key in ('accuracy', 'exact_accuracy'):
                    if key in m:
                        metric_acc[set_name].setdefault(key, []).append(m[key])

    if dist.is_initialized():
        dist.destroy_process_group()

    # After all repeats, print aggregate stats (rank 0)
    if RANK == 0 and args.repeats > 1:
        
        print('\nAggregate metrics across repeats:')
        for set_name, metrics_dict in metric_acc.items():
            print(f"Set: {set_name}")
            for key, vals in metrics_dict.items():
                arr = np.array(vals, dtype=float)
                mean = float(arr.mean())
                std = float(arr.std(ddof=0))
                print(f"  {key}: mean={mean:.6f}, std={std:.6f} (n={len(vals)})")


if __name__ == '__main__':
    main()
