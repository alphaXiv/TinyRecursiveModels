# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Motivation

Tiny Recursion Model (TRM) is a recursive reasoning model that achieves amazing scores of 45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a tiny 7M parameters neural network. The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap. Currently, there is too much focus on exploiting LLMs rather than devising and expanding new lines of direction. With recursive reasoning, it turns out that “less is more”: you don’t always need to crank up model size in order for a model to reason and solve hard problems. A tiny model pretrained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

This work came to be after I learned about the recent innovative Hierarchical Reasoning Model (HRM). I was amazed that an approach using small models could do so well on hard tasks like the ARC-AGI competition (reaching 40% accuracy when normally only Large Language Models could compete). But I kept thinking that it is too complicated, relying too much on biological arguments about the human brain, and that this recursive reasoning process could be greatly simplified and improved. Tiny Recursion Model (TRM) simplifies recursive reasoning to its core essence, which ultimately has nothing to do with the human brain, does not require any mathematical (fixed-point) theorem, nor any hierarchy.

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;"/>
  <br/>
  <sub>TRM iteratively updates latent z and answer y.</sub>
  </p>

## Quickstart



```bash
# 1) Create env (Python 3.10+ recommended)
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 2) Install PyTorch (pick ONE that fits your machine)
# CPU only:
pip install torch torchvision torchaudio
# CUDA 12.6 wheels (Linux w/ NVIDIA drivers):
# pip install --pre --upgrade torch torchvision torchaudio \
#   --index-url https://download.pytorch.org/whl/nightly/cu126

# 3) Install project deps and optimizer
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2

# 4) Optional: log to Weights & Biases
# wandb login
```

## Datasets

All builders output into `data/<dataset-name>/` with the expected `train/` and `test/` splits plus metadata.

```bash
# ARC-AGI-1 (uses files in kaggle/combined already in this repo)
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

# Note: don't train on both ARC-AGI-1 and ARC-AGI-2 simultaneously if you plan to evaluate both; ARC-AGI-2 train includes some ARC-AGI-1 eval puzzles.

# Sudoku-Extreme (1k base, 1k augments)
python dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

# Maze-Hard (30x30)
python dataset/build_maze_dataset.py
```

## Training

Training is configured via Hydra. CLI overrides like `arch.L_layers=2` are applied on top of `config/cfg_pretrain.yaml` and the chosen `config/arch/*.yaml`.

Tips
- Set `+run_name=<name>` to label runs; checkpoints land in `checkpoints/<Project>/<Run>/`.
- Use `torchrun` for multi-GPU. Replace `--nproc-per-node` with your GPU count.

### ARC-AGI-1 (attention, multi-GPU)

```bash
run_name="pretrain_att_arc1concept"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc1concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr=2e-4 weight_decay=0.1 puzzle_emb_lr=1e-2 \
  global_batch_size=1536 lr_warmup_steps=4000 \
  epochs=100000 eval_interval=5000 checkpoint_every_eval=True \
  +run_name=${run_name} ema=True
```

### ARC-AGI-2 (attention, multi-GPU)

```bash
run_name="pretrain_att_arc2concept"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/arc2concept-aug-1000]" \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr=2e-4 weight_decay=0.1 puzzle_emb_lr=1e-2 \
  global_batch_size=1536 lr_warmup_steps=4000 \
  epochs=100000 eval_interval=5000 checkpoint_every_eval=True \
  +run_name=${run_name} ema=True
```

### Sudoku-Extreme (MLP and attention variants)

MLP-Tiny variant:

```bash
run_name="pretrain_mlp_t_sudoku"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr_warmup_steps=4000 \
  global_batch_size=1536 \
  +run_name=${run_name} ema=True
```

Attention variant:

```bash
run_name="pretrain_att_sudoku"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  lr_warmup_steps=4000 \
  global_batch_size=1536 \
  +run_name=${run_name} ema=True
```

### Maze-Hard 30x30 (attention)

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  global_batch_size=1536 lr_warmup_steps=4000 \
  checkpoint_every_eval=True \
  +run_name=${run_name} ema=True
```

## Evaluate checkpoints (local)

Use the evaluation-only runner that mirrors `pretrain.py` evaluation.

Single GPU / CPU smoke test (one batch):

```bash
python scripts/run_eval_only.py \
  --checkpoint trained_models/step_32550_sudoku_epoch50k \
  --dataset data/sudoku-extreme-1k-aug-1000 \
  --one-batch
```

Multi-GPU full eval:

```bash
torchrun --nproc_per_node=8 scripts/run_eval_only.py \
  --checkpoint trained_models/step_32550_sudoku_epoch50k \
  --dataset data/sudoku-extreme-1k-aug-1000 \
  --outdir checkpoints/sudoku_eval_run \
  --eval-save-outputs inputs labels puzzle_identifiers preds \
  --global-batch-size 1536 \
  --apply-ema
```

Maze example:

```bash
torchrun --nproc_per_node=8 scripts/run_eval_only.py \
  --checkpoint trained_models/maze_hard_step_32550 \
  --dataset data/maze-30x30-hard-1k \
  --outdir checkpoints/maze_eval_run \
  --global-batch-size 1536 \
  --apply-ema
```

ARC-AGI-1 example (attention):

```bash
torchrun --nproc_per_node=8 scripts/run_eval_only.py \
  --checkpoint trained_models/step_259320_arc_ag1_attn_type_h3l4 \
  --dataset data/arc1concept-aug-1000 \
  --outdir checkpoints/arc1_eval_run \
  --global-batch-size 1024 \
  --apply-ema
```

## Evaluate in the cloud (Modal)

We ship a Modal app for easy, reproducible, GPU-backed evaluation and web endpoints.

```bash
# 1) Install and authenticate
pip install modal
modal token new

# 2) ARC-AGI-1 eval job (A100 x2 by default in infra)
modal run infra/modal_app.py::run_eval_arc_job \
  --dataset-path=data/arc1concept-aug-1000 \
  --batch-size=512

# 3) Maze eval job
modal run infra/modal_app.py::run_eval_maze_job \
  --batch-size=256
```

The app also exposes simple visualizers and a realtime endpoint; see function docstrings in `infra/modal_app.py`.

## Visualization

Open the HTML pages directly in a browser:

- `arc_visualizer.html`
- `maze_visualizer.html`
- `sudoku_visualizer.html`
- `unified_visualizer.html`

## Pretrained Weights

If you'd like to download the pretrained model weights used in experiments, they are available on Hugging Face:

- Maze (30x30 TRM weights): https://huggingface.co/alphaXiv/trm-model-maze
- Sudoku (TRM weights): https://huggingface.co/alphaXiv/trm-model-sudoku
- ARC AGI 1 (TRM attention weights): https://huggingface.co/alphaXiv/trm-model-arc-agi-1

## Reproducing paper numbers

- Build the exact datasets above (`arc1concept-aug-1000`, `arc2concept-aug-1000`, `maze-30x30-hard-1k`, `sudoku-extreme-1k-aug-1000`).
- Use the training commands in this README (matching `scripts/cmd.sh` but with minor fixes like line breaks and env-safe flags).
- Keep seeds at defaults (`seed=0` in `config/cfg_pretrain.yaml`); runs are deterministic modulo CUDA kernels.
- Evaluate with `scripts/run_eval_only.py` and report `exact_accuracy` and per-task metrics. The script will compute Wilson 95% CI when dataset metadata is present.

## Troubleshooting

- PyTorch install: pick wheels matching your CUDA; on macOS (CPU/MPS) training will be very slow — prefer Linux + NVIDIA GPU for training.
- NCCL errors: ensure you run under `torchrun` on a Linux box with GPUs and that `nvidia-smi` shows all devices.
- Checkpoints and EMA: training saves EMA by default when `ema=True`; the eval script applies EMA unless disabled.
- Missing optimizer: install `adam-atan2` (see quickstart) for training. Evaluation-only runs do not require it.


This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
