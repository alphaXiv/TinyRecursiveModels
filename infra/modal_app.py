
"""Modal app for maze solving evaluation and prediction.

Usage:
  modal run infra/modal_app.py:run_eval_local --checkpoint-path data/maze-30x30-hard-1k/step_50000 --dataset-path data/maze-30x30-hard-1k/test

Notes:
- This script runs evaluation on mounted repo data using multi-GPU torchrun.
- Provides web endpoints for visualization and prediction.
"""

import os
import subprocess
import sys
from huggingface_hub import hf_hub_download
import modal
from fastapi import Response, HTTPException
import shutil

# Create a persistent volume for repo and data
volume = modal.Volume.from_name("tinyrecursive-data", create_if_missing=True)

IMAGE = (
    modal.Image.debian_slim()
    .run_commands(
        "apt-get update",
        "apt-get install -y git",
        "pip install --upgrade pip"
    )
    .pip_install("fastapi[standard]")
    .pip_install("uvicorn[standard]")
    .pip_install([
        "wandb",
        "pyyaml",
        "numpy",
        "Pillow",
        "huggingface_hub",
        "tqdm",
        "pandas",
        "einops",
        "coolname",
        "pydantic",
        "argdantic",
        "omegaconf",
        "hydra-core",
        "packaging",
        "ninja",
        "wheel",
        "setuptools",
        "setuptools-scm",
        "pydantic-core",
        "numba",
        "triton"
    ])
    .pip_install([
        "torch",
        "torchvision",
        "torchaudio"
    ], index_url="https://download.pytorch.org/whl/cu126")
)

APP_NAME = "tinyrecursive-eval"
app = modal.App(name=APP_NAME, image=IMAGE)

# CORS: allow visualizer and predict to call across Modal subdomains
from fastapi.middleware.cors import CORSMiddleware
fastapi_app = None
try:
    # Create a small FastAPI app only to attach middleware; Modal will mount endpoints into its own FastAPI instance.
    from fastapi import FastAPI
    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Optionally restrict to specific Modal domains
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass


# Helpers
def _safe_name(s: str) -> str:
    return ''.join(ch for ch in s if ch.isalnum() or ch in ('-', '_')).lower()


def _ensure_repo(repo_dir: str = "/data/repo") -> str:
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    
    # if os.path.exists(repo_dir):
        # shutil.rmtree(repo_dir)
    
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    # Ensure run_eval_only.py is at repo root (idempotent). If not, copy from scripts/ if present.
    try:
        src = os.path.join(repo_dir, "scripts", "run_eval_only.py")
        dst = os.path.join(repo_dir, "run_eval_only.py")
        if not os.path.exists(dst) and os.path.exists(src):
            try:
                os.replace(src, dst)
                print(f"Moved eval runner to repo root: {dst}")
            except Exception:
                shutil.copy2(src, dst)
                print(f"Copied eval runner to repo root: {dst}")
    except Exception as e:
        print("WARNING: Failed to place run_eval_only.py at repo root:", e)
    return repo_dir


def _symlink_latest(parent_dir: str, run_dir: str):
    """Create or update a 'latest' symlink under parent_dir pointing to run_dir.
    If symlink creation fails (FS limitations), write latest.txt with the run name.
    """
    latest_link = os.path.join(parent_dir, "latest")
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            try:
                os.remove(latest_link)
            except Exception:
                pass
        os.symlink(run_name, latest_link)
    except Exception:
        # Fallback: write a file
        try:
            with open(os.path.join(parent_dir, "latest.txt"), "w") as f:
                f.write(run_name)
        except Exception as e:
            print("Failed to record latest run:", e)


def _pick_latest_run(base_dir: str) -> str | None:
    """Return the absolute path of the latest run dir under base_dir.
    Prefers 'latest' symlink, else newest by mtime. Returns None if none.
    """
    if not os.path.isdir(base_dir):
        return None
    latest_link = os.path.join(base_dir, "latest")
    if os.path.islink(latest_link):
        target = os.readlink(latest_link)
        run_dir = os.path.join(base_dir, target)
        if os.path.isdir(run_dir):
            return run_dir
    # else choose most recent subdir
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != "latest"]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def _get_eval_script_path(repo_dir: str) -> str:
    """Return absolute path to run_eval_only.py. Prefer repo root, else scripts/.
    Logs existence for debugging.
    """
    candidates = [
        os.path.join(repo_dir, "run_eval_only.py"),
        os.path.join(repo_dir, "scripts", "run_eval_only.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"Resolved eval script: {p} (exists=True)")
            return p
    print(f"Resolved eval script: not found in candidates {candidates}")
    # Return the preferred location even if missing to surface a clearer error downstream
    return candidates[0]


# Internal helpers that perform the actual work. These can be called from both
# webhook endpoints and job/CLI functions without chaining Modal function calls.
def _do_prepare_dataset(include_maze: bool,
                        include_sudoku: bool,
                        sudoku_output_dir: str,
                        sudoku_subsample_size: int,
                        sudoku_num_aug: int):
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)

    # Install requirements (idempotent)
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    if include_maze:
        cmd = ["python", "dataset/build_maze_dataset.py"]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)

    if include_sudoku:
        cmd = [
            "python", "dataset/build_sudoku_dataset.py",
            "--output-dir", sudoku_output_dir,
            "--subsample-size", str(sudoku_subsample_size),
            "--num-aug", str(sudoku_num_aug),
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)

    return {"status": "Datasets prepared", "maze": include_maze, "sudoku": include_sudoku, "sudoku_output_dir": sudoku_output_dir}


def _do_download_sudoku_weights(model: str):
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)

    model = (model or "mlp").lower()
    if model not in ("mlp", "attn"):
        raise HTTPException(status_code=400, detail="model must be 'mlp' or 'attn'")

    save_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights")
    os.makedirs(save_dir, exist_ok=True)

    if model == "mlp":
        filename = "step_32550_sudoku_epoch50k"
    else:
        filename = "step_39060_sudoku_60k_epoch_attn_type"

    ckpt_path = hf_hub_download(
        repo_id="YuvrajSingh9886/sudoku-extreme-trm",
        filename=filename,
        local_dir=save_dir,
    )
    return {"status": "ok", "model": model, "checkpoint": ckpt_path}


def _do_download_all_weights():
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)
    results: dict[str, str] = {}
    # Maze weights
    maze_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights")
    os.makedirs(maze_dir, exist_ok=True)
    maze_ckpt = hf_hub_download(repo_id="YuvrajSingh9886/maze-hard-trm", filename="step_32550", local_dir=maze_dir)
    results["maze"] = maze_ckpt
    # Sudoku
    results["sudoku_mlp"] = _do_download_sudoku_weights("mlp")["checkpoint"]  # type: ignore
    results["sudoku_attn"] = _do_download_sudoku_weights("attn")["checkpoint"]  # type: ignore
    return {"status": "ok", "paths": results}


def _do_run_eval_sudoku(model: str, dataset_path: str | None, batch_size: int = 64):
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)

    # Ensure weights exist
    resp = _do_download_sudoku_weights(model)
    ckpt_path = resp["checkpoint"]

    # Dataset path: prefer 1k-aug-1000 if present, else fallback to full, unless overridden
    if dataset_path:
        dataset_dir = dataset_path
    else:
        dataset_dir = os.path.join(repo_dir, "data", "sudoku-extreme-1k-aug-1000")
        if not os.path.isdir(dataset_dir):
            dataset_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full")
    if not os.path.isdir(dataset_dir):
        print("WARNING: Sudoku dataset folder not found at", dataset_dir)

    # Output in per-run directory: out/sudoku/<model>/<run_id>
    parent = os.path.join(repo_dir, "out", "sudoku", (model or "mlp").lower())
    os.makedirs(parent, exist_ok=True)
    import time
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Choose arch override for MLP vs attention by temporarily swapping arch file
    config_path = os.path.join(repo_dir, "config", "cfg_pretrain.yaml")
    arch_dir = os.path.join(repo_dir, "config", "arch")
    trm_path = os.path.join(arch_dir, "trm.yaml")
    backup_path = os.path.join(arch_dir, "trm.yaml.bak")

    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node=2", eval_script,
        "--config", config_path,
        "--checkpoint", ckpt_path,
        "--dataset", dataset_dir,
        "--outdir", out_dir,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]

    need_mlp = (model or "mlp").lower() == "mlp"
    restored = False
    try:
        if need_mlp:
            # Backup original trm.yaml and set mlp_t: True inline
            if os.path.exists(trm_path):
                shutil.copy2(trm_path, backup_path)
            # Parse YAML and set mlp_t True
            import yaml as _yaml
            with open(trm_path, "r", encoding="utf-8") as f:
                data = _yaml.safe_load(f)
            # Set flag
            data["mlp_t"] = True
            # Write back
            with open(trm_path, "w", encoding="utf-8") as f:
                _yaml.safe_dump(data, f, sort_keys=False)
        print("Running Sudoku eval:", " ".join(cmd))
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        # Restore original arch file if we swapped it
        if need_mlp and os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, trm_path)
                os.remove(backup_path)
                restored = True
            except Exception as _e:
                print("WARNING: Failed to restore original arch file:", _e)
    _symlink_latest(parent, out_dir)
    return {"status": "Evaluation completed", "output_dir": out_dir, "result": getattr(result, 'returncode', 0), "run_id": run_id}


def _do_run_eval_maze(batch_size: int = 64):
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)

    # Ensure weights
    _do_download_all_weights()
    ckpt_path = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights", "step_32550")
    dataset_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k")
    if not os.path.isdir(dataset_dir):
        print("WARNING: Maze dataset folder not found at", dataset_dir)

    # subprocess.run(["ls"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    parent = os.path.join(repo_dir, "out", "maze", "default")
    os.makedirs(parent, exist_ok=True)
    import time
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node=2", eval_script,
        "--checkpoint", ckpt_path,
        "--dataset", dataset_dir,
        "--outdir", out_dir,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    print("Running Maze eval:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    _symlink_latest(parent, out_dir)
    return {"status": "Evaluation completed", "output_dir": out_dir, "result": getattr(result, 'returncode', 0), "run_id": run_id}


@app.function(image=IMAGE, volumes={"/data": volume})
def prepare_dataset_job(include_maze: bool = True,
                        include_sudoku: bool = False,
                        sudoku_output_dir: str = "data/sudoku-extreme-1k-aug-1000",
                        sudoku_subsample_size: int = 1000,
                        sudoku_num_aug: int = 1000):
    """Job: Prepare datasets on Modal (callable with .remote)."""
    return _do_prepare_dataset(
        include_maze=include_maze,
        include_sudoku=include_sudoku,
        sudoku_output_dir=sudoku_output_dir,
        sudoku_subsample_size=sudoku_subsample_size,
        sudoku_num_aug=sudoku_num_aug,
    )


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def prepare_dataset(include_maze: bool = True,
                    include_sudoku: bool = False,
                    sudoku_output_dir: str = "data/sudoku-extreme-1k-aug-1000",
                    sudoku_subsample_size: int = 1000,
                    sudoku_num_aug: int = 1000):
    """Webhook: Prepare datasets via HTTP; delegates to job function."""
    return _do_prepare_dataset(
        include_maze=include_maze,
        include_sudoku=include_sudoku,
        sudoku_output_dir=sudoku_output_dir,
        sudoku_subsample_size=sudoku_subsample_size,
        sudoku_num_aug=sudoku_num_aug,
    )


@app.function(image=IMAGE, timeout=3600, gpu="A100:2",volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def eval_test(checkpoint_path: str = "data/maze-30x30-hard-1k-weights/step_32550", dataset_path: str = "data/maze-30x30-hard-1k", out_dir: str = "out", batch_size: int = 64):
    """Run evaluation on the mounted test dataset and save predictions to persistent volume.

    This will raise on any failure (no fallback behavior) as requested.
    """
    repo_dir = _ensure_repo()

    os.chdir(repo_dir)

    local_out = os.path.join(repo_dir, out_dir)
    os.makedirs(local_out, exist_ok=True)
    
    # Build command to run evaluation with torchrun on both GPUs
    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node=2", eval_script,
        "--checkpoint", os.path.join(repo_dir, checkpoint_path),
        "--dataset", os.path.join(repo_dir, dataset_path),
        "--outdir", local_out,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    print(f"Running evaluation command: {' '.join(cmd)}")
    # Run and raise on failure
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    return {"status": "Evaluation completed", "output_dir": local_out, "result": result}

@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def download_weights():
    """Download pre-trained weights from Hugging Face."""
    
    # Use persistent data directory
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    repo_dir = "/data/repo"
    
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download weights
    checkpoint_path = hf_hub_download(
        repo_id="YuvrajSingh9886/maze-hard-trm",
        filename="step_32550",
        local_dir=data_dir
    )
    return {"status": "Weights downloaded", "path": checkpoint_path}


# @app.function(image=IMAGE, gpu="T4", volumes={"/data": volume}, timeout=3600)
@modal.fastapi_endpoint(docs=True)
def run_eval_local(checkpoint_path: str="data/maze-30x30-hard-1k", dataset_path: str = "maze-30x30-hard-1k-weights/step_32550", out_dir: str = "out", batch_size: int = 64):
    """Run evaluation locally on mounted repo data.

    Args:
      checkpoint_path: path to checkpoint file in repo
      dataset_path: path to dataset directory in repo
      out_dir: output directory

    Returns:
      dict with results
    """
    
    # Use persistent data directory
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    repo_dir = "/data/repo"
    
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    # Change to repo directory
    os.chdir(repo_dir)
 
    # Set output dir
    local_out = os.path.join(repo_dir, out_dir)
    os.makedirs(local_out, exist_ok=True)
    # shutil.move("scripts/run_eval_only.py", "./")
    # subprocess.run(["mv", "scripts/run_eval_only.py", "../"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # subprocess.run(["ls", "data", "maze-30x30-hard-1k"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # Run evaluation using subprocess with torchrun for multi-GPU
    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node=2", eval_script,
        "--checkpoint", os.path.join(repo_dir, checkpoint_path),
        "--dataset", os.path.join(repo_dir, dataset_path),
        "--outdir", local_out,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    print("Evaluation completed.")

    metrics = {}  # No metrics returned from subprocess

    # Return results
    return {'message': 'Evaluation completed', 'output_dir': out_dir}


def _do_predict(grid: object | None = None, index: int | None = None, file: str | None = None, task: str | None = None, model: str | None = None, run: str | None = None):
    """Core predict logic shared by webhook and job/CLI."""
    # As requested: use test-set predictions from evaluation outputs on the persistent volume.
    repo_dir = "/data/repo"
    # Choose output directory depending on optional task/model/run
    base_out = os.path.join(repo_dir, "out")
    safe_task = _safe_name(task or "maze")
    if safe_task not in ("maze", "sudoku"):
        raise HTTPException(status_code=400, detail="task must be 'maze' or 'sudoku'")
    # Maze has a single model; enforce
    if safe_task == "maze":
        if model not in (None, "", "default"):
            raise HTTPException(status_code=400, detail="maze has a single model; omit 'model' or use model=default")
        safe_model = "default"
    else:
        safe_model = _safe_name(model or "mlp")
        if safe_model not in ("mlp", "attn"):
            raise HTTPException(status_code=400, detail="For sudoku, model must be 'mlp' or 'attn'")
    task_dir = os.path.join(base_out, safe_task, safe_model)
    # Resolve run directory
    if run:
        safe_run = _safe_name(run)
        out_dir = os.path.join(task_dir, safe_run)
    else:
        guess = _pick_latest_run(task_dir)
        if guess is None:
            raise HTTPException(status_code=404, detail=f"No runs found in {task_dir}. Run evaluation first.")
        out_dir = guess

    if not os.path.exists(repo_dir):
        raise RuntimeError(f"Repo not found at {repo_dir}; expected it to be cloned into the persistent volume")

    # Determine which preds file to use (strict, no silent fallbacks)
    if not os.path.isdir(out_dir):
        raise HTTPException(status_code=404, detail=f"Output directory not found: {out_dir}")

    preds_path = None
    if file is not None:
        # Strict: only allow files within out_dir
        candidate = os.path.join(out_dir, file)
        candidate = os.path.realpath(candidate)
        if not candidate.startswith(os.path.realpath(out_dir) + os.sep):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not os.path.exists(candidate):
            raise HTTPException(status_code=404, detail=f"Specified preds file not found: {file}")
        preds_path = candidate
    else:
        # Choose deterministically: file with largest step number in name step_<N>_all_preds.*
        import re
        best = None
        best_step = -1
        for fname in os.listdir(out_dir):
            m = re.match(r"step_(\d+)_all_preds\.[0-9]+$", fname)
            if m:
                step = int(m.group(1))
                if step > best_step:
                    best_step = step
                    best = fname
        if best is None:
            raise HTTPException(status_code=404, detail=f"No preds file matching 'step_<N>_all_preds.<rank>' found in {out_dir}. Run eval.")
        preds_path = os.path.join(out_dir, best)

    # Load predictions saved by evaluate (torch.save of a dict of tensors)
    import torch
    data = torch.load(preds_path, map_location='cpu')

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"Loaded predictions file {preds_path} is not a dict of tensors")
    if 'preds' not in data:
        raise HTTPException(status_code=400, detail=f"Loaded predictions file {preds_path} does not contain key 'preds'")

    preds = data['preds']  # Expect shape (N, seq_len[, num_classes]) or (N, H, W)

    # If caller provided an explicit grid (GET query or POST body), try to use it.
    provided_grid = None
    if grid is not None:
        # grid may be a JSON string (from query) or a Python list (from body)
        import json
        try:
            if isinstance(grid, str):
                provided_grid = json.loads(grid)
            else:
                provided_grid = grid  # assume list-like
        except Exception:
            return {"error": "Failed to parse provided grid; send JSON list in POST body or JSON string in 'grid' query param."}

    import numpy as np
    ret_index = None
    if provided_grid is not None:
        grid_arr = np.array(provided_grid)
    else:
        # Strict index selection
        try:
            total = len(preds)
        except Exception:
            raise HTTPException(status_code=400, detail="'preds' tensor is not indexable")
        idx = 0 if index is None else int(index)
        if idx < 0 or idx >= total:
            raise HTTPException(status_code=400, detail=f"Index {idx} out of range (0..{total-1})")
        sel = preds[idx]
        ret_index = idx
        # Reduce to 2D int grid if possible; handle logits with argmax on last dim.
        try:
            if getattr(sel, 'ndim', 0) >= 2 and sel.shape[-1] > 8:  # heuristic: last dim likely num_classes
                arr = sel.argmax(dim=-1)
            else:
                arr = sel
            grid_arr = arr.detach().cpu().numpy() if hasattr(arr, 'detach') else np.array(arr)
        except Exception:
            grid_arr = np.array(sel)

    # Normalize predicted grid to square 2D array strictly
    import math
    if getattr(grid_arr, 'ndim', 0) == 2:
        h, w = int(grid_arr.shape[0]), int(grid_arr.shape[1])
        if h != w:
            raise HTTPException(status_code=400, detail=f"Predicted grid is not square: {h}x{w}")
        solved_maze = np.asarray(grid_arr).tolist()
        side = h
    else:
        L = grid_arr.shape[-1] if getattr(grid_arr, 'ndim', 0) > 0 else len(grid_arr)
        side_f = math.sqrt(L)
        if not float(side_f).is_integer():
            raise HTTPException(status_code=400, detail=f"Predicted grid length {L} is not a perfect square")
        side = int(side_f)
        solved_maze = np.asarray(grid_arr).reshape((side, side)).tolist()

    # Prepare input_maze: if user provided a grid, echo it; otherwise load corresponding inputs shard
    input_maze = None
    if provided_grid is not None:
        # Validate provided input grid strictly
        in_arr = np.array(provided_grid)
        if getattr(in_arr, 'ndim', 0) == 2:
            h, w = int(in_arr.shape[0]), int(in_arr.shape[1])
            if h != w:
                raise HTTPException(status_code=400, detail=f"Provided input grid is not square: {h}x{w}")
            input_maze = in_arr.tolist()
        else:
            L_in = in_arr.shape[-1] if getattr(in_arr, 'ndim', 0) > 0 else len(in_arr)
            side_f = math.sqrt(L_in)
            if not float(side_f).is_integer():
                raise HTTPException(status_code=400, detail=f"Provided input grid length {L_in} is not a perfect square")
            side_in = int(side_f)
            input_maze = np.asarray(in_arr).reshape((side_in, side_in)).tolist()
    else:
        # Load inputs directly from the same saved dict (evaluate saved 'inputs' alongside 'preds')
        inputs_tensor = None
        for k in ("inputs", "all_inputs", "all__inputs", "input", "x", "xs", "grids"):
            if k in data:
                inputs_tensor = data[k]
                break
        if inputs_tensor is None:
            raise HTTPException(status_code=400, detail=(
                f"Inputs not found in {os.path.basename(preds_path)}. "
                "Re-run evaluation with eval_save_outputs including 'inputs'."
            ))

        if ret_index is None:
            raise HTTPException(status_code=500, detail="ret_index not set while selecting input grid")

        try:
            sel_in = inputs_tensor[ret_index]
        except Exception:
            raise HTTPException(status_code=400, detail="Failed to index into saved inputs tensor")

        try:
            in_arr = sel_in.detach().cpu().numpy() if hasattr(sel_in, 'detach') else np.array(sel_in)
        except Exception:
            in_arr = np.array(sel_in)

        if getattr(in_arr, 'ndim', 0) == 2:
            h, w = int(in_arr.shape[0]), int(in_arr.shape[1])
            if h != w:
                raise HTTPException(status_code=400, detail=f"Saved input grid is not square: {h}x{w}")
            input_maze = np.asarray(in_arr).tolist()
        else:
            L_in = in_arr.shape[-1] if getattr(in_arr, 'ndim', 0) > 0 else (in_arr.size if hasattr(in_arr, 'size') else len(in_arr))
            side_f = math.sqrt(L_in)
            if not float(side_f).is_integer():
                raise HTTPException(status_code=400, detail=f"Saved input grid length {L_in} is not a perfect square")
            side_in = int(side_f)
            input_maze = np.asarray(in_arr).reshape((side_in, side_in)).tolist()

    return {"solved_maze": solved_maze, "input_maze": input_maze, "source_file": preds_path, "index": ret_index, "task": safe_task, "model": safe_model, "run_dir": out_dir}


@app.function(image = IMAGE, volumes= {"/data": volume})
@modal.fastapi_endpoint(docs=True)
def predict(grid: object = None, index: int | None = None, file: str | None = None, task: str | None = None, model: str | None = None, run: str | None = None):
    """Webhook: Predict solved grid from inputs or saved eval outputs."""
    return _do_predict(grid=grid, index=index, file=file, task=task, model=model, run=run)


@app.function(image=IMAGE, volumes={"/data": volume})
def predict_job(grid: object = None, index: int | None = None, file: str | None = None, task: str | None = None, model: str | None = None, run: str | None = None):
    """Job: Predict via remote function for use from CLI entrypoint."""
    return _do_predict(grid=grid, index=index, file=file, task=task, model=model, run=run)


@app.function(image=IMAGE, volumes={"/data": volume})
def download_sudoku_weights_job(model: str = "mlp"):
    """Job: Download Sudoku checkpoints from Hugging Face. model: 'mlp' or 'attn'."""
    repo_dir = "/data/repo"
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git", repo_dir], check=True)
    os.chdir(repo_dir)

    model = (model or "mlp").lower()
    if model not in ("mlp", "attn"):
        raise HTTPException(status_code=400, detail="model must be 'mlp' or 'attn'")

    from huggingface_hub import hf_hub_download
    save_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights")
    os.makedirs(save_dir, exist_ok=True)

    if model == "mlp":
        filename = "step_32550_sudoku_epoch50k"
    else:
        filename = "step_39060_sudoku_60k_epoch_attn_type"

    ckpt_path = hf_hub_download(
        repo_id="YuvrajSingh9886/sudoku-extreme-trm",
        filename=filename,
        local_dir=save_dir,
    )
    return {"status": "ok", "model": model, "checkpoint": ckpt_path}


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def download_sudoku_weights(model: str = "mlp"):
    """Webhook: delegates to download_sudoku_weights_job."""
    return _do_download_sudoku_weights(model=model)


@app.function(image=IMAGE, volumes={"/data": volume}, gpu="A100:2", timeout=3600)
def run_eval_sudoku_job(model: str = "mlp", dataset_path: str | None = None, batch_size: int = 64):
    """Job: Run evaluation for Sudoku using selected model. Writes outputs to out/sudoku/<model>."""
    repo_dir = "/data/repo"
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git", repo_dir], check=True)
    os.chdir(repo_dir)

    # Ensure weights exist
    resp = _do_download_sudoku_weights(model)  # type: ignore
    ckpt_path = resp["checkpoint"]

    # Dataset path: prefer 1k-aug-1000 if present, else fallback to full, unless overridden
    if dataset_path:
        dataset_dir = dataset_path
    else:
        dataset_dir = os.path.join(repo_dir, "data", "sudoku-extreme-1k-aug-1000")
        if not os.path.isdir(dataset_dir):
            dataset_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full")
    if not os.path.isdir(dataset_dir):
        print("WARNING: Sudoku dataset folder not found at", dataset_dir)

    # Output in per-run directory: out/sudoku/<model>/<run_id>
    parent = os.path.join(repo_dir, "out", "sudoku", (model or "mlp").lower())
    os.makedirs(parent, exist_ok=True)
    import time
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Delegate to internal helper that handles MLP arch override and subset flags
    return _do_run_eval_sudoku(model=model, dataset_path=dataset_path, batch_size=batch_size)


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def run_eval_sudoku(model: str = "mlp", dataset_path: str | None = None, batch_size: int = 64):
    """Webhook: delegates to run_eval_sudoku_job."""
    return _do_run_eval_sudoku(model=model, dataset_path=dataset_path, batch_size=batch_size)


@app.function(image=IMAGE, volumes={"/data": volume})
def download_all_weights_job():
    """Job: Download all required weights: maze default, sudoku mlp and attn."""
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)
    results = {}
    # Maze weights
    maze_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights")
    os.makedirs(maze_dir, exist_ok=True)
    maze_ckpt = hf_hub_download(repo_id="YuvrajSingh9886/maze-hard-trm", filename="step_32550", local_dir=maze_dir)
    results["maze"] = maze_ckpt
    # Sudoku
    results["sudoku_mlp"] = _do_download_sudoku_weights("mlp")["checkpoint"]  # type: ignore
    results["sudoku_attn"] = _do_download_sudoku_weights("attn")["checkpoint"]  # type: ignore
    return {"status": "ok", "paths": results}


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def download_all_weights():
    """Webhook: delegates to download_all_weights_job."""
    return _do_download_all_weights()


@app.function(image=IMAGE, volumes={"/data": volume}, gpu="A100:2", timeout=3600)
def run_eval_maze_job(batch_size: int = 64):
    """Job: Run evaluation for Maze (single model). Writes outputs to out/maze/default/<run_id>."""
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)

    # Ensure weights
    _do_download_all_weights()
    ckpt_path = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights", "step_32550")
    dataset_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k")
    if not os.path.isdir(dataset_dir):
        print("WARNING: Maze dataset folder not found at", dataset_dir)

    parent = os.path.join(repo_dir, "out", "maze", "default")
    os.makedirs(parent, exist_ok=True)
    import time
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)
    # subprocess.run(["ls"], stdout=sys.stdout, stderr=sys.stderr, check=False)
    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node=2", eval_script,
        "--checkpoint", ckpt_path,
        "--dataset", dataset_dir,
        "--outdir", out_dir,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    print("Running Maze eval:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    _symlink_latest(parent, out_dir)
    return {"status": "Evaluation completed", "output_dir": out_dir, "result": getattr(result, 'returncode', 0), "run_id": run_id}


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def run_eval_maze(batch_size: int = 64):
    """Webhook: delegates to run_eval_maze_job."""
    return _do_run_eval_maze(batch_size=batch_size)


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def list_runs(task: str = "maze", model: str | None = None):
    """List available runs under out/<task>/<model>. For maze, model is always 'default'."""
    repo_dir = _ensure_repo()
    base_out = os.path.join(repo_dir, "out")
    safe_task = _safe_name(task)
    if safe_task not in ("maze", "sudoku"):
        raise HTTPException(status_code=400, detail="task must be 'maze' or 'sudoku'")
    if safe_task == "maze":
        safe_model = "default"
    else:
        safe_model = _safe_name(model or "mlp")
        if safe_model not in ("mlp", "attn"):
            raise HTTPException(status_code=400, detail="For sudoku, model must be 'mlp' or 'attn'")
    parent = os.path.join(base_out, safe_task, safe_model)
    if not os.path.isdir(parent):
        return {"runs": []}
    entries = []
    for d in os.listdir(parent):
        p = os.path.join(parent, d)
        if os.path.isdir(p) and d != "latest":
            entries.append({"name": d, "mtime": os.path.getmtime(p)})
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    latest = None
    if os.path.islink(os.path.join(parent, "latest")):
        latest = os.readlink(os.path.join(parent, "latest"))
    return {"task": safe_task, "model": safe_model, "parent": parent, "latest": latest, "runs": entries}


@app.function(volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def get_maze_visualizer():
    """Serve the simple maze-only visualizer HTML."""
    repo_dir = "/data/repo"
    os.chdir(repo_dir)
    html_path = os.path.join(repo_dir, "maze_visualizer.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"Maze visualizer HTML not found at {html_path}; ensure repo is cloned")

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return Response(content, media_type="text/html")


@app.function(volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def get_sudoku_visualizer():
    """Serve the Sudoku visualizer HTML."""
    repo_dir = "/data/repo"
    os.chdir(repo_dir)
    html_path = os.path.join(repo_dir, "sudoku_visualizer.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"Sudoku visualizer HTML not found at {html_path}; ensure repo is cloned")

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return Response(content, media_type="text/html")


@app.function(volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def get_unified_visualizer():
    """Serve the unified visualizer HTML (task/model/run switcher)."""
    repo_dir = "/data/repo"
    os.chdir(repo_dir)
    html_path = os.path.join(repo_dir, "unified_visualizer.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"Unified visualizer HTML not found at {html_path}; ensure repo is cloned")

    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return Response(content, media_type="text/html")


def _serve_asset_from_repo(filename: str) -> Response:
    """Internal helper to serve asset files from the repo."""
    repo_dir = _ensure_repo()
    asset_path = os.path.join(repo_dir, "assets", filename)
    if os.path.exists(asset_path):
        with open(asset_path, "rb") as f:
            content = f.read()
        return Response(content, media_type="application/javascript" if filename.endswith(".js") else "text/plain")
    print(f"get_asset: asset not found at {asset_path}")
    return Response("File not found", status_code=404, media_type="text/plain")


@modal.fastapi_endpoint(docs=True)
def get_asset(filename: str):
    """Serve asset files from the repo."""
    return _serve_asset_from_repo(filename)


@modal.fastapi_endpoint(docs=True)
def assets(filename: str):
    """Compatibility wrapper: serve files at /assets/{filename} by delegating to get_asset.

    Some HTML in the repo requests /assets/npyjs.js directly. This wrapper ensures
    that path resolves to the same file-serving logic without changing other code.
    """
    # Delegate to internal helper to avoid calling the decorated endpoint directly
    return _serve_asset_from_repo(filename)


# ------------------------------
# Local entrypoints (for `modal run`)
# ------------------------------

@app.local_entrypoint()
def cli_prepare_dataset(include_maze: bool = True,
                        include_sudoku: bool = False,
                        sudoku_output_dir: str = "data/sudoku-extreme-1k-aug-1000",
                        sudoku_subsample_size: int = 1000,
                        sudoku_num_aug: int = 1000):
    """Local entrypoint: run dataset preparation remotely.
    Example:
      modal run infra/modal_app.py::cli_prepare_dataset --include-sudoku=true
    """
    res = prepare_dataset_job.remote(
        include_maze=include_maze,
        include_sudoku=include_sudoku,
        sudoku_output_dir=sudoku_output_dir,
        sudoku_subsample_size=sudoku_subsample_size,
        sudoku_num_aug=sudoku_num_aug,
    )
    print(res)


@app.local_entrypoint()
def cli_run_eval_maze(batch_size: int = 64):
    """Local entrypoint: trigger Maze evaluation."""
    res = run_eval_maze_job.remote(batch_size=batch_size)  # type: ignore
    print(res)


@app.local_entrypoint()
def cli_run_eval_sudoku(model: str = "mlp", dataset_path: str | None = None, batch_size: int = 64):
    """Local entrypoint: trigger Sudoku evaluation.
    Example:
      modal run infra/modal_app.py::cli_run_eval_sudoku --model=attn
    """
    res = run_eval_sudoku_job.remote(model=model, dataset_path=dataset_path, batch_size=batch_size)  # type: ignore
    print(res)


@app.local_entrypoint()
def cli_download_all_weights():
    """Local entrypoint: download all weights remotely."""
    res = download_all_weights_job.remote()  # type: ignore
    print(res)


@app.local_entrypoint()
def cli_predict(task: str = "maze",
                model: str | None = None,
                run: str | None = None,
                file: str | None = None,
                index: int | None = None,
                grid_json: str | None = None,
                grid_file: str | None = None):
    """Local entrypoint: call predict remotely with flags.
    Examples:
      # Maze latest run, first example
      modal run infra/modal_app.py::cli_predict --task=maze --index=0

      # Sudoku (MLP), latest run, 10th example
      modal run infra/modal_app.py::cli_predict --task=sudoku --model=mlp --index=9

      # Pin run and file
      modal run infra/modal_app.py::cli_predict --task=maze --run=20251015-202010 --file=step_32550_all_preds.0 --index=0

      # Provide grid via JSON string
      modal run infra/modal_app.py::cli_predict --task=sudoku --model=attn --grid-json='[[0,1,2],[3,4,5],[6,7,8]]'

      # Provide grid via JSON file path
      modal run infra/modal_app.py::cli_predict --task=maze --grid-file=/path/to/grid.json
    """
    grid = None
    import json as _json
    if grid_json:
        grid = _json.loads(grid_json)
    elif grid_file:
        with open(grid_file, "r", encoding="utf-8") as f:
            grid = _json.load(f)

    res = predict_job.remote(grid=grid, index=index, file=file, task=task, model=model, run=run)  # type: ignore
    print(res)