"""Modal app for maze solving evaluation and prediction.

Usage:
  modal run infra/modal_app.py:run_eval_local --checkpoint-path data/maze-30x30-hard-1k/step_50000 --dataset-path data/maze-30x30-hard-1k/test

Notes:
- This script runs evaluation on mounted repo data using multi-GPU torchrun.
- Provides web endpoints for visualization and prediction.
"""

import os
import sys
import subprocess
import shutil
import threading
import json as _json
import time
import uuid
import math
import re

import numpy as np
import torch
import yaml
from typing import Any, Dict, cast
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
import modal
from fastapi import Response, HTTPException, Body, Query, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Create a persistent volume for repo and data
volume = modal.Volume.from_name("tinyrecursive-data", create_if_missing=True)

NO_GPU=1

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

fastapi_app = None
try:
    # Create a small FastAPI app only to attach middleware; Modal will mount endpoints into its own FastAPI instance.
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

# Simple in-memory cache for loaded prediction files to speed up repeated predict calls
_PRED_CACHE_LOCK = threading.Lock()
_PRED_CACHE: dict[str, dict] = {}
_PRED_CACHE_META: dict[str, float] = {}  # store mtime to invalidate when file changes
_PRED_CACHE_MAX = 8  # keep a handful of recent files

def _load_preds_cached(preds_path: str):
    """Load a torch-saved predictions file with a tiny mtime-based cache.

    The cache keeps a few recent files keyed by path and invalidates entries when the
    file's modification time changes. Returns the loaded object (typically a dict of tensors).
    """
    mtime = os.path.getmtime(preds_path)
    with _PRED_CACHE_LOCK:
        cached = _PRED_CACHE.get(preds_path)
        if cached is not None and _PRED_CACHE_META.get(preds_path) == mtime:
            return cached
    # Load outside lock (IO)
    data = torch.load(preds_path, map_location='cpu')
    with _PRED_CACHE_LOCK:
        # Evict if needed
        if len(_PRED_CACHE) >= _PRED_CACHE_MAX:
            # drop the oldest by mtime
            oldest = sorted(_PRED_CACHE_META.items(), key=lambda kv: kv[1])[0][0]
            _PRED_CACHE.pop(oldest, None)
            _PRED_CACHE_META.pop(oldest, None)
        _PRED_CACHE[preds_path] = data
        _PRED_CACHE_META[preds_path] = mtime
    return data

# ------------------------------
# Realtime model inference (cached models)
# ------------------------------
_RT_MODELS_LOCK = threading.Lock()
_RT_MODELS: dict[tuple[str,str], dict] = {}

def _load_checkpoint_compat(model, ckpt_path: str):
    """Load checkpoint into ACT-wrapped model with key normalization.
    Mirrors logic from pretrain.load_checkpoint without requiring PretrainConfig type.
    """
    import torch
    sd = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess keys: strip compile/DataParallel-style prefixes so they match the target module
    def normalize_key(key: str) -> str:
        # remove any leading '.' that can appear in some saved dicts
        if key.startswith('.'):
            key = key[1:]
        # common prefixes to strip
        for prefix in ("_orig_mod.", "_orig._mod.", "module."):
            if key.startswith(prefix):
                return key[len(prefix):]
        return key

    def preprocess_state_dict_keys(state_dict: dict) -> dict:
        new_sd: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            nk = normalize_key(k) if isinstance(k, str) else k
            new_sd[nk] = v
        return new_sd

    sd = preprocess_state_dict_keys(sd)

    # # Resize puzzle embedding if shape mismatch (handle both presence/absence of prefix in ckpt)
    # try:
    #     expected = model.model.inner.puzzle_emb.weights.shape  # type: ignore[attr-defined]
    #     candidates = [
    #         "model.inner.puzzle_emb.weights",
    #         "_orig_mod.model.inner.puzzle_emb.weights",
    #         "module.model.inner.puzzle_emb.weights",
    #     ]
    #     name_in_sd = next((n for n in candidates if n in sd), None)
    #     if name_in_sd is not None and getattr(sd[name_in_sd], 'shape', None) != expected:
    #         pe = sd[name_in_sd]
    #         sd[name_in_sd] = (pe.mean(dim=0, keepdim=True).expand(expected).contiguous())
    # except Exception:
    #     pass

    # Load non-strict to tolerate benign missing buffers/keys after normalization
    # try:
    incompat = model.load_state_dict(sd, strict=True)
    # except TypeError:
        # Older torch without strict kw?
        # incompat = model.load_state_dict(sd)  # type: ignore[assignment]
    # Best-effort visibility for debugging: log key mismatches (does not raise)
    try:
        missing = getattr(incompat, "missing_keys", [])
        unexpected = getattr(incompat, "unexpected_keys", [])
        if missing or unexpected:
            print(f"load_state_dict: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
            if len(missing) <= 8:
                print("  missing:", missing)
            if len(unexpected) <= 8:
                print("  unexpected:", unexpected)
    except Exception:
        pass

def _load_dataset_metadata(repo_dir: str, task: str) -> dict:
    """Return dataset metadata for the given task by reading the first existing dataset.json.

    Looks under standard paths for Maze and Sudoku test sets inside the mounted repo.
    Raises RuntimeError if none are found.
    """
    if task == "maze":
        meta_paths = [
            os.path.join(repo_dir, "data", "maze-30x30-hard-1k", "test", "dataset.json"),
        ]
    else:
        meta_paths = [
            os.path.join(repo_dir, "data", "sudoku-extreme-1k-aug-1000", "test", "dataset.json"),
            os.path.join(repo_dir, "data", "sudoku-extreme-full", "test", "dataset.json"),
        ]
    for p in meta_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return _json.load(f)
    raise RuntimeError(f"Dataset metadata not found for task={task} under known paths: {meta_paths}")

def _get_realtime_model(task: str, model: str | None):
    """Return cached ACT-wrapped model and metadata for realtime inference.

    Ensures the repo is present and importable, resolves the architecture config, constructs
    the TinyRecursiveReasoningModel with ACT head, moves it to device, and loads weights.
    """
    repo_dir = _ensure_repo()
    # Ensure local repo's Python packages (e.g., 'models') are importable
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    safe_task = (task or "maze").lower()
    safe_model = (model or ("default" if safe_task=="maze" else "mlp")).lower()
    if safe_task == "maze":
        safe_model = "default"
    key = (safe_task, safe_model)

    with _RT_MODELS_LOCK:
        if key in _RT_MODELS:
            return _RT_MODELS[key]

    # Build new model
    os.environ.setdefault("DISABLE_COMPILE", "1")
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    from models.losses import ACTLossHead

    # Arch config: load via OmegaConf to resolve ${...} interpolations into concrete values
    arch_path = os.path.join(repo_dir, "config", "arch", "trm.yaml")
    from omegaconf import OmegaConf
    from typing import Any, Dict, cast
    oc = OmegaConf.load(arch_path)
    if safe_task == "sudoku" and safe_model == "mlp":
        # Toggle MLP variant for sudoku when requested
        oc.mlp_t = True
    arch_container = OmegaConf.to_container(oc, resolve=True)
    if not isinstance(arch_container, dict):
        raise RuntimeError("Architecture config did not resolve to a dict")
    arch_cfg: Dict[str, Any] = cast(Dict[str, Any], arch_container)

    # Dataset metadata
    meta = _load_dataset_metadata(repo_dir, safe_task)
    seq_len = int(meta["seq_len"])  # 900 (maze) or 81 (sudoku)
    vocab_size = int(meta["vocab_size"])  # 5 (maze) or 11 (sudoku)
    num_ids = int(meta["num_puzzle_identifiers"])  # typically 1

    # Build model config dict expected by TinyRecursiveReasoningModel_ACTV1
    model_cfg: Dict[str, Any] = {**arch_cfg,
                                 "batch_size": 1,
                                 "seq_len": seq_len,
                                 "vocab_size": vocab_size,
                                 "num_puzzle_identifiers": num_ids}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Instantiate and wrap with loss head (to reuse ACT loop semantics)
    core = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    loss_cfg = arch_cfg.get("loss") if isinstance(arch_cfg.get("loss"), dict) else None
    loss_type = (loss_cfg.get("loss_type") if isinstance(loss_cfg, dict) else None) or "softmax_cross_entropy"
    act = ACTLossHead(core, loss_type=loss_type)
    act.eval()
    act.to(device)

    # Load weights from pre-downloaded checkpoints
    if safe_task == "maze":
        ckpt_path = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights", "step_32550")
    else:
        ckpt_path = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights",
                                 "step_32550_sudoku_epoch50k" if safe_model=="mlp" else "step_39060_sudoku_60k_epoch_attn_type")

    # Build a minimal object with the attribute expected by load_checkpoint
    _load_checkpoint_compat(act, ckpt_path)

    entry = {"model": act, "meta": meta, "device": device, "task": safe_task, "model_name": safe_model}
    with _RT_MODELS_LOCK:
        _RT_MODELS[key] = entry
    return entry

def _format_grid_for_task(task: str, meta: dict, grid: list) -> list:
    """Normalize inbound UI grid to the model token space as a flat list of ints per task."""
    arr = np.array(grid)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        arr = arr.reshape(side, side)
    if task == "sudoku":
        # Map 0..9 -> 1..10 to match dataset tokens (pad is 0)
        arr = arr.clip(0, 9) + 1
    # Maze assumed already in token space (best effort)
    return arr.astype(int).tolist()

def _postprocess_preds_for_task(task: str, meta: dict, pred_tokens_1d: list[int]) -> list[list[int]]:
    """Map model token predictions back to UI grid for the given task (2D square int grid)."""
    arr = np.array(pred_tokens_1d)
    side = int(np.sqrt(arr.size))
    arr = arr.reshape(side, side)
    if task == "sudoku":
        # Map tokens 1..10 -> 0..9 for UI
        arr = (arr - 1).clip(0, 9)
    return arr.astype(int).tolist()

@app.function(image=IMAGE, volumes={"/data": volume}, gpu="A100:2")
@modal.fastapi_endpoint(docs=True, method="POST")
def predict_realtime(
    grid: object | None = Body(default=None),
    task: str | None = Body(default=None),
    model: str | None = Body(default=None),
    grid_q: str | None = Query(default=None, alias="grid"),
    task_q: str | None = Query(default=None, alias="task"),
    model_q: str | None = Query(default=None, alias="model"),
):
    """Realtime prediction.

    Accepts task/model/grid via JSON body or query parameters and returns a JSON
    payload with normalized input, solved grid, and ACT inference steps.
    - task: "maze" or "sudoku" (default: maze)
    - model: for sudoku: "mlp" or "attn"; ignored for maze
    - grid: 2D square list or flat list of ints (0 = empty for sudoku)
    """
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    try:
        eff_task = task if task is not None else task_q
        eff_model = model if model is not None else model_q
        eff_task = (eff_task or "maze").lower()
        eff_model = (eff_model or ("default" if eff_task=="maze" else "mlp")).lower()
        if eff_task not in ("maze", "sudoku"):
            raise HTTPException(status_code=400, detail="task must be 'maze' or 'sudoku'")
        if eff_task == "maze":
            eff_model = "default"
        else:
            if eff_model not in ("mlp", "attn"):
                raise HTTPException(status_code=400, detail="For sudoku, model must be 'mlp' or 'attn'")

        # Parse grid
        g = grid if grid is not None else grid_q
        if isinstance(g, str):
            g = _json.loads(g)
        if g is None:
            raise HTTPException(status_code=400, detail="grid is required for realtime prediction")
        if isinstance(g, tuple):
            g = list(g)
        if not isinstance(g, list):
            raise HTTPException(status_code=400, detail="grid must be a list or nested list")

        rt = _get_realtime_model(eff_task, eff_model)
        act = rt["model"]
        meta = rt["meta"]
        device = rt["device"]

        # Prepare batch tensors
        inp2d = _format_grid_for_task(eff_task, meta, g)
        flat = np.array(inp2d).reshape(-1)
        inputs = torch.from_numpy(flat.astype("int32")).unsqueeze(0).to(device)
        puzzle_identifiers = torch.zeros((1,), dtype=torch.int32, device=device)
        labels = inputs.clone()  # dummy labels; unused for preds
        batch = {"inputs": inputs, "labels": labels, "puzzle_identifiers": puzzle_identifiers}

        # Run ACT loop until halting
        with torch.inference_mode():
            carry = act.initial_carry(batch)
            # Ensure carry tensors are on the same device as the model/batch
            try:
                ic = getattr(carry, "inner_carry", None)
                if ic is not None:
                    if hasattr(ic, "z_H"):
                        ic.z_H = ic.z_H.to(device)
                    if hasattr(ic, "z_L"):
                        ic.z_L = ic.z_L.to(device)
                if hasattr(carry, "steps"):
                    carry.steps = carry.steps.to(device)
                if hasattr(carry, "halted"):
                    carry.halted = carry.halted.to(device)
                if hasattr(carry, "current_data") and isinstance(carry.current_data, dict):
                    for k, v in list(carry.current_data.items()):
                        try:
                            carry.current_data[k] = v.to(device)
                        except Exception:
                            pass
            except Exception:
                pass
            return_keys = {"preds", "logits"}
            steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = act(
                    return_keys=return_keys,
                    batch=batch,
                    carry=carry,
                )
                steps += 1
                if bool(all_finish):
                    break

        # Get predictions as 1D tokens
        pred_tokens = preds.get("preds")
        if pred_tokens is None:
            logits = preds.get("logits")
            pred_tokens = torch.argmax(logits, dim=-1)
        pred_tokens = pred_tokens.squeeze(0).detach().to("cpu").numpy().astype(int).tolist()
        solved = _postprocess_preds_for_task(eff_task, meta, pred_tokens)

        payload = {
            "task": eff_task,
            "model": eff_model,
            "input_maze": inp2d,
            "solved_maze": solved,
            "inference_steps": steps,
        }
        return JSONResponse(content=payload, headers=headers)
    except HTTPException as e:
        return JSONResponse(content={"detail": e.detail}, status_code=e.status_code, headers=headers)
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500, headers=headers)


# Helpers
def _safe_name(s: str) -> str:
    return ''.join(ch for ch in s if ch.isalnum() or ch in ('-', '_')).lower()


def _ensure_repo(repo_dir: str = "/data/repo") -> str:
    """Ensure the repo exists on the persistent volume. Clone if missing.

    Intentionally DOES NOT fetch/reset on every call to keep eval/predict fast.
    Use _bootstrap_repo_and_weights() (called from prepare_dataset) to update.
    """
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        # Ensure run_eval_only.py is at repo root on first clone
        try:
            src = os.path.join(repo_dir, "scripts", "run_eval_only.py")
            dst = os.path.join(repo_dir, "run_eval_only.py")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Synced eval runner to repo root: {dst}")
        except Exception as e:
            print("WARNING: Failed to place run_eval_only.py at repo root:", e)
    return repo_dir


def _bootstrap_repo_and_weights(repo_dir: str = "/data/repo") -> dict:
    """One-time, explicit update + download of all required weights.

    Called only by prepare_dataset to keep eval/predict fast at runtime.
    """
    repo_dir = _ensure_repo(repo_dir)
    # Update to latest main exactly once when requested
    try:
        print(f"Updating repo at {repo_dir}...")
        subprocess.run(["git", "-C", repo_dir, "fetch", "--all", "--prune"], check=True)
        subprocess.run(["git", "-C", repo_dir, "reset", "--hard", "origin/main"], check=True)
        subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], check=True)
    except Exception as e:
        print("WARNING: Repo update failed; proceeding with existing contents:", e)

    # Ensure eval runner at root (in case of updates)
    try:
        src = os.path.join(repo_dir, "scripts", "run_eval_only.py")
        dst = os.path.join(repo_dir, "run_eval_only.py")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Synced eval runner to repo root: {dst}")
    except Exception as e:
        print("WARNING: Failed to sync run_eval_only.py:", e)

    # Download all weights once
    results: dict[str, str] = {}
    # Maze weights
    maze_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights")
    os.makedirs(maze_dir, exist_ok=True)
    maze_ckpt_path = os.path.join(maze_dir, "step_32550")
    if not os.path.exists(maze_ckpt_path):
        results["maze"] = hf_hub_download(repo_id="YuvrajSingh9886/maze-hard-trm", filename="step_32550", local_dir=maze_dir)
    else:
        results["maze"] = maze_ckpt_path

    # Sudoku weights (mlp/attn)
    sud_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights")
    os.makedirs(sud_dir, exist_ok=True)
    mlp_file = os.path.join(sud_dir, "step_32550_sudoku_epoch50k")
    attn_file = os.path.join(sud_dir, "step_39060_sudoku_60k_epoch_attn_type")
    if not os.path.exists(mlp_file):
        results["sudoku_mlp"] = hf_hub_download(repo_id="YuvrajSingh9886/sudoku-extreme-trm", filename="step_32550_sudoku_epoch50k", local_dir=sud_dir)
    else:
        results["sudoku_mlp"] = mlp_file
    if not os.path.exists(attn_file):
        results["sudoku_attn"] = hf_hub_download(repo_id="YuvrajSingh9886/sudoku-extreme-trm", filename="step_39060_sudoku_60k_epoch_attn_type", local_dir=sud_dir)
    else:
        results["sudoku_attn"] = attn_file

    return {"status": "bootstrapped", "paths": results}


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
    # Prefer the source of truth under scripts/ to avoid stale copies at repo root
    candidates = [
        os.path.join(repo_dir, "scripts", "run_eval_only.py"),
        os.path.join(repo_dir, "run_eval_only.py"),
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
    # One-time repo sync + weights
    repo_dir = _bootstrap_repo_and_weights()
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




def _do_run_eval_sudoku(model: str,
                        dataset_path: str | None,
                        batch_size: int = 16,
                        one_batch: bool = False,
                        repeats: int = 1,
                        seed_start: int = 0):
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)
    # Use pre-downloaded weights (set by prepare step)
    ckpt_path = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights",
                             "step_32550_sudoku_epoch50k" if (model or "mlp").lower()=="mlp" else "step_39060_sudoku_60k_epoch_attn_type")

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
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Choose arch override for MLP vs attention by temporarily swapping arch file
    # Hydra.initialize requires config_path to be relative to the current working directory.
    # We chdir(repo_dir) above, so pass a relative path here.
    config_path = "config/cfg_pretrain.yaml"
    arch_dir = os.path.join(repo_dir, "config", "arch")
    trm_path = os.path.join(arch_dir, "trm.yaml")
    backup_path = os.path.join(arch_dir, "trm.yaml.bak")

    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node={}".format(NO_GPU), eval_script,
        "--config", config_path,
        "--checkpoint", ckpt_path,
        "--dataset", dataset_dir,
        "--outdir", out_dir,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    # Ensure repo root is on PYTHONPATH so imports like `from models...` resolve when script lives under scripts/
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_dir}:{env.get('PYTHONPATH','')}"
    if repeats and int(repeats) > 1:
        cmd.extend(["--repeats", str(int(repeats))])
        if seed_start:
            cmd.extend(["--seed-start", str(int(seed_start))])
    if one_batch:
        cmd.append("--one-batch")

    need_mlp = (model or "mlp").lower() == "mlp"
    restored = False
    try:
        if need_mlp:
            # Backup original trm.yaml and set mlp_t: True inline
            if os.path.exists(trm_path):
                shutil.copy2(trm_path, backup_path)
            # Parse YAML and set mlp_t True
            with open(trm_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Set flag
            data["mlp_t"] = True
            # Write back
            with open(trm_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
        print("Running Sudoku eval:", " ".join(cmd))
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True, cwd=repo_dir, env=env)
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


def _do_run_eval_maze(batch_size: int = 256,
                      one_batch: bool = False,
                      repeats: int = 1,
                      seed_start: int = 0):
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)
    # Use pre-downloaded weights
    ckpt_path = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights", "step_32550")
    dataset_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k")
    if not os.path.isdir(dataset_dir):
        print("WARNING: Maze dataset folder not found at", dataset_dir)

    parent = os.path.join(repo_dir, "out", "maze", "default")
    os.makedirs(parent, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node={}".format(NO_GPU), eval_script,
        "--checkpoint", ckpt_path,
        "--dataset", dataset_dir,
        "--outdir", out_dir,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    # Ensure repo root is on PYTHONPATH so imports like `from models...` resolve when script lives under scripts/
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_dir}:{env.get('PYTHONPATH','')}"
    if repeats and int(repeats) > 1:
        cmd.extend(["--repeats", str(int(repeats))])
        if seed_start:
            cmd.extend(["--seed-start", str(int(seed_start))])
    if one_batch:
        cmd.append("--one-batch")
    print("Running Maze eval:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True, cwd=repo_dir, env=env)
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
def eval_test(checkpoint_path: str = "data/maze-30x30-hard-1k-weights/step_32550", dataset_path: str = "data/maze-30x30-hard-1k", out_dir: str = "out", batch_size: int = 256):
    """Run evaluation on the mounted test dataset and save predictions to persistent volume.

    This will raise on any failure (no fallback behavior) as requested.
    """
    repo_dir = _ensure_repo()

    os.chdir(repo_dir)

    local_out = os.path.join(repo_dir, out_dir)
    os.makedirs(local_out, exist_ok=True)
    
    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node={}".format(NO_GPU), eval_script,
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


# @app.function(image=IMAGE, gpu="T4", volumes={"/data": volume}, timeout=3600)
@modal.fastapi_endpoint(docs=True)
def run_eval_local(checkpoint_path: str="data/maze-30x30-hard-1k", dataset_path: str = "maze-30x30-hard-1k-weights/step_32550", out_dir: str = "out", batch_size: int = 256):
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
    # Run evaluation using subprocess with torchrun for multi-GPU
    eval_script = _get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node={}".format(NO_GPU), eval_script,
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

    # Load predictions saved by evaluate (torch.save of a dict of tensors) with cache
    data = _load_preds_cached(preds_path)

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"Loaded predictions file {preds_path} is not a dict of tensors")
    if 'preds' not in data:
        raise HTTPException(status_code=400, detail=f"Loaded predictions file {preds_path} does not contain key 'preds'")

    preds = data['preds']  # Expect shape (N, seq_len[, num_classes]) or (N, H, W)

    # If caller provided an explicit grid (GET query or POST body), try to use it.
    provided_grid = None
    if grid is not None:
        # grid may be a JSON string (from query) or a Python list (from body)
        try:
            if isinstance(grid, str):
                provided_grid = _json.loads(grid)
            else:
                provided_grid = grid  # assume list-like
        except Exception:
            return {"error": "Failed to parse provided grid; send JSON list in POST body or JSON string in 'grid' query param."}

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
    target_maze = None
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

        # Load ground-truth labels from the same saved dict if available
        labels_tensor = None
        for k in ("labels", "all_labels", "all__labels", "label", "y", "ys", "solutions", "targets"):
            if k in data:
                labels_tensor = data[k]
                break
        if labels_tensor is not None:
            try:
                sel_lbl = labels_tensor[ret_index]
            except Exception:
                raise HTTPException(status_code=400, detail="Failed to index into saved labels tensor")

            try:
                lbl_arr = sel_lbl.detach().cpu().numpy() if hasattr(sel_lbl, 'detach') else np.array(sel_lbl)
            except Exception:
                lbl_arr = np.array(sel_lbl)

            if getattr(lbl_arr, 'ndim', 0) == 2:
                lh, lw = int(lbl_arr.shape[0]), int(lbl_arr.shape[1])
                if lh != lw:
                    raise HTTPException(status_code=400, detail=f"Saved label grid is not square: {lh}x{lw}")
                target_maze = np.asarray(lbl_arr).tolist()
            else:
                L_lbl = lbl_arr.shape[-1] if getattr(lbl_arr, 'ndim', 0) > 0 else (lbl_arr.size if hasattr(lbl_arr, 'size') else len(lbl_arr))
                side_f_lbl = math.sqrt(L_lbl)
                if not float(side_f_lbl).is_integer():
                    raise HTTPException(status_code=400, detail=f"Saved label grid length {L_lbl} is not a perfect square")
                side_lbl = int(side_f_lbl)
                target_maze = np.asarray(lbl_arr).reshape((side_lbl, side_lbl)).tolist()

    payload = {"solved_maze": solved_maze, "input_maze": input_maze, "source_file": preds_path, "index": ret_index, "task": safe_task, "model": safe_model, "run_dir": out_dir}
    # If caller POSTed a grid, persist the I/O→O/P triplet under out/<task>/<model>/custom/
    if provided_grid is not None:
        try:
            custom_dir = os.path.join(base_out, safe_task, safe_model, "custom")
            os.makedirs(custom_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            name = f"{ts}-{uuid.uuid4().hex[:8]}.json"
            save_path = os.path.join(custom_dir, name)
            to_save = {
                "task": safe_task,
                "model": safe_model,
                "input_maze": input_maze,
                "solved_maze": solved_maze,
                "target_maze": payload.get("target_maze"),
                "created_at": ts,
                "source": "custom-post",
            }
            with open(save_path, "w", encoding="utf-8") as f:
                _json.dump(to_save, f)
            payload["saved_file"] = name
            payload["saved_dir"] = custom_dir
        except Exception as _e:
            # Non-fatal if saving fails; continue returning prediction
            payload["saved_error"] = str(_e)
    if target_maze is not None:
        payload["target_maze"] = target_maze
    return payload


@app.function(image = IMAGE, volumes= {"/data": volume})
@modal.fastapi_endpoint(docs=True)
def predict(
    grid: object | None = Body(default=None),
    index: int | None = None,
    file: str | None = None,
    # Accept task/model/run from body OR query; merge below.
    task: str | None = Body(default=None),
    model: str | None = Body(default=None),
    run: str | None = Body(default=None),
    task_q: str | None = Query(default=None, alias="task"),
    model_q: str | None = Query(default=None, alias="model"),
    run_q: str | None = Query(default=None, alias="run"),
    grid_q: str | None = Query(default=None, alias="grid"),
):
    """Webhook: Predict solved grid from inputs or saved eval outputs."""
    # Always include permissive CORS headers so visualizers on a different Modal subdomain can fetch this endpoint.
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    try:
        # Merge body and query values; prefer body when provided
        eff_task = task if task is not None else task_q
        eff_model = model if model is not None else model_q
        eff_run = run if run is not None else run_q
        eff_grid = grid if grid is not None else grid_q
        payload = _do_predict(grid=eff_grid, index=index, file=file, task=eff_task, model=eff_model, run=eff_run)
        return JSONResponse(content=payload, headers=headers)
    except HTTPException as e:
        # Ensure CORS headers are present even on errors (e.g., 404, 400) so the browser surfaces the JSON.
        body = {"detail": e.detail}
        return JSONResponse(content=body, status_code=e.status_code, headers=headers)


@app.function(image=IMAGE, volumes={"/data": volume})
def predict_job(grid: object = None, index: int | None = None, file: str | None = None, task: str | None = None, model: str | None = None, run: str | None = None):
    """Job: Predict via remote function for use from CLI entrypoint."""
    return _do_predict(grid=grid, index=index, file=file, task=task, model=model, run=run)


 

@app.function(image=IMAGE, volumes={"/data": volume}, gpu="A100:2", timeout=3600)
def run_eval_sudoku_job(model: str = "mlp",
                        dataset_path: str | None = None,
                        batch_size: int = 256,
                        one_batch: bool = False,
                        repeats: int = 1,
                        seed_start: int = 0):
    """Job: Run evaluation for Sudoku using selected model. Writes outputs to out/sudoku/<model>."""
    repo_dir = "/data/repo"
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git", repo_dir], check=True)
    os.chdir(repo_dir)

    # Weights expected to be present (bootstrapped via prepare)

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

    # Delegate to internal helper that handles MLP arch override and flags
    return _do_run_eval_sudoku(model=model, dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, repeats=repeats, seed_start=seed_start)


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def run_eval_sudoku(model: str = "mlp",
                    dataset_path: str | None = None,
                    batch_size: int = 256,
                    one_batch: bool = False,
                    repeats: int = 1,
                    seed_start: int = 0):
    """Webhook: delegates to GPU-backed job function to ensure NCCL has GPUs."""
    return run_eval_sudoku_job.remote(model=model, dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, repeats=repeats, seed_start=seed_start)


 

@app.function(image=IMAGE, volumes={"/data": volume}, gpu="A100:2", timeout=3600)
def run_eval_maze_job(batch_size: int = 256,
                      one_batch: bool = False,
                      repeats: int = 1,
                      seed_start: int = 0):
    """Job: Run evaluation for Maze (single model). Writes outputs to out/maze/default/<run_id>."""
    # Delegate to internal helper that forwards repeats/seed_start and handles outputs/symlink
    return _do_run_eval_maze(batch_size=batch_size, one_batch=one_batch, repeats=repeats, seed_start=seed_start)


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def run_eval_maze(batch_size: int = 256,
                  one_batch: bool = False,
                  repeats: int = 1,
                  seed_start: int = 0):
    """Webhook: delegates to GPU-backed job function to ensure NCCL has GPUs."""
    return run_eval_maze_job.remote(batch_size=batch_size, one_batch=one_batch, repeats=repeats, seed_start=seed_start)


@app.function(volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def get_maze_visualizer():
    """Serve the simple maze-only visualizer HTML."""
    repo_dir = _ensure_repo()
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
    repo_dir = _ensure_repo()
    os.chdir(repo_dir)
    html_path = os.path.join(repo_dir, "sudoku_visualizer.html")

    if not os.path.exists(html_path):
        raise FileNotFoundError(f"Sudoku visualizer HTML not found at {html_path}; ensure repo is cloned")

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
def cli_run_eval_maze(batch_size: int = 256,
                      one_batch: bool = False,
                      repeats: int = 1,
                      seed_start: int = 0,
                      eval_only: bool = True):
    """Local entrypoint: trigger Maze evaluation.

    Note: eval-only is always enforced internally; this flag is accepted for CLI
    compatibility and ignored (subprocess adds --eval-only regardless).
    """
    _ = eval_only  # accepted but not needed; subprocess always uses --eval-only
    res = run_eval_maze_job.remote(batch_size=batch_size, one_batch=one_batch, repeats=repeats, seed_start=seed_start)  # type: ignore
    print(res)


@app.local_entrypoint()
def cli_run_eval_sudoku(model: str = "mlp",
                                                dataset_path: str | None = None,
                                                batch_size: int = 256,
                                                one_batch: bool = False,
                                                repeats: int = 1,
                                                seed_start: int = 0,
                                                eval_only: bool = True):
        """Local entrypoint: trigger Sudoku evaluation.
        Example:
        modal run infra/modal_app.py::cli_run_eval_sudoku --model=attn
    """
        _ = eval_only  # accepted but not needed; subprocess always uses --eval-only
        res = run_eval_sudoku_job.remote(model=model, dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, repeats=repeats, seed_start=seed_start)  # type: ignore
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
    """
    grid = None
    if grid_json:
        grid = _json.loads(grid_json)
    elif grid_file:
        with open(grid_file, "r", encoding="utf-8") as f:
            grid = _json.load(f)

    res = predict_job.remote(grid=grid, index=index, file=file, task=task, model=model, run=run)  # type: ignore
    print(res)