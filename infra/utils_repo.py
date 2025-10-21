import os
import re
import sys
import shutil
import subprocess
import threading
import time
from typing import Any

import torch
from fastapi import HTTPException

from .modal_base import volume

# Lightweight cache for prediction files
_PRED_CACHE_LOCK = threading.Lock()
_PRED_CACHE: dict[str, dict] = {}
_PRED_CACHE_META: dict[str, float] = {}
_PRED_CACHE_MAX = 8

def load_preds_cached(preds_path: str):
    mtime = os.path.getmtime(preds_path)
    with _PRED_CACHE_LOCK:
        cached = _PRED_CACHE.get(preds_path)
        if cached is not None and _PRED_CACHE_META.get(preds_path) == mtime:
            return cached
    data = torch.load(preds_path, map_location='cpu')
    with _PRED_CACHE_LOCK:
        if len(_PRED_CACHE) >= _PRED_CACHE_MAX:
            oldest = sorted(_PRED_CACHE_META.items(), key=lambda kv: kv[1])[0][0]
            _PRED_CACHE.pop(oldest, None)
            _PRED_CACHE_META.pop(oldest, None)
        _PRED_CACHE[preds_path] = data
        _PRED_CACHE_META[preds_path] = mtime
    return data

def safe_name(s: str) -> str:
    return ''.join(ch for ch in s if ch.isalnum() or ch in ('-', '_')).lower()


def ensure_repo(repo_dir: str = "/data/repo") -> str:
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        try:
            src = os.path.join(repo_dir, "scripts", "run_eval_only.py")
            dst = os.path.join(repo_dir, "run_eval_only.py")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Synced eval runner to repo root: {dst}")
        except Exception as e:
            print("WARNING: Failed to place run_eval_only.py at repo root:", e)
    return repo_dir


def bootstrap_repo_and_weights(repo_dir: str = "/data/repo") -> dict:
    repo_dir = ensure_repo(repo_dir)
    try:
        print(f"Updating repo at {repo_dir}...")
        subprocess.run(["git", "-C", repo_dir, "fetch", "--all", "--prune"], check=True)
        subprocess.run(["git", "-C", repo_dir, "reset", "--hard", "origin/main"], check=True)
        subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"], check=True)
    except Exception as e:
        print("WARNING: Repo update failed; proceeding with existing contents:", e)

    try:
        src = os.path.join(repo_dir, "scripts", "run_eval_only.py")
        dst = os.path.join(repo_dir, "run_eval_only.py")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Synced eval runner to repo root: {dst}")
    except Exception as e:
        print("WARNING: Failed to sync run_eval_only.py:", e)

    from huggingface_hub import hf_hub_download
    results: dict[str, str] = {}

    maze_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights")
    if os.path.isdir(maze_dir):
        shutil.rmtree(maze_dir)
    os.makedirs(maze_dir, exist_ok=True)
    maze_ckpt_path = hf_hub_download(
        repo_id="YuvrajSingh9886/maze-hard-trm",
        filename="step_32550",
        local_dir=maze_dir,
    )
    results["maze"] = maze_ckpt_path

    sud_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights")
    if os.path.isdir(sud_dir):
        shutil.rmtree(sud_dir)
    os.makedirs(sud_dir, exist_ok=True)
    mlp_path = hf_hub_download(
        repo_id="YuvrajSingh9886/sudoku-extreme-trm",
        filename="step_32550_sudoku_epoch50k",
        local_dir=sud_dir,
    )
    attn_path = hf_hub_download(
        repo_id="YuvrajSingh9886/sudoku-extreme-trm",
        filename="step_39060_sudoku_60k_epoch_attn_type",
        local_dir=sud_dir,
    )
    results["sudoku_mlp"] = mlp_path
    results["sudoku_attn"] = attn_path

    arc_dir = os.path.join(repo_dir, "data", "arc-agi1-weights")
    if os.path.isdir(arc_dir):
        shutil.rmtree(arc_dir)
    os.makedirs(arc_dir, exist_ok=True)
    arc_path = hf_hub_download(
        repo_id="YuvrajSingh9886/arc_agi_1_trm_model",
        filename="step_259320_arc_ag1_attn_type_h3l4",
        local_dir=arc_dir,
    )
    results["arc_attn"] = arc_path

    return {"status": "bootstrapped", "paths": results}


def symlink_latest(parent_dir: str, run_dir: str):
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
        try:
            with open(os.path.join(parent_dir, "latest.txt"), "w") as f:
                f.write(run_name)
        except Exception as e:
            print("Failed to record latest run:", e)


def pick_latest_run(base_dir: str) -> str | None:
    if not os.path.isdir(base_dir):
        return None
    latest_link = os.path.join(base_dir, "latest")
    if os.path.islink(latest_link):
        target = os.readlink(latest_link)
        run_dir = os.path.join(base_dir, target)
        if os.path.isdir(run_dir):
            return run_dir
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != "latest"]
    if not subdirs:
        return None
    subdirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return subdirs[0]


def get_eval_script_path(repo_dir: str) -> str:
    candidates = [
        os.path.join(repo_dir, "run_eval_only.py"),
        os.path.join(repo_dir, "scripts", "run_eval_only.py"),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"Resolved eval script: {p} (exists=True)")
            return p
    print(f"Resolved eval script: not found in candidates {candidates}")
    return candidates[0]


def resolve_checkpoint(weights_dir: str,
                       prefer_attn: bool | None = None,
                       name_contains: list[str] | None = None) -> str:
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
    files: list[tuple[str,str,float]] = []
    for fn in os.listdir(weights_dir):
        p = os.path.join(weights_dir, fn)
        if os.path.isfile(p):
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0
            files.append((fn, p, mtime))
    if not files:
        raise FileNotFoundError(f"No files in weights directory: {weights_dir}")
    candidates = files
    if name_contains:
        def match_all(s: str) -> bool:
            s_low = s.lower()
            return all(sub.lower() in s_low for sub in name_contains)
        filt = [t for t in files if match_all(t[0])]
        candidates = filt or files
    elif prefer_attn is not None:
        if prefer_attn:
            filt = [t for t in files if 'attn' in t[0].lower()]
        else:
            filt = [t for t in files if 'attn' not in t[0].lower()]
        candidates = filt or files
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][1]
