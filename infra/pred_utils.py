import os
import json as _json
from typing import Any, Dict, cast

import numpy as np

from .utils_repo import ensure_repo


def load_dataset_metadata(task: str) -> dict:
    repo_dir = ensure_repo()
    if task == "maze":
        meta_paths = [
            os.path.join(repo_dir, "data", "maze-30x30-hard-1k", "test", "dataset.json"),
        ]
    elif task == "sudoku":
        meta_paths = [
            os.path.join(repo_dir, "data", "sudoku-extreme-1k-aug-1000", "test", "dataset.json"),
            os.path.join(repo_dir, "data", "sudoku-extreme-full", "test", "dataset.json"),
        ]
    else:
        meta_paths = [
            os.path.join(repo_dir, "data", "arc1concept-aug-1000", "test", "dataset.json"),
            os.path.join(repo_dir, "data", "test_arc1", "test", "dataset.json"),
        ]
    for p in meta_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return _json.load(f)
    raise RuntimeError(f"Dataset metadata not found for task={task} under known paths: {meta_paths}")


def format_grid_for_task(task: str, meta: dict, grid: list) -> list:
    arr = np.array(grid)
    if arr.ndim == 1:
        side = int(np.sqrt(arr.size))
        arr = arr.reshape(side, side)
    if task == "sudoku":
        arr = arr.clip(0, 9) + 1
    elif task == "arc":
        from dataset.build_arc_dataset import arc_grid_to_np, np_grid_to_seq_translational_augment
        grid_np = arc_grid_to_np(arr.tolist())
        inp_vec, _ = np_grid_to_seq_translational_augment(grid_np, grid_np, do_translation=False)
        return inp_vec.astype(int).tolist()
    return arr.astype(int).tolist()


essential_crop_cache: dict[str, Any] = {}


def postprocess_preds_for_task(task: str, meta: dict, pred_tokens_1d: list[int]) -> list[list[int]]:
    import numpy as np
    arr = np.array(pred_tokens_1d)
    side = int(np.sqrt(arr.size))
    arr = arr.reshape(side, side)
    if task == "sudoku":
        arr = (arr - 1).clip(0, 9)
    elif task == "arc":
        g = arr
        if g.shape == (30, 30):
            pass
        max_area = 0
        max_r = 0
        max_c = 0
        nr, nc = g.shape
        num_c = nc
        for num_r in range(1, nr + 1):
            for c in range(1, num_c + 1):
                x = g[num_r - 1, c - 1]
                if (x < 2) or (x > 11):
                    num_c = c - 1
                    break
            area = num_r * num_c
            if area > max_area:
                max_area = area
                max_r, max_c = num_r, num_c
        if max_r > 0 and max_c > 0:
            g = g[:max_r, :max_c]
        g = (g - 2).clip(0, 9)
        return g.astype(int).tolist()
    return arr.astype(int).tolist()
