import os
import math
import json as _json
from typing import Optional

import numpy as np
from fastapi import Body, Query, HTTPException
from fastapi.responses import Response, JSONResponse
import time
import modal

from .modal_base import app, IMAGE, volume
from .utils_repo import safe_name, ensure_repo, pick_latest_run, load_preds_cached
from .pred_utils import load_dataset_metadata, postprocess_preds_for_task


def _do_predict(grid: object | None = None, index: int | None = None, file: str | None = None, task: str | None = None, model: str | None = None, run: str | None = None):
    repo_dir = "/data/repo"
    base_out = os.path.join(repo_dir, "out")
    safe_task = safe_name(task or "maze")
    if safe_task not in ("maze", "sudoku", "arc"):
        raise HTTPException(status_code=400, detail="task must be 'maze', 'sudoku', or 'arc'")
    if safe_task == "maze":
        if model not in (None, "", "default"):
            raise HTTPException(status_code=400, detail="maze has a single model; omit 'model' or use model=default")
        safe_model = "default"
    elif safe_task == "sudoku":
        safe_model = safe_name(model or "mlp")
        if safe_model not in ("mlp", "attn"):
            raise HTTPException(status_code=400, detail="For sudoku, model must be 'mlp' or 'attn'")
    else:
        safe_model = "attn"
    task_dir = os.path.join(base_out, safe_task, safe_model)

    if run:
        safe_run = safe_name(run)
        out_dir = os.path.join(task_dir, safe_run)
    else:
        guess = pick_latest_run(task_dir)
        if guess is None:
            raise HTTPException(status_code=404, detail=f"No runs found in {task_dir}. Run evaluation first.")
        out_dir = guess

    if not os.path.exists(repo_dir):
        raise RuntimeError(f"Repo not found at {repo_dir}; expected it to be cloned into the persistent volume")
    if not os.path.isdir(out_dir):
        raise HTTPException(status_code=404, detail=f"Output directory not found: {out_dir}")

    preds_path = None
    if file is not None:
        candidate = os.path.join(out_dir, file)
        candidate = os.path.realpath(candidate)
        if not candidate.startswith(os.path.realpath(out_dir) + os.sep):
            raise HTTPException(status_code=400, detail="Invalid file path")
        if not os.path.exists(candidate):
            raise HTTPException(status_code=404, detail=f"Specified preds file not found: {file}")
        preds_path = candidate
    else:
        best = None
        best_step = -1
        for fname in os.listdir(out_dir):
            import re
            m = re.match(r"step_(\d+)_all_preds\.[0-9]+$", fname)
            if m:
                step = int(m.group(1))
                if step > best_step:
                    best_step = step
                    best = fname
        if best is None:
            raise HTTPException(status_code=404, detail=f"No preds file matching 'step_<N>_all_preds.<rank>' found in {out_dir}. Run eval.")
        preds_path = os.path.join(out_dir, best)

    data = load_preds_cached(preds_path)
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail=f"Loaded predictions file {preds_path} is not a dict of tensors")
    if 'preds' not in data:
        raise HTTPException(status_code=400, detail=f"Loaded predictions file {preds_path} does not contain key 'preds'")

    preds = data['preds']

    provided_grid = None
    if grid is not None:
        try:
            if isinstance(grid, str):
                provided_grid = _json.loads(grid)
            else:
                provided_grid = grid
        except Exception:
            return {"error": "Failed to parse provided grid; send JSON list in POST body or JSON string in 'grid' query param."}

    ret_index = None
    if provided_grid is not None:
        grid_arr = np.array(provided_grid)
    else:
        try:
            total = len(preds)
        except Exception:
            raise HTTPException(status_code=400, detail="'preds' tensor is not indexable")
        idx = 0 if index is None else int(index)
        if idx < 0 or idx >= total:
            raise HTTPException(status_code=400, detail=f"Index {idx} out of range (0..{total-1})")
        sel = preds[idx]
        ret_index = idx
        try:
            if getattr(sel, 'ndim', 0) >= 2 and sel.shape[-1] > 8:
                arr = sel.argmax(dim=-1)
            else:
                arr = sel
            grid_arr = arr.detach().cpu().numpy() if hasattr(arr, 'detach') else np.array(arr)
        except Exception:
            grid_arr = np.array(sel)

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

    if safe_task == "arc":
        try:
            meta_arc = load_dataset_metadata("arc")
        except Exception:
            meta_arc = {}
        try:
            tokens = np.array(solved_maze).reshape(-1).astype(int).tolist()
            solved_maze = postprocess_preds_for_task("arc", meta_arc, tokens)
        except Exception:
            pass

    input_maze = None
    target_maze = None
    if provided_grid is not None:
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

    if safe_task == "arc" and provided_grid is None:
        try:
            meta_arc = load_dataset_metadata("arc")
        except Exception:
            meta_arc = {}
        try:
            if input_maze is not None:
                inp_tokens = np.array(input_maze).reshape(-1).astype(int).tolist()
                input_maze = postprocess_preds_for_task("arc", meta_arc, inp_tokens)
        except Exception:
            pass
        try:
            if target_maze is not None:
                tgt_tokens = np.array(target_maze).reshape(-1).astype(int).tolist()
                target_maze = postprocess_preds_for_task("arc", meta_arc, tgt_tokens)
        except Exception:
            pass

    payload = {"solved_maze": solved_maze, "input_maze": input_maze, "source_file": preds_path, "index": ret_index, "task": safe_task, "model": safe_model, "run_dir": out_dir}
    if provided_grid is not None:
        try:
            custom_dir = os.path.join(base_out, safe_task, safe_model, "custom")
            os.makedirs(custom_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            import uuid
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
            payload["saved_error"] = str(_e)
    if target_maze is not None:
        payload["target_maze"] = target_maze
    return payload


@app.function(image = IMAGE, volumes= {"/data": volume})
@modal.fastapi_endpoint(docs=True, method="GET")
def predict(
    grid: object | None = Body(default=None),
    index: int | None = None,
    file: str | None = None,
    task: str | None = Body(default=None),
    model: str | None = Body(default=None),
    run: str | None = Body(default=None),
    task_q: str | None = Query(default=None, alias="task"),
    model_q: str | None = Query(default=None, alias="model"),
    run_q: str | None = Query(default=None, alias="run"),
    grid_q: str | None = Query(default=None, alias="grid"),
):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    try:
        eff_task = task if task is not None else task_q
        eff_model = model if model is not None else model_q
        eff_run = run if run is not None else run_q
        eff_grid = grid if grid is not None else grid_q
        payload = _do_predict(grid=eff_grid, index=index, file=file, task=eff_task, model=eff_model, run=eff_run)
        return JSONResponse(content=payload, headers=headers)
    except HTTPException as e:
        return JSONResponse(content={"detail": e.detail}, status_code=e.status_code, headers=headers)
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500, headers=headers)


@app.function(image = IMAGE, volumes= {"/data": volume})
@modal.fastapi_endpoint(docs=False, method="OPTIONS")
def predict_options():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Max-Age": "86400",
    }
    return Response(status_code=204, headers=headers)


@app.function(image=IMAGE, volumes={"/data": volume})
def predict_job(grid: object = None, index: int | None = None, file: str | None = None, task: str | None = None, model: str | None = None, run: str | None = None):
    return _do_predict(grid=grid, index=index, file=file, task=task, model=model, run=run)
