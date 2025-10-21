import os
import sys
import importlib
from typing import Any, Dict, cast

import numpy as np
import torch
from omegaconf import OmegaConf
from fastapi import Body, Query, HTTPException
from fastapi.responses import JSONResponse
import modal

from .modal_base import app, IMAGE, volume, NO_GPU
from .utils_repo import ensure_repo
from .pred_utils import load_dataset_metadata, format_grid_for_task, postprocess_preds_for_task

_RT_MODELS_LOCK = torch.multiprocessing.get_context("spawn").Lock() if hasattr(torch, 'multiprocessing') else None
_RT_MODELS: dict[tuple[str,str], dict] = {}


def _load_checkpoint_compat(model, ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    def normalize_key(key: str) -> str:
        if key.startswith('.'):
            key = key[1:]
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
    incompat = model.load_state_dict(sd, strict=True)
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


def _get_realtime_model(task: str, model: str | None):
    repo_dir = ensure_repo()
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    safe_task = (task or "maze").lower()
    safe_model = (model or ("default" if safe_task=="maze" else ("mlp" if safe_task=="sudoku" else "attn"))).lower()
    if safe_task == "maze":
        safe_model = "default"
    elif safe_task == "arc":
        safe_model = "attn"
    key = (safe_task, safe_model)

    if key in _RT_MODELS:
        return _RT_MODELS[key]

    os.environ.setdefault("DISABLE_COMPILE", "1")
    trm_mod = importlib.import_module("models.recursive_reasoning.trm")
    TinyRecursiveReasoningModel_ACTV1 = getattr(trm_mod, "TinyRecursiveReasoningModel_ACTV1")
    losses_mod = importlib.import_module("models.losses")
    ACTLossHead = getattr(losses_mod, "ACTLossHead")

    arch_path = os.path.join(repo_dir, "config", "arch", "trm.yaml")
    oc = OmegaConf.load(arch_path)
    if safe_task == "sudoku" and safe_model == "mlp":
        oc.mlp_t = True
    else:
        oc.mlp_t = False
    arch_cfg: Dict[str, Any] = cast(Dict[str, Any], OmegaConf.to_container(oc, resolve=True))

    meta = load_dataset_metadata(safe_task)
    seq_len = int(meta["seq_len"])
    vocab_size = int(meta["vocab_size"])
    num_ids = int(meta["num_puzzle_identifiers"])

    model_cfg: Dict[str, Any] = {**arch_cfg,
                                 "batch_size": 1,
                                 "seq_len": seq_len,
                                 "vocab_size": vocab_size,
                                 "num_puzzle_identifiers": num_ids}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    core = TinyRecursiveReasoningModel_ACTV1(model_cfg)
    loss_cfg = arch_cfg.get("loss") if isinstance(arch_cfg.get("loss"), dict) else None
    loss_type = (loss_cfg.get("loss_type") if isinstance(loss_cfg, dict) else None) or "softmax_cross_entropy"
    act = ACTLossHead(core, loss_type=loss_type)
    act.eval()
    act.to(device)

    if safe_task == "maze":
        ckpt_path = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights", "step_32550")
    elif safe_task == "sudoku":
        ckpt_path = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights",
                                 "step_32550_sudoku_epoch50k" if safe_model=="mlp" else "step_39060_sudoku_60k_epoch_attn_type")
    else:
        ckpt_path = os.path.join(repo_dir, "data", "arc-agi1-weights", "step_259320_arc_ag1_attn_type_h3l4")

    _load_checkpoint_compat(act, ckpt_path)

    entry = {"model": act, "meta": meta, "device": device, "task": safe_task, "model_name": safe_model}
    _RT_MODELS[key] = entry
    return entry


@app.function(image=IMAGE, volumes={"/data": volume}, gpu=f"A100:{NO_GPU}")
@modal.fastapi_endpoint(docs=True, method="POST")
def predict_realtime(
    grid: object | None = Body(default=None),
    task: str | None = Body(default=None),
    model: str | None = Body(default=None),
    grid_q: str | None = Query(default=None, alias="grid"),
    task_q: str | None = Query(default=None, alias="task"),
    model_q: str | None = Query(default=None, alias="model"),
):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    try:
        eff_task = task if task is not None else task_q
        eff_model = model if model is not None else model_q
        eff_task = (eff_task or "maze").lower()
        eff_model = (eff_model or ("default" if eff_task=="maze" else ("mlp" if eff_task=="sudoku" else "attn"))).lower()
        if eff_task not in ("maze", "sudoku", "arc"):
            raise HTTPException(status_code=400, detail="task must be 'maze', 'sudoku', or 'arc'")
        if eff_task == "maze":
            eff_model = "default"
        elif eff_task == "sudoku" and eff_model not in ("mlp", "attn"):
            raise HTTPException(status_code=400, detail="For sudoku, model must be 'mlp' or 'attn'")
        else:
            eff_model = "attn"

        entry = _get_realtime_model(eff_task, eff_model)
        model_entry = entry["model"]
        meta = entry["meta"]
        device = entry["device"]

        # Normalize grid to token space
        eff_grid = grid if grid is not None else grid_q
        if eff_grid is None:
            raise HTTPException(status_code=400, detail="grid is required")
        if isinstance(eff_grid, str):
            import json as _json
            eff_grid = _json.loads(eff_grid)
        inp_tokens = format_grid_for_task(eff_task, meta, cast(list, eff_grid))

        # Build batch to mirror original modal_app behavior (include labels and puzzle_identifiers)
        x = torch.tensor(inp_tokens, device=device).view(1, -1)
        labels = x.clone()
        puzzle_identifiers = torch.zeros((1,), dtype=torch.int32, device=device)
        batch = {"inputs": x, "labels": labels, "puzzle_identifiers": puzzle_identifiers}

        # Run ACT loop until halting, ensuring carry tensors are on device
        with torch.inference_mode():
            carry = model_entry.initial_carry(batch)  # type: ignore
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
                carry, loss, metrics, preds, all_finish = model_entry(
                    return_keys=return_keys,
                    batch=batch,
                    carry=carry,
                )
                steps += 1
                if bool(all_finish):
                    break
            pred_tokens = preds.get("preds")
            if pred_tokens is None:
                logits = preds.get("logits")
                pred_tokens = torch.argmax(logits, dim=-1)
            preds_vec = pred_tokens.detach().cpu().numpy().reshape(-1).tolist()
            solved = postprocess_preds_for_task(eff_task, meta, preds_vec)

        # Optional nicety: echo provided grid in 2D form if it was a flat list
        try:
            import numpy as _np
            arr = _np.array(eff_grid)
            if arr.ndim == 1:
                side = int(_np.sqrt(arr.size))
                display_input = arr.reshape(side, side).astype(int).tolist()
            else:
                display_input = arr.astype(int).tolist()
        except Exception:
            display_input = None

        payload = {
            "task": eff_task,
            "model": eff_model,
            "input_maze": display_input,
            "solved_maze": solved,
            "inference_steps": steps,
        }
        return JSONResponse(content=payload, headers=headers)
    except HTTPException as e:
        return JSONResponse(content={"detail": e.detail}, status_code=e.status_code, headers=headers)
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=500, headers=headers)
