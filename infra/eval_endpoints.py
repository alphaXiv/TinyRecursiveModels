import os
import sys
import time
import shutil
import yaml
import subprocess
from fastapi import Response, HTTPException
import modal

from .modal_base import app, IMAGE, volume, NO_GPU
from .utils_repo import (
    ensure_repo,
    bootstrap_repo_and_weights,
    get_eval_script_path,
    resolve_checkpoint,
    symlink_latest,
)


def _patch_arc_evaluator(repo_dir: str):
    try:
        arc_eval_path = os.path.join(repo_dir, "evaluators", "arc.py")
        if os.path.exists(arc_eval_path):
            with open(arc_eval_path, "r", encoding="utf-8") as f:
                arc_src = f.read()
            if "from numba import njit" in arc_src or "@njit" in arc_src:
                new_src = arc_src
                new_src = new_src.replace("from numba import njit\n", "")
                new_src = new_src.replace("@njit\n", "")
                new_src = new_src.replace("def _crop(", "def _crop_np(")
                new_src = new_src.replace("_crop(", "_crop_np(")
                new_src = new_src.replace(
                    ".get(input_hash, {})", ".get(input_hash, [])"
                )
                with open(arc_eval_path, "w", encoding="utf-8") as f:
                    f.write(new_src)
                print("Patched evaluators/arc.py in cloned repo to remove numba and use pure NumPy cropping.")
    except Exception as _e:
        print("WARNING: Failed to patch evaluators/arc.py:", _e)


# ----------------------- dataset prep -----------------------
@app.function(image=IMAGE, volumes={"/data": volume}, timeout=7200)
def prepare_dataset_job(include_maze: bool = True,
                        include_sudoku: bool = False,
                        include_arc: bool = False,
                        sudoku_output_dir: str = "data/sudoku-extreme-1k-aug-1000",
                        sudoku_subsample_size: int = 1000,
                        sudoku_num_aug: int = 1000):
    repo_dir = ensure_repo()
    bootstrap_repo_and_weights()
    os.chdir(repo_dir)

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

    if include_arc:
        cmd = [
            sys.executable, "-m", "dataset.build_arc_dataset",
            "--input-file-prefix", "kaggle/combined/arc-agi",
            "--output-dir", "data/arc1concept-aug-1000",
            "--subsets", "training", "evaluation", "concept",
            "--test-set-name", "evaluation",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)

    return {
        "status": "Datasets prepared",
        "maze": include_maze,
        "sudoku": include_sudoku,
        "arc": include_arc,
        "sudoku_output_dir": sudoku_output_dir,
    }


@app.function(image=IMAGE, volumes={"/data": volume}, timeout=7200)
@modal.fastapi_endpoint(docs=True)
def prepare_dataset(include_maze: bool = True,
                    include_sudoku: bool = False,
                    include_arc: bool = False,
                    sudoku_output_dir: str = "data/sudoku-extreme-1k-aug-1000",
                    sudoku_subsample_size: int = 1000,
                    sudoku_num_aug: int = 1000):
    return prepare_dataset_job.remote(
        include_maze=include_maze,
        include_sudoku=include_sudoku,
        include_arc=include_arc,
        sudoku_output_dir=sudoku_output_dir,
        sudoku_subsample_size=sudoku_subsample_size,
        sudoku_num_aug=sudoku_num_aug,
    )


# ----------------------- eval runners -----------------------

def _do_run_eval_sudoku(model: str,
                        dataset_path: str | None,
                        batch_size: int = 16,
                        one_batch: bool = False,
                        checkpoint: str | None = None):
    repo_dir = ensure_repo()
    os.chdir(repo_dir)
    need_mlp = (model or "mlp").lower() == "mlp"
    if checkpoint:
        ckpt_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(repo_dir, checkpoint)
    else:
        sud_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full-weights")
        try:
            ckpt_path = resolve_checkpoint(sud_dir, prefer_attn=not need_mlp)
        except Exception:
            ckpt_path = os.path.join(sud_dir, "step_32550_sudoku_epoch50k" if need_mlp else "step_39060_sudoku_60k_epoch_attn_type")

    if dataset_path:
        dataset_dir = dataset_path
    else:
        dataset_dir = os.path.join(repo_dir, "data", "sudoku-extreme-1k-aug-1000")
        if not os.path.isdir(dataset_dir):
            dataset_dir = os.path.join(repo_dir, "data", "sudoku-extreme-full")
    if not os.path.isdir(dataset_dir):
        print("WARNING: Sudoku dataset folder not found at", dataset_dir)

    parent = os.path.join(repo_dir, "out", "sudoku", (model or "mlp").lower())
    os.makedirs(parent, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    arch_dir = os.path.join(repo_dir, "config", "arch")
    trm_path = os.path.join(arch_dir, "trm.yaml")
    backup_path = os.path.join(arch_dir, "trm.yaml.bak")

    eval_script = get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", f"--nproc_per_node={NO_GPU}", eval_script,
        "--checkpoint", ckpt_path,
        "--dataset", dataset_dir,
        "--outdir", out_dir,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    if one_batch:
        cmd.append("--one-batch")

    need_mlp = (model or "mlp").lower() == "mlp"
    backed_up = False
    try:
        if os.path.exists(trm_path):
            with open(trm_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            desired = True if need_mlp else False
            current = bool(data.get("mlp_t", False))
            if current != desired:
                if os.path.exists(trm_path):
                    shutil.copy2(trm_path, backup_path)
                    backed_up = True
                data["mlp_t"] = desired
                with open(trm_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False)
        print("Resolved checkpoint:", ckpt_path)
        print("Running Sudoku eval:", " ".join(cmd))
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        if backed_up and os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, trm_path)
                os.remove(backup_path)
            except Exception as _e:
                print("WARNING: Failed to restore original arch file:", _e)
    symlink_latest(parent, out_dir)
    return {"status": "Evaluation completed", "output_dir": out_dir, "result": getattr(result, 'returncode', 0), "run_id": run_id}


def _do_run_eval_maze(batch_size: int = 256,
                      one_batch: bool = False,
                      checkpoint: str | None = None):
    repo_dir = ensure_repo()
    os.chdir(repo_dir)
    if checkpoint:
        ckpt_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(repo_dir, checkpoint)
    else:
        maze_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k-weights")
        try:
            ckpt_path = resolve_checkpoint(maze_dir)
        except Exception:
            ckpt_path = os.path.join(maze_dir, "step_32550")
    dataset_dir = os.path.join(repo_dir, "data", "maze-30x30-hard-1k")
    if not os.path.isdir(dataset_dir):
        print("WARNING: Maze dataset folder not found at", dataset_dir)

    parent = os.path.join(repo_dir, "out", "maze", "default")
    os.makedirs(parent, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    arch_dir = os.path.join(repo_dir, "config", "arch")
    trm_path = os.path.join(arch_dir, "trm.yaml")
    backup_path = os.path.join(arch_dir, "trm.yaml.bak")
    backed_up = False
    try:
        if os.path.exists(trm_path):
            with open(trm_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            desired = False
            current = bool(data.get("mlp_t", False))
            if current != desired:
                shutil.copy2(trm_path, backup_path)
                backed_up = True
                data["mlp_t"] = desired
                with open(trm_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(data, f, sort_keys=False)

        eval_script = get_eval_script_path(repo_dir)
        cmd = [
            "torchrun", f"--nproc_per_node={NO_GPU}", eval_script,
            "--checkpoint", ckpt_path,
            "--dataset", dataset_dir,
            "--outdir", out_dir,
            "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
            "--eval-only",
            "--bf16",
            "--global-batch-size", str(int(batch_size)),
        ]
        if one_batch:
            cmd.append("--one-batch")
        print("Running Maze eval:", " ".join(cmd))
        result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    finally:
        if backed_up and os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, trm_path)
                os.remove(backup_path)
            except Exception as _e:
                print("WARNING: Failed to restore original arch file:", _e)
    symlink_latest(parent, out_dir)
    return {"status": "Evaluation completed", "output_dir": out_dir, "result": getattr(result, 'returncode', 0), "run_id": run_id}


def _do_run_eval_arc(dataset_path: str | None = None,
                     batch_size: int = 256,
                     one_batch: bool = False,
                     checkpoint: str | None = None):
    repo_dir = ensure_repo()
    os.chdir(repo_dir)
    os.environ.setdefault("DISABLE_COMPILE", "1")
    _patch_arc_evaluator(repo_dir)

    if checkpoint:
        ckpt_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(repo_dir, checkpoint)
    else:
        arc_dir = os.path.join(repo_dir, "data", "arc-agi1-weights")
        try:
            ckpt_path = resolve_checkpoint(arc_dir, prefer_attn=True)
        except Exception:
            ckpt_path = os.path.join(arc_dir, "step_259320_arc_ag1_attn_type_h3l4")

    if dataset_path:
        dataset_dir = dataset_path
    else:
        dataset_dir = os.path.join(repo_dir, "data", "arc1concept-aug-1000")
        if not os.path.isdir(dataset_dir):
            dataset_dir = os.path.join(repo_dir, "data", "test_arc1")
    if not os.path.isdir(dataset_dir):
        print("WARNING: ARC dataset folder not found at", dataset_dir)

    parent = os.path.join(repo_dir, "out", "arc", "attn")
    os.makedirs(parent, exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(parent, run_id)
    os.makedirs(out_dir, exist_ok=True)

    eval_script = get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", "--nproc_per_node=8", eval_script,
        "--checkpoint", ckpt_path,
        "--dataset", dataset_dir,
        "--outdir", out_dir,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    if one_batch:
        cmd.append("--one-batch")
    print("Running ARC eval:", " ".join(cmd))
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    symlink_latest(parent, out_dir)
    return {"status": "Evaluation completed", "output_dir": out_dir, "result": getattr(result, 'returncode', 0), "run_id": run_id}


# ----------------------- public job/webhooks -----------------------
@app.function(image=IMAGE, volumes={"/data": volume}, gpu=f"A100:{NO_GPU}")
def run_eval_sudoku_job(model: str = "mlp",
                        dataset_path: str | None = None,
                        batch_size: int = 256,
                        one_batch: bool = False,
                        checkpoint: str | None = None):
    ensure_repo()
    return _do_run_eval_sudoku(model=model, dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def run_eval_sudoku(model: str = "mlp",
                    dataset_path: str | None = None,
                    batch_size: int = 256,
                    one_batch: bool = False,
                    checkpoint: str | None = None):
    return run_eval_sudoku_job.remote(model=model, dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)


@app.function(image=IMAGE, volumes={"/data": volume}, gpu=f"A100:{NO_GPU}")
def run_eval_maze_job(batch_size: int = 256,
                      one_batch: bool = False,
                      checkpoint: str | None = None):
    ensure_repo()
    return _do_run_eval_maze(batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def run_eval_maze(batch_size: int = 256,
                  one_batch: bool = False,
                  checkpoint: str | None = None):
    return run_eval_maze_job.remote(batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)


@app.function(image=IMAGE, volumes={"/data": volume}, gpu="A100:2", timeout=28800)
def run_eval_arc_job(dataset_path: str | None = None,
                     batch_size: int = 256,
                     one_batch: bool = False,
                     checkpoint: str | None = None):
    ensure_repo()
    return _do_run_eval_arc(dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def run_eval_arc(dataset_path: str | None = None,
                 batch_size: int = 256,
                 one_batch: bool = False,
                 checkpoint: str | None = None):
    return run_eval_arc_job.remote(dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)


@app.function(image=IMAGE, volumes={"/data": volume}, timeout=3600)
@modal.fastapi_endpoint(docs=True)
def eval_test(checkpoint_path: str = "data/maze-30x30-hard-1k-weights/step_32550", dataset_path: str = "data/maze-30x30-hard-1k", out_dir: str = "out", batch_size: int = 256):
    repo_dir = ensure_repo()
    os.chdir(repo_dir)

    local_out = os.path.join(repo_dir, out_dir)
    os.makedirs(local_out, exist_ok=True)

    eval_script = get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", f"--nproc_per_node={NO_GPU}", eval_script,
        "--checkpoint", os.path.join(repo_dir, checkpoint_path),
        "--dataset", os.path.join(repo_dir, dataset_path),
        "--outdir", local_out,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only",
        "--bf16",
        "--global-batch-size", str(int(batch_size)),
    ]
    print(f"Running evaluation command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    return {"status": "Evaluation completed", "output_dir": local_out, "result": result}


@modal.fastapi_endpoint(docs=True)
def run_eval_local(checkpoint_path: str="data/maze-30x30-hard-1k", dataset_path: str = "maze-30x30-hard-1k-weights/step_32550", out_dir: str = "out", batch_size: int = 256):
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    repo_dir = "/data/repo"

    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    os.chdir(repo_dir)

    local_out = os.path.join(repo_dir, out_dir)
    os.makedirs(local_out, exist_ok=True)
    eval_script = get_eval_script_path(repo_dir)
    cmd = [
        "torchrun", f"--nproc_per_node={NO_GPU}", eval_script,
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

    return {'message': 'Evaluation completed', 'output_dir': out_dir}
