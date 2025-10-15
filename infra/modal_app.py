
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
from fastapi import Response
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


@app.function(image=IMAGE, volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def prepare_dataset():
    """Prepare the maze dataset on Modal."""
    
    # Use persistent data directory
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    repo_dir = "/data/repo"
    
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    # Change to repo directory
    os.chdir(repo_dir)
    
    # Install requirements
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Run dataset preparation
    cmd = ["python", "dataset/build_maze_dataset.py"]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    return {"status": "Dataset prepared successfully"}


@app.function(image=IMAGE, timeout=3600, gpu="A100:2",volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def eval_test(checkpoint_path: str = "data/maze-30x30-hard-1k-weights/step_32550", dataset_path: str = "data/maze-30x30-hard-1k", out_dir: str = "out"):
    """Run evaluation on the mounted test dataset and save predictions to persistent volume.

    This will raise on any failure (no fallback behavior) as requested.
    """
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    repo_dir = "/data/repo"

    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    os.chdir(repo_dir)

    local_out = os.path.join(repo_dir, out_dir)
    os.makedirs(local_out, exist_ok=True)
    
    # shutil.move('scripts/run_eval_only.py', './')
    # Build command to run evaluation single-process (no torchrun distributed)
    cmd = [
        sys.executable, "run_eval_only.py",
        "--checkpoint", os.path.join(repo_dir, checkpoint_path),
        "--dataset", os.path.join(repo_dir, dataset_path),
        "--outdir", local_out,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only"
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
def run_eval_local(checkpoint_path: str="data/maze-30x30-hard-1k", dataset_path: str = "maze-30x30-hard-1k-weights/step_32550", out_dir: str = "out"):
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
    cmd = [
        "torchrun", "--nproc_per_node=2", "run_eval_only.py",
        "--checkpoint", os.path.join(repo_dir, checkpoint_path),
        "--dataset", os.path.join(repo_dir, dataset_path),
        "--outdir", local_out,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds",
        "--eval-only"
    ]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    print("Evaluation completed.")

    metrics = {}  # No metrics returned from subprocess

    # Return results
    return {'message': 'Evaluation completed', 'output_dir': out_dir}


@app.function(image = IMAGE, volumes= {"/data": volume})
@modal.fastapi_endpoint(docs=True)
def predict(grid: object = None):
    """Predict solved maze from input grid.

    Accepts either:
    - POST with a JSON body containing a list (the grid), or
    - GET with a `grid` query parameter containing JSON (stringified list).
    If no grid is provided, the function will load saved predictions from the
    evaluation outputs and return the first example.
    """
    # As requested: use test-set predictions from evaluation outputs on the persistent volume.
    repo_dir = "/data/repo"
    out_dir = os.path.join(repo_dir, "out")

    if not os.path.exists(repo_dir):
        raise RuntimeError(f"Repo not found at {repo_dir}; expected it to be cloned into the persistent volume")

    # Expect evaluation to have saved preds into out_dir
    preds_path = None
    # Look for files that contain 'all_preds' or 'preds' inside out_dir
    for fname in os.listdir(out_dir):
        if fname.endswith('.0') or 'all_preds' in fname or 'preds' in fname:
            preds_path = os.path.join(out_dir, fname)
            break

    if preds_path is None:
        raise RuntimeError(f"No prediction file found in {out_dir}; run eval_test first")

    # Load predictions saved by evaluate (torch.save of a dict of tensors)
    import torch
    data = torch.load(preds_path, map_location='cpu')

    if 'preds' not in data:
        raise RuntimeError(f"Loaded predictions file {preds_path} does not contain key 'preds'")

    preds = data['preds']  # Expect shape (N, seq_len, ...)

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
    if provided_grid is not None:
        grid_arr = np.array(provided_grid)
    else:
        # For now take the first example from saved preds
        first = preds[0]
        try:
            grid_arr = first.argmax(axis=-1).cpu().numpy() if getattr(first, 'ndim', 0) > 1 else first.cpu().numpy()
        except Exception:
            grid_arr = first.cpu().numpy()

    # Infer grid size
    import math
    L = grid_arr.shape[-1] if getattr(grid_arr, 'ndim', 0) > 0 else len(grid_arr)
    side = int(math.sqrt(L))
    if side * side != L:
        # If not a perfect square, attempt 30x30 as default
        side = 30
        L = side * side
        grid_arr = grid_arr.flat[:L]

    solved_maze = np.asarray(grid_arr).reshape((side, side)).tolist()

    return {"solved_maze": solved_maze, "source_file": preds_path}


# @app.function(volumes={"/data": volume})
# @modal.fastapi_endpoint(docs=True)
# def get_visualizer():
#     """Serve the maze visualizer HTML."""
#     # Serve the external `puzzle_visualizer.html` from the repo in the persistent volume
#     repo_dir = "/data/repo"
#     os.chdir(repo_dir)
#     html_path = os.path.join(repo_dir, "puzzle_visualizer.html")

#     subprocess.run(["ls"], stdout=sys.stdout, stderr=sys.stderr, check=False)
#     if not os.path.exists(html_path):
#         raise FileNotFoundError(f"Visualizer HTML not found at {html_path}; ensure repo is cloned and file exists")

#     with open(html_path, 'r', encoding='utf-8') as f:
#         content = f.read()

#     # Attempt to inline npyjs if the HTML references it so the client
#     # doesn't need to fetch /assets/npyjs.js separately (which may 404
#     # due to routing differences). This ensures the visualizer gets the
#     # library directly from the repository file.
#     try:
#         # Look for common script tags that reference the asset
#         if 'npyjs.js' in content:
#             asset_file = os.path.join(repo_dir, 'assets', 'npyjs.js')
#             if os.path.exists(asset_file):
#                 with open(asset_file, 'r', encoding='utf-8') as af:
#                     npyjs_src = af.read()

#                 # Replace any <script src="...npyjs.js"></script> with inline code
#                 # Match both /get_asset?filename=npyjs.js and assets/npyjs.js occurrences
#                 content = content.replace('<script src="/get_asset?filename=npyjs.js"></script>', f'<script>\n{npyjs_src}\n</script>')
#                 content = content.replace('<script src="assets/npyjs.js"></script>', f'<script>\n{npyjs_src}\n</script>')
#                 content = content.replace("<script src='assets/npyjs.js'></script>", f"<script>\n{npyjs_src}\n</script>")
#                 print(f"Inlined npyjs from {asset_file} into visualizer HTML")
#             else:
#                 print(f"npyjs asset not found at {asset_file}; leaving HTML unchanged")
#     except Exception as e:
#         print(f"Failed to inline npyjs.js into visualizer HTML: {e}")

#     return Response(content, media_type="text/html")


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


@modal.fastapi_endpoint(docs=True)
def get_asset(filename: str):
    """Serve asset files from the repo."""
    
    # Use persistent data directory
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    repo_dir = "/data/repo"
    
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    asset_path = os.path.join(repo_dir, "assets", filename)
    
    
    if os.path.exists(asset_path):
        with open(asset_path, "rb") as f:
            content = f.read()
        return Response(content, media_type="application/javascript" if filename.endswith(".js") else "text/plain")
    # Add a helpful log message for debugging missing assets
    print(f"get_asset: asset not found at {asset_path}")
    return Response("File not found", status_code=404, media_type="text/plain")


@modal.fastapi_endpoint(docs=True)
def assets(filename: str):
    """Compatibility wrapper: serve files at /assets/{filename} by delegating to get_asset.

    Some HTML in the repo requests /assets/npyjs.js directly. This wrapper ensures
    that path resolves to the same file-serving logic without changing other code.
    """
    # Delegate to get_asset which clones the repo if needed and reads the file
    return get_asset(filename)