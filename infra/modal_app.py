
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


@app.function(image=IMAGE, gpu="A100", volumes={"/data": volume}, timeout=3600)
@modal.fastapi_endpoint(docs=True)
def run_eval_local(checkpoint_path: str="dataset/maze-30x30-hard-1k/test", dataset_path: str = "maze-30x30-hard-1k-weights/step_32550", out_dir: str = "out"):
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
    subprocess.run(["ls"], stdout=sys.stdout, stderr=sys.stderr, check=True)
    # Run evaluation using subprocess with torchrun for multi-GPU
    cmd = [
        "torchrun", "--nproc_per_node=2", "run_eval_only.py",
        "--checkpoint", os.path.join(repo_dir, checkpoint_path),
        "--dataset", os.path.join(repo_dir, dataset_path),
        "--outdir", local_out,
        "--eval-save-outputs", "inputs", "labels", "puzzle_identifiers", "preds"
    ]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    print("Evaluation completed.")

    metrics = {}  # No metrics returned from subprocess

    # Return results
    return {'message': 'Evaluation completed', 'output_dir': out_dir}


@app.function()
@modal.fastapi_endpoint(method="POST")
def predict(grid: list):
    """Predict solved maze from input grid."""
    # For now, return a mock solved maze
    # In the future, this will load the actual model and run inference
    solved_maze = [
        [1, 4, 1],
        [4, 2, 4],
        [1, 4, 3]
    ]
    return {"solved_maze": solved_maze}


@app.function()
@modal.fastapi_endpoint(docs=True)
def get_visualizer():
    """Serve the maze visualizer HTML."""
    html_content = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Maze Solver - Tiny Recursive Models</title>
  <style>
    body { font-family: sans-serif; margin: 16px; }
    .selector-area { margin-bottom: 1rem; }
    .grid-canvas { margin: 4px; border: 1px solid #ccc; }
    .example-container { display: inline-block; margin: 0 16px 16px 0; vertical-align: top; }
    .puzzle-display { margin-top: 1rem; }
    .puzzle-id { font-weight: bold; margin-bottom: 0.5rem; }
    #groupList, #puzzleList { margin: 1rem 0; }
    .group-item, .puzzle-item { cursor: pointer; margin: 4px 8px 4px 0; padding: 2px 6px; border: 1px solid #aaa; display: inline-block; }
    .group-item:hover, .puzzle-item:hover { background: #eef; }
  </style>
</head>
<body>
<h1>Maze Solver - Tiny Recursive Models</h1>

<div class="selector-area">
  <p>This is a demo interface for the Maze-Hard solver.</p>
  <p>Load maze data and click "Predict Solved Maze" to get AI-generated solutions.</p>
</div>

<div>
  <div id="groupList"></div>
  <div id="puzzleList"></div>
  <div class="puzzle-display" id="puzzleView"></div>
  <button id="predictBtn" style="margin-top: 1rem; padding: 10px 20px; font-size: 16px;">Predict Solved Maze</button>
  <div id="predictionResult" style="margin-top: 1rem;"></div>
</div>

<script>
document.getElementById('predictBtn').addEventListener('click', async () => {
  const resultDiv = document.getElementById('predictionResult');
  resultDiv.innerHTML = 'Getting prediction from AI...';

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ grid: [[1,4,1],[4,2,4],[1,4,1]] })
    });
    const data = await response.json();
    const solved = data.solved_maze;
    resultDiv.innerHTML = '<h3>AI-Solved Maze:</h3>' + renderGrid(solved);
  } catch (error) {
    resultDiv.innerHTML = 'Error: ' + error.message;
  }
});

function renderGrid(grid) {
  let html = '<div style="display: inline-block; border: 1px solid #000;">';
  for (let row of grid) {
    html += '<div style="display: flex;">';
    for (let cell of row) {
      let color;
      if (cell === 1) color = "#000000"; // wall
      else if (cell === 2) color = "#FF0000"; // start
      else if (cell === 3) color = "#00FF00"; // goal
      else if (cell === 4) color = "#FFFFFF"; // open
      else color = "#FFFFFF";
      html += `<div style="width: 20px; height: 20px; background-color: ${color}; border: 1px solid #ccc;"></div>`;
    }
    html += '</div>';
  }
  html += '</div>';
  return html;
}
</script>
</body>
</html>
"""
    return modal.asgi.Response(html_content, media_type="text/html")


@app.function(volumes={"/data": volume})
@modal.fastapi_endpoint()
def get_asset(filename: str):
    """Serve asset files from the repo."""
    
    # Use persistent data directory
    repo_url = "https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git"
    repo_dir = "/data/repo"
    
    if not os.path.exists(repo_dir):
        print(f"Cloning repo from {repo_url}...")
        subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
    
    try:
        asset_path = os.path.join(repo_dir, "assets", filename)
        with open(asset_path, "rb") as f:
            content = f.read()
        return modal.asgi.Response(content, media_type="application/javascript" if filename.endswith(".js") else "text/plain")
    except FileNotFoundError:
        return modal.asgi.Response("File not found", status_code=404)