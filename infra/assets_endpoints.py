import os
import modal
from fastapi import Response

from .modal_base import app, IMAGE, volume
from .utils_repo import ensure_repo


@app.function(volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def get_maze_visualizer():
    repo_dir = ensure_repo()
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
    repo_dir = ensure_repo()
    os.chdir(repo_dir)
    html_path = os.path.join(repo_dir, "sudoku_visualizer.html")
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"Sudoku visualizer HTML not found at {html_path}; ensure repo is cloned")
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return Response(content, media_type="text/html")


@app.function(volumes={"/data": volume})
@modal.fastapi_endpoint(docs=True)
def get_arc_visualizer():
    repo_dir = ensure_repo()
    os.chdir(repo_dir)
    html_path = os.path.join(repo_dir, "arc_visualizer.html")
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"ARC visualizer HTML not found at {html_path}; ensure repo is cloned")
    with open(html_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return Response(content, media_type="text/html")


def _serve_asset_from_repo(filename: str) -> Response:
    repo_dir = ensure_repo()
    asset_path = os.path.join(repo_dir, "assets", filename)
    if os.path.exists(asset_path):
        with open(asset_path, "rb") as f:
            content = f.read()
        return Response(content, media_type="application/javascript" if filename.endswith(".js") else "text/plain")
    return Response("File not found", status_code=404, media_type="text/plain")


@modal.fastapi_endpoint(docs=True)
def get_asset(filename: str):
    return _serve_asset_from_repo(filename)


@modal.fastapi_endpoint(docs=True)
def assets(filename: str):
    return _serve_asset_from_repo(filename)
