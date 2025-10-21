import modal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Persistent volume for repo/data
volume = modal.Volume.from_name("tinyrecursive-data", create_if_missing=True)

# Number of GPUs to request for single-node torchrun
NO_GPU = 1

# Base image with dependencies
IMAGE = (
    modal.Image.debian_slim()
    .run_commands(
        "apt-get update",
        "apt-get install -y git",
        "pip install --upgrade pip",
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
        "triton",
    ])
    .pip_install([
        "torch",
        "torchvision",
        "torchaudio",
    ], index_url="https://download.pytorch.org/whl/cu126")
)

APP_NAME = "tinyrecursive-eval"
app = modal.App(name=APP_NAME, image=IMAGE)

# A small FastAPI app (Modal will mount endpoints into its own FastAPI instance)
fastapi_app = None
try:
    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass
