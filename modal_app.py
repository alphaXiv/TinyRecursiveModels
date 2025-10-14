#!/usr/bin/env python3
"""Modal app to run evaluation on a GPU instance and upload predictions to Google Cloud Storage.

Usage (local test):
  modal run modal_app.py --checkpoint_gcs gs://your-bucket/checkpoints/step_50000 --dataset_gcs_prefix gs://your-bucket/datasets/maze-30x30-hard-1k/test --out_gcs_prefix gs://your-bucket/preds/maze-30x30-hard-1k

Notes:
- This script downloads the checkpoint and dataset files from a GCS prefix, runs the same evaluation
  logic used by `scripts/run_eval_only.py` (single-GPU/single-process), saves prediction .npy files
  locally, uploads them to the provided GCS output prefix, and creates a small PNG preview for quick
  visual verification.

Before running on Modal:
- Configure a Google Cloud service account JSON and put it into a Modal secret or set
  GOOGLE_APPLICATION_CREDENTIALS in the environment for local testing.
- Ensure the Modal image you use contains a CUDA-enabled PyTorch compatible with the GPU type.
  The README_MODAL.md explains how to set the image and dependencies.
"""

import os
import tempfile
import shutil
import yaml
import pathlib
from typing import List

import numpy as np
import torch
from PIL import Image

import modal
from google.cloud import storage

# Import repo helpers
from pretrain import PretrainConfig, create_dataloader, create_evaluators, init_train_state, evaluate


APP_NAME = "tinyrecursive-eval"
app = modal.App(name=APP_NAME)

# NOTE: The Image below is intentionally minimal. On Modal, pick an official CUDA image or
# construct a Dockerfile that installs the correct torch+cuda wheel for your GPU.
IMAGE = modal.Image.debian_slim().pip_install(
    "pyyaml",
    "numpy",
    "google-cloud-storage",
    "Pillow",
)


def download_gcs_prefix(gcs_uri: str, local_dir: str, client: storage.Client):
    # gcs_uri like: gs://bucket/path/prefix
    assert gcs_uri.startswith("gs://")
    parts = gcs_uri[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        # Skip directories
        if blob.name.endswith('/'):
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        target_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        blob.download_to_filename(target_path)


def upload_file_to_gcs(local_path: str, gcs_uri: str, client: storage.Client):
    assert gcs_uri.startswith("gs://")
    parts = gcs_uri[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    bucket = client.bucket(bucket_name)

    dest_name = prefix.rstrip('/') + '/' + os.path.basename(local_path)
    blob = bucket.blob(dest_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{dest_name}"


@app.function(image=IMAGE, timeout=3600, gpu="A10G")
def run_eval_gcs(checkpoint_gcs: str, dataset_gcs_prefix: str, out_gcs_prefix: str, config_path: str = 'config/cfg_pretrain.yaml'):
    """Download checkpoint + dataset from GCS, run evaluation, upload preds and preview PNG.

    Args:
      checkpoint_gcs: full GCS URI to checkpoint file (gs://bucket/path/step_50000)
      dataset_gcs_prefix: prefix to dataset/test files (gs://bucket/datasets/maze-.../test)
      out_gcs_prefix: target output prefix (gs://bucket/preds/...) where results will be uploaded
      config_path: path inside repo to config yaml (defaults to repo config)

    Returns:
      dict with uploaded file URIs and preview path
    """

    # Create GCS client (assumes GOOGLE_APPLICATION_CREDENTIALS set in env or default creds available)
    client = storage.Client()

    tmpdir = tempfile.mkdtemp(prefix="modal_eval_")
    try:
        # 1) Download dataset files
        local_dataset_dir = os.path.join(tmpdir, 'dataset')
        os.makedirs(local_dataset_dir, exist_ok=True)
        print(f"Downloading dataset from {dataset_gcs_prefix} to {local_dataset_dir}")
        download_gcs_prefix(dataset_gcs_prefix, local_dataset_dir, client)

        # 2) Download checkpoint
        local_checkpoint = os.path.join(tmpdir, os.path.basename(checkpoint_gcs))
        print(f"Downloading checkpoint {checkpoint_gcs} to {local_checkpoint}")
        # direct download of single blob
        parts = checkpoint_gcs[5:].split('/', 1)
        bucket_name = parts[0]
        blob_name = parts[1]
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_checkpoint)

        # 3) Load config YAML and override test dataset and checkpoint path to local dirs
        with open(config_path, 'rt') as f:
            cfg = yaml.safe_load(f)

        cfg['data_paths_test'] = [local_dataset_dir]
        cfg['load_checkpoint'] = local_checkpoint
        # Set checkpoint_path (for outputs) to a local folder that will be uploaded
        local_out = os.path.join(tmpdir, 'out')
        os.makedirs(local_out, exist_ok=True)
        cfg['checkpoint_path'] = local_out
        # Ensure preds are saved
        cfg['eval_save_outputs'] = ['inputs', 'labels', 'puzzle_identifiers', 'preds']

        config_obj = PretrainConfig(**cfg)

        # Set random seed
        torch.random.manual_seed(config_obj.seed)

        # Create dataloader
        train_loader, train_metadata = create_dataloader(config_obj, 'train', rank=0, world_size=1, test_set_mode=False, epochs_per_iter=1, global_batch_size=config_obj.global_batch_size)
        try:
            eval_loader, eval_metadata = create_dataloader(config_obj, 'test', rank=0, world_size=1, test_set_mode=True, epochs_per_iter=1, global_batch_size=config_obj.global_batch_size)
        except Exception as e:
            print('NO EVAL DATA FOUND', e)
            return {'error': 'no eval data'}

        evaluators = []
        try:
            evaluators = create_evaluators(config_obj, eval_metadata)
        except Exception:
            evaluators = []

        # Init model & train_state (loads checkpoint inside create_model through init_train_state)
        train_state = init_train_state(config_obj, eval_metadata, rank=0, world_size=1)

        # Run evaluation (this will save preds to config_obj.checkpoint_path)
        train_state.model.eval()
        metrics = evaluate(
            config=config_obj,
            train_state=train_state,
            eval_loader=eval_loader,
            eval_metadata=eval_metadata,
            evaluators=evaluators,
            rank=0,
            world_size=1,
            cpu_group=None,
        )

        print('Evaluation completed. Metrics:', metrics)

        # 4) Upload results
        uploaded = {}
        # Upload all files in local_out
        for root, _, files in os.walk(local_out):
            for fname in files:
                local_path = os.path.join(root, fname)
                # create a GCS destination path under out_gcs_prefix maintaining subdirs
                rel = os.path.relpath(local_path, local_out)
                dest_uri = out_gcs_prefix.rstrip('/') + '/' + rel
                # Ensure dest folder exists; upload
                # Here we'll use a simple helper: upload file to folder (preserve filename)
                uploaded_uri = upload_file_to_gcs(local_path, out_gcs_prefix.rstrip('/') + '/', client)
                uploaded[fname] = uploaded_uri

        # 5) Create a small PNG preview of the first predicted grid (if preds available)
        preview_uri = None
        preds_npy = os.path.join(local_out, 'preds.npy')
        if os.path.exists(preds_npy):
            preds = np.load(preds_npy)
            # preds shape likely (N, seqLen) — render first example
            arr = preds[0]
            seq_len = arr.shape[0]
            side = int(np.sqrt(seq_len))
            grid = arr.reshape((side, side)).astype(np.uint8)
            # Map small integers to colors (greyscale)
            img = Image.fromarray((grid * (255 // (grid.max() + 1))).astype(np.uint8), mode='L')
            preview_local = os.path.join(local_out, 'preview.png')
            img.save(preview_local)
            preview_uri = upload_file_to_gcs(preview_local, out_gcs_prefix.rstrip('/') + '/', client)

        return {'uploaded': uploaded, 'preview': preview_uri, 'metrics': metrics}

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


@app.local_entrypoint()
def main(checkpoint_gcs: str, dataset_gcs_prefix: str, out_gcs_prefix: str):
    # Call remote function
    print('Launching remote evaluation on Modal...')
    result = run_eval_gcs.remote(checkpoint_gcs, dataset_gcs_prefix, out_gcs_prefix)
    print('Remote invocation submitted. Result (may block until completion):')
    print(result)
