# Modal deployment guide (GPU evaluation + GCS storage)

This document shows how to run the model evaluation on Modal using a GPU instance, store inputs/checkpoint on Google Cloud Storage (GCS), and upload prediction outputs and a PNG preview back to GCS. The `modal_app.py` in the repo provides a ready-to-adapt Modal app.

Prerequisites
- A Modal account and the Modal CLI installed and configured. See https://modal.com/docs for setup.
- A Google Cloud project and a service account JSON with `Storage Object Admin` roles (or sufficient permissions to read dataset and write preds).
- `gcloud` and `gsutil` (optional, but handy) installed.
- A GPU-compatible Modal Image (you may need to build a Dockerfile matching your CUDA/Torch version if the built-in image doesn't match your GPU).

Quick overview
1. Build or locate your dataset locally using the repo's dataset builder.
2. Upload dataset files and your checkpoint file to a GCS bucket.
3. Configure Modal to provide GCS credentials to the remote function.
4. Run `modal run modal_app.py --checkpoint_gcs <...> --dataset_gcs_prefix <...> --out_gcs_prefix <...>` to run evaluation.
5. Results (preds `.npy` and a `preview.png`) will be uploaded to the output GCS prefix.

1) Build the dataset locally (example using maze builder)
```bash
# Create dataset (example produces a folder with test/all__*.npy and identifiers.json)
python dataset/build_maze_dataset.py preprocess_data
```

2) Upload dataset and checkpoint to GCS
```bash
# Set these variables
BUCKET=gs://my-bucket
DATASET_LOCAL=./data/maze-30x30-hard-1k
CHECKPOINT_LOCAL=./checkpoints/step_50000

# Upload dataset 'test' folder and identifiers.json
gsutil -m cp -r ${DATASET_LOCAL}/test ${BUCKET}/datasets/maze-30x30-hard-1k/
gsutil cp ${DATASET_LOCAL}/../identifiers.json ${BUCKET}/datasets/maze-30x30-hard-1k/identifiers.json

# Upload checkpoint file
gsutil cp ${CHECKPOINT_LOCAL} ${BUCKET}/checkpoints/step_50000
```

3) Create a Google Cloud service account key and add it to Modal secrets
- Create a service account in GCP and download a JSON key file, e.g. `gcp-key.json`.
- Add the JSON to Modal as a secret or provide it through environment variable `GOOGLE_APPLICATION_CREDENTIALS`.

Example using Modal secrets (CLI):
```bash
# Create a modal secret from the JSON file
modal secret upload gcp_key --from-file gcp-key.json
```

In `modal_app.py` we assume the runtime will see application default credentials (the client library checks `GOOGLE_APPLICATION_CREDENTIALS` or default auth). To pass credentials to Modal, you can add the secret to the function invocation or set env var in the modal Image. See Modal docs for "secrets" usage; a simple pattern is to mount the secret file into the container and set `GOOGLE_APPLICATION_CREDENTIALS` to that path.

4) Run modal function
```bash
modal run modal_app.py --checkpoint_gcs gs://my-bucket/checkpoints/step_50000 --dataset_gcs_prefix gs://my-bucket/datasets/maze-30x30-hard-1k/test --out_gcs_prefix gs://my-bucket/preds/maze-30x30-hard-1k
```

Notes:
- The remote invocation may print progress in the Modal UI and return a JSON-like result with uploaded URIs and a preview image path.
- If the repo model code depends on a specific CUDA/PyTorch wheel (common), you may need to build a custom Modal Image using a Dockerfile that installs the correct torch wheel (see Modal documentation on custom Docker images and GPU support).

5) Visualize predictions
- The `puzzle_visualizer.html` included in this repo is a client-side visualizer that expects a local directory or uploaded folder with `test/all__inputs.npy`, `test/all__labels.npy`, `test/all__preds.npy`, `test/all__puzzle_indices.npy`, `test/all__group_indices.npy`, `test/all__puzzle_identifiers.npy`, and `identifiers.json` at top-level.
- To see results in the browser, you can either:
  - Download the predicted files from GCS to your machine and open `puzzle_visualizer.html` and upload the folder in the page; or
  - Host the visualizer on a small static web server and serve the files from GCS via signed URLs.

Example: download preds locally and open visualizer
```bash
gsutil -m cp -r gs://my-bucket/preds/maze-30x30-hard-1k ./preds_local
open puzzle_visualizer.html
# Use the file upload control in the page to upload the folder ./preds_local/test and identifiers.json
```

Troubleshooting
- If imports fail on Modal (torch not found), you'll need to ensure the Image includes a compatible `torch` wheel for the GPU and CUDA version. The easiest path is to build a Dockerfile that installs the needed wheel and use `modal.Image.from_dockerfile(...)`.
- If GCS authentication fails inside Modal, ensure the service account secret is mounted and `GOOGLE_APPLICATION_CREDENTIALS` points to it.

Next improvements
- Implement an asynchronous job queue in Modal if test inference runs are long; return a job id immediately and poll for completion.
- Generate an index HTML that loads `puzzle_visualizer.html` and points to GCS signed URLs to avoid manual download.

If you want I can:
- create the Dockerfile + Modal Image that pins a torch+cuda wheel matching a target GPU; or
- add code that uploads signed URLs and a small static HTML to GCS so the visualizer can load the predictions directly from the cloud.
