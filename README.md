# Less is More: Recursive Reasoning with Tiny Networks

This is the codebase for the paper: "Less is More: Recursive Reasoning with Tiny Networks". TRM is a recursive reasoning approach that achieves amazing scores on Maze-Hard puzzles using a tiny 7M parameters neural network.

[Paper](https://arxiv.org/abs/2510.04871)

### Motivation

Tiny Recursion Model (TRM) is a recursive reasoning model that achieves amazing scores on hard Maze-Hard puzzles with a tiny 7M parameters neural network. The idea that one must rely on massive foundational models trained for millions of dollars by some big corporation in order to achieve success on hard tasks is a trap. Currently, there is too much focus on exploiting LLMs rather than devising and expanding new lines of direction. With recursive reasoning, it turns out that "less is more": you don't always need to crank up model size in order for a model to reason and solve hard problems. A tiny model pretrained from scratch, recursing on itself and updating its answers over time, can achieve a lot without breaking the bank.

This work came to be after I learned about the recent innovative Hierarchical Reasoning Model (HRM). I was amazed that an approach using small models could do so well on hard tasks like the ARC-AGI competition (reaching 40% accuracy when normally only Large Language Models could compete). But I kept thinking that it is too complicated, relying too much on biological arguments about the human brain, and that this recursive reasoning process could be greatly simplified and improved. Tiny Recursion Model (TRM) simplifies recursive reasoning to its core essence, which ultimately has nothing to do with the human brain, does not require any mathematical (fixed-point) theorem, nor any hierarchy.

### How TRM works

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) recursively improves its predicted answer y with a tiny network. It starts with the embedded input question x and initial embedded answer y and latent z. For up to K improvements steps, it tries to improve its answer y. It does so by i) recursively updating n times its latent z given the question x, current answer y, and current latent z (recursive reasoning), and then ii) updating its answer y given the current answer y and current latent z. This recursive process allows the model to progressively improve its answer (potentially addressing any errors from its previous answer) in an extremely parameter-efficient manner while minimizing overfitting.

### Requirements

- Modal account (for cloud computing)
- Local machine for Modal CLI

### Setup (Everything on Modal)

1. **Clone Repository Locally** (for Modal mounting):
```bash
git clone https://github.com/YuvrajSingh-mist/TinyRecursiveModels.git
cd TinyRecursiveModels
```

2. **Modal Setup**:
```bash
pip install modal
modal setup
```

3. **Prepare Dataset on Modal**:
```bash
modal run infra/modal_app.py::prepare_dataset
```

4. **Download Weights on Modal**:
```bash
modal run infra/modal_app.py::download_weights
```

5. **Run Training on Modal**:
```bash
modal run infra/modal_app.py::run_training
```

6. **Run Evaluation on Modal**:
```bash
modal run infra/modal_app.py::run_eval_local --checkpoint-path data/maze-30x30-hard-1k/step_50000 --dataset-path data/maze-30x30-hard-1k/test
```

7. **Launch Web Interface**:
```bash
modal serve infra/modal_app.py
```
Visit the provided URL to access the maze visualizer with prediction capabilities.

## Modal Cloud Operations

All computation runs on Modal's cloud infrastructure with powerful GPUs:

### Available Endpoints
- `POST /prepare_dataset`: Prepare the maze dataset
- `POST /download_weights`: Download pre-trained weights from Hugging Face  
- `POST /run_training`: Run full training pipeline (4 A100 GPUs)
- `POST /run_eval_local`: Run evaluation with custom paths (2 A100 GPUs)
- `GET /get_visualizer`: Serve maze visualizer HTML
- `GET /get_asset?filename=<file>`: Serve static assets
- `GET /hello`: Simple hello endpoint
- `POST /predict`: Predict solved maze from input data

### Web Interface
```bash
modal serve infra/modal_app.py
```
Visit the URL to visualize mazes and get AI-powered predictions.

If you find our work useful, please consider citing:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

and the Hierarchical Reasoning Model (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Hierarchical Reasoning Model Analysis [code](https://github.com/arcprize/hierarchical-reasoning-model-analysis).

## Multi-GPU Inference on Modal

The project supports multi-GPU inference on Modal.com for scalable evaluation.

### Setup Modal

1. Install Modal CLI:
```bash
pip install modal
modal setup
```

### Running Multi-GPU Evaluation

```bash
modal run infra/modal_app.py::run_eval_local --checkpoint-path data/maze-30x30-hard-1k/step_50000 --dataset-path data/maze-30x30-hard-1k/test
```

This will run evaluation on 2 A100 GPUs using torchrun.

### Deploying the FastAPI Web App

The Modal app includes FastAPI endpoints for the puzzle visualizer and predictions.

#### Temporary Development Server
```bash
modal serve infra/modal_app.py
```

This starts a temporary server. Visit the printed URL to access the visualizer and API docs at `/docs`.

#### Permanent Deployment
```bash
modal deploy infra/modal_app.py
```

This deploys the app permanently. The endpoints will be available at the deployed URL.

#### Available Endpoints
- `GET /get_visualizer`: Serves the puzzle visualizer HTML
- `GET /get_asset?filename=npyjs.js`: Serves static assets
- `GET /hello`: Simple hello message
- `POST /predict`: Accepts puzzle data as JSON and returns solved maze (currently dummy implementation)

Integrate the predict endpoint into the visualizer's JavaScript to call the API when the predict button is clicked. The endpoint accepts POST requests with puzzle data and returns the solved maze.
