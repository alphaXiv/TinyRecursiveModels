import json as _json

from .modal_base import app
from .eval_endpoints import (
    prepare_dataset_job,
    run_eval_maze_job,
    run_eval_sudoku_job,
)
from .predict_endpoints import predict_job


@app.local_entrypoint()
def cli_prepare_dataset(include_maze: bool = True,
                        include_sudoku: bool = False,
                        include_arc: bool = True,
                        sudoku_output_dir: str = "data/sudoku-extreme-1k-aug-1000",
                        sudoku_subsample_size: int = 1000,
                        sudoku_num_aug: int = 1000):
    res = prepare_dataset_job.remote(
        include_maze=include_maze,
        include_sudoku=include_sudoku,
        include_arc=include_arc,
        sudoku_output_dir=sudoku_output_dir,
        sudoku_subsample_size=sudoku_subsample_size,
        sudoku_num_aug=sudoku_num_aug,
    )
    print(res)


@app.local_entrypoint()
def cli_run_eval_maze(batch_size: int = 256,
                      one_batch: bool = False,
                      eval_only: bool = True,
                      checkpoint: str | None = None,
                      model: str | None = None):
    _ = eval_only
    _ = model
    res = run_eval_maze_job.remote(batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)  # type: ignore
    print(res)


@app.local_entrypoint()
def cli_run_eval_sudoku(model: str = "mlp",
                        dataset_path: str | None = None,
                        batch_size: int = 256,
                        one_batch: bool = False,
                        eval_only: bool = True,
                        checkpoint: str | None = None):
    _ = eval_only
    res = run_eval_sudoku_job.remote(model=model, dataset_path=dataset_path, batch_size=batch_size, one_batch=one_batch, checkpoint=checkpoint)  # type: ignore
    print(res)


@app.local_entrypoint()
def cli_predict(task: str = "maze",
                model: str | None = None,
                run: str | None = None,
                file: str | None = None,
                index: int | None = None,
                grid_json: str | None = None,
                grid_file: str | None = None):
    grid = None
    if grid_json:
        grid = _json.loads(grid_json)
    elif grid_file:
        with open(grid_file, "r", encoding="utf-8") as f:
            grid = _json.load(f)
    res = predict_job.remote(grid=grid, index=index, file=file, task=task, model=model, run=run)  # type: ignore
    print(res)
