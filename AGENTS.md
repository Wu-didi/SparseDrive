# Repository Guidelines

## Project Structure & Module Organization
Implementation builds on mmdetection3d. `projects/mmdet3d_plugin/models` hosts SparseDrive perception, motion, planning, and map heads; `datasets/` provides nuScenes loaders/pipelines; `ops/` contains deformable CUDA that must be compiled. Stage-specific configs reside in `projects/configs`, generic tooling under `tools/`, and runnable wrappers (data creation, train/test/vis) under `scripts/`. Keep symlinked datasets and derived artifacts in `data/`, while `docs/` and `resources/` store figures.

## Build, Test, and Development Commands
- `conda create -n sparsedrive python=3.8` and `pip install -r requirement.txt` (plus the pinned CUDA 11.6 PyTorch wheel) bootstrap the environment.
- Compile ops once per machine: `cd projects/mmdet3d_plugin/ops && python3 setup.py develop`.
- Generate nuScenes metadata with `sh scripts/create_data.sh`; run `sh scripts/kmeans.sh` whenever you alter class anchors (outputs land in `data/kmeans` and `vis/kmeans`).
- Stage 2 training is the default inside `sh scripts/train.sh`; call `bash tools/dist_train.sh projects/configs/sparsedrive_small_stage1.py <gpus> --deterministic` when pretraining perception only.
- Evaluate and visualize via `bash tools/dist_test.sh <cfg> <ckpt> <gpus> --deterministic --eval bbox` followed by `sh scripts/visualize.sh --result-path work_dirs/<cfg>/results.pkl`.

## Coding Style & Naming Conventions
Use 4-space indentation, limit lines to ~120 chars, and run `yapf -i <file>` before committing. Functions and modules stay `snake_case`, classes `CamelCase`, and configs follow `sparsedrive_<variant>_stage<idx>.py`. Register new blocks with the proper MMDetection3D registries (`@DETECTORS.register_module`, etc.) and keep docstrings concise but parameter-aware.

## Testing Guidelines
Distributed evaluation (`bash tools/dist_test.sh ... --eval bbox`) is required for any change touching perception, mapping, or planning; append additional metrics like `--eval bbox_map planning` when comparing to paper tables. Update the checkpoint path inside `scripts/test.sh` before running `sh scripts/test.sh` for convenience jobs, and archive the resulting metrics/logs in `work_dirs/<cfg>/`. For fast sanity checks (e.g., CUDA kernels or tensor shapes), prefer `python tools/test.py <cfg> <ckpt>` or small harnesses such as `python test_fun.py`.

## Commit & Pull Request Guidelines
History favors short imperative summaries (`add rssm`, `edit data loader`), so describe the change in one concise line and mention the subsystem touched. Every PR should list the exact commands run (train/test scripts, dataset converters), link relevant issues, and paste the latest metrics or qualitative evidence produced from `work_dirs/`. Flag breaking config changes, include screenshots when visualizers change, and confirm formatting/tests in the description.

## Security & Configuration Tips
Avoid committing nuScenes assets or proprietary checkpoints—reference `data/` or `ckpt/` paths instead. Scripts assume `$PYTHONPATH` includes the repo root; when launching notebooks or new CLIs, export it explicitly and strip any absolute paths from configs before sharing.
