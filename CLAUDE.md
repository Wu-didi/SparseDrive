# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SparseDrive is an end-to-end autonomous driving framework that uses sparse scene representation. It unifies detection, tracking, online mapping, motion prediction, and planning in a single model architecture. Built on top of mmdetection3d and trained on the nuScenes dataset.

Key characteristics:
- Sparse-centric paradigm using instance-level sparse representation
- Symmetric sparse perception for detection/tracking and online mapping
- Parallel motion planner that performs prediction and planning simultaneously
- Instance memory queue for temporal modeling across frames
- Two-stage training: Stage1 (perception only), Stage2 (perception + motion/planning)

## Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n sparsedrive python=3.8 -y
conda activate sparsedrive

# Install PyTorch (CUDA 11.6)
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# Install dependencies
pip3 install -r requirement.txt

# Compile CUDA ops (deformable aggregation)
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
```

### Data Preparation
```bash
# Create symlink to nuScenes dataset (update path accordingly)
mkdir -p data
ln -s /path/to/nuscenes ./data/nuscenes

# Generate dataset info files
sh scripts/create_data.sh

# Generate anchors via K-means clustering
sh scripts/kmeans.sh
```

### Training
```bash
# Stage 1: Train detection, tracking, and mapping only
# Edit scripts/train.sh to uncomment stage1 section
bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage1.py <num_gpus> --deterministic

# Stage 2: Train full model with motion prediction and planning
bash ./tools/dist_train.sh projects/configs/sparsedrive_small_stage2.py <num_gpus> --deterministic

# Or use the convenience script
sh scripts/train.sh
```

### Testing/Evaluation
```bash
# Test a checkpoint
bash ./tools/dist_test.sh \
    projects/configs/sparsedrive_small_stage2.py \
    /path/to/checkpoint.pth \
    <num_gpus> \
    --deterministic \
    --eval bbox

# Or use the convenience script (update checkpoint path in scripts/test.sh first)
sh scripts/test.sh
```

### Visualization
```bash
# Visualize results (requires results.pkl from evaluation)
sh scripts/visualize.sh
# Internally calls: python tools/visualization/visualize.py <config> --result-path <results.pkl>
```

### Single GPU Training/Testing
```bash
# Train on single GPU
python tools/train.py <config_file> [options]

# Test on single GPU
python tools/test.py <config_file> <checkpoint_file> [options]
```

## Architecture Overview

### Main Model (SparseDrive)
- **Location**: `projects/mmdet3d_plugin/models/sparsedrive.py`
- **Components**:
  - Image backbone: ResNet50 with FPN (`img_backbone`, `img_neck`)
  - Grid mask data augmentation (training only)
  - SparseDriveHead: Manages all task-specific modules
  - Instance memory bank for temporal consistency

### Task Modules

**Detection (3D Object Detection)**
- Path: `projects/mmdet3d_plugin/models/detection3d/`
- Key files:
  - `decoder.py`: SparseBox3DDecoder - main detection decoder
  - `detection3d_head.py`: Detection head with instance queries
  - `target.py`: Ground truth assignment for training
- Uses deformable attention and sparse instance queries

**Tracking (Multi-Object Tracking)**
- Integrated with detection module via instance memory queue
- Path: `projects/mmdet3d_plugin/models/instance_bank.py`
- Maintains temporal consistency of instances across frames

**Mapping (Online HD Map Construction)**
- Path: `projects/mmdet3d_plugin/models/map/`
- Key files:
  - `decoder.py`: Map element decoder
  - `map_blocks.py`: Map-specific attention blocks
- Detects lane boundaries, crossings, and dividers

**Motion Prediction**
- Path: `projects/mmdet3d_plugin/models/motion/`
- Key files:
  - `decoder.py`: Motion decoder for trajectory prediction
  - `motion_planning_head.py`: Unified motion and planning head
  - `instance_queue.py`: Temporal instance tracking for motion
- Predicts future trajectories for detected agents (multi-modal)

**Planning (Ego Trajectory Planning)**
- Integrated with motion module (parallel design)
- Uses hierarchical planning selection with collision-aware rescoring
- Outputs safe ego vehicle trajectories

### Core Building Blocks
- **DeformableFeatureAggregation** (`blocks.py`): Multi-scale feature aggregation
- **DenseDepthNet** (`blocks.py`): Depth estimation network
- **AsymmetricFFN** (`blocks.py`): Feed-forward network for transformers
- **Attention modules** (`attention.py`): Various attention mechanisms

### Configuration System
- Configs located in: `projects/configs/`
- Two main configs: `sparsedrive_small_stage1.py`, `sparsedrive_small_stage2.py`
- Stage1: Only detection, tracking, mapping enabled (`with_motion_plan=False`)
- Stage2: All tasks enabled (`with_motion_plan=True`)
- Key parameters:
  - `queue_length=4`: Number of historical frames + current
  - `fut_ts=12`: Future timesteps for motion prediction
  - `ego_fut_ts=6`: Future timesteps for ego planning
  - `fut_mode=6`: Number of trajectory modes

### Dataset
- **Location**: `projects/mmdet3d_plugin/datasets/nuscenes_3d_dataset.py`
- **Pipeline**: `projects/mmdet3d_plugin/datasets/pipelines/`
- nuScenes-specific with CAN bus data for ego vehicle states
- Data converter: `tools/data_converter/nuscenes_converter.py`

### Custom CUDA Operations
- **Path**: `projects/mmdet3d_plugin/ops/`
- Deformable feature aggregation (requires compilation)
- Must run `python setup.py develop` in ops directory before use

## Key Implementation Details

### Two-Stage Training Strategy
1. **Stage 1**: Pretrain perception modules (detection, tracking, mapping)
   - Faster convergence for perception tasks
   - Provides good instance representations for motion/planning
2. **Stage 2**: Fine-tune entire model including motion prediction and planning
   - Loads stage1 checkpoint as initialization
   - Enables end-to-end optimization

### Instance Memory Queue
- Maintains temporal instance embeddings across frames
- Length controlled by `queue_length` parameter
- Critical for tracking and motion prediction
- Located in `instance_bank.py` and `motion/instance_queue.py`

### Parallel Motion Planner
- Motion prediction and ego planning share similar architecture
- Processed in parallel for efficiency
- Collision-aware rescoring selects safe trajectories from multiple modes
- Implementation in `motion/motion_planning_head.py`

### Camera Augmentation
- Grid mask augmentation during training
- Optional random camera masking (RandCamMask in sparsedrive.py)
- Multi-view camera inputs: 6 cameras for nuScenes

## File Organization Patterns

When modifying or adding features:
- **Model components**: Add to `projects/mmdet3d_plugin/models/`
- **Dataset modifications**: Update `projects/mmdet3d_plugin/datasets/`
- **New configs**: Create in `projects/configs/` following existing naming
- **Evaluation metrics**: Check `projects/mmdet3d_plugin/datasets/evaluation/`
- **Training scripts**: Tools in `tools/` directory

## Common Workflows

### Adding a New Module
1. Implement in appropriate subdirectory under `projects/mmdet3d_plugin/models/`
2. Register in corresponding `__init__.py`
3. Add module to config file
4. Update model initialization in `sparsedrive.py` if needed

### Debugging Training
- Check tensorboard logs: `work_dirs/<config_name>/` contains tensorboard events
- Text logs also in work_dirs with training metrics
- Use `--deterministic` flag for reproducible debugging

### Modifying Data Processing
- Data transforms in `projects/mmdet3d_plugin/datasets/pipelines/`
- Update pipeline in config file's `train_pipeline` or `test_pipeline`
- K-means anchors in `data/kmeans/` (regenerate if changing anchor strategy)

## Dependencies

Core frameworks:
- PyTorch 1.13.0 (CUDA 11.6)
- mmcv-full 1.7.1
- mmdet 2.28.2
- nuscenes-devkit 1.1.10
- flash-attn 2.3.2 (for efficient attention)

Key libraries:
- pyquaternion for rotation handling
- motmetrics for tracking evaluation
- scikit-learn for K-means clustering
