# Shortcut Solvability

## Purpose

This directory contains the public shortcut-solvability release for the benchmark audit paper: compact result artifacts plus minimal first-party code for training and evaluating the fixed-instruction DINOv2/task-id policies on LIBERO and CALVIN.

## Contents

1. `results/libero/results.csv` and `results/libero/best_checkpoint/*/{summary.json,trials.csv}` record the four LIBERO official 50-trial-per-task cells: Spatial `495/500 = 99.0%`, Object `500/500 = 100.0%`, Goal `494/500 = 98.8%`, and Long/LIBERO-10 `462/500 = 92.4%`.
2. `results/calvin/results.csv` and `results/calvin/best_checkpoint/*/{summary.json,trials.csv}` record the three CALVIN cells. `D -> D` and `ABCD -> D` have complete official-result bundles; `ABC -> D` preserves the measured `1000`-sequence result but its historical full artifact manifest was not recovered.
3. `results/calvin/official_1000_eval_sequences.json` is the official CALVIN sequence list used by these runs.
4. `configs/libero/*.yaml` and `configs/calvin/*.yaml` are the selected final training/evaluation configs for the released cells.
5. `code/shortcut_policy/` contains the runnable loaders, model, training commands, evaluation commands, and CALVIN public-data preparation command.

## Upstream Inputs

1. LIBERO uses the official LIBERO benchmark code and official raw HDF5 demos under `LIBERO/libero/datasets/<suite>/*_demo.hdf5`. The released configs use `dataset_type: original`.
2. CALVIN uses official split directories such as `<DATA_ROOT>/task_D_D/{training,validation}`, `<DATA_ROOT>/task_ABC_D/{training,validation}`, and `<DATA_ROOT>/task_ABCD_D/{training,validation}`. Each split must contain `episode_*.npz`, `ep_start_end_ids.npy`, and `lang_annotations/auto_lang_ann.npy`.
3. The visual backbone is DINOv2 ViT-B/14. Put the DINOv2 torch-hub checkout at `$TORCH_HOME/hub/facebookresearch_dinov2_main`; torch hub will look for the usual DINOv2 checkpoint cache under `$TORCH_HOME/hub/checkpoints/`.
4. Binary policy checkpoints are intentionally excluded from Git. A separate weight package should use the layout below.

```text
<CHECKPOINT_ROOT>/
|-- shortcut_libero/
|   |-- libero_spatial/step13425.pt
|   |-- libero_object/step16110.pt
|   |-- libero_goal/step5370.pt
|   `-- libero_10/step15215.pt
`-- shortcut_calvin/
    |-- calvin_D_D/step40000.pt
    |-- calvin_ABCD_D/step100000.pt
    `-- calvin_ABC_D_supporting_incomplete_provenance/step100000.pt
```

The checkpoint archives are laid out for direct extraction:

```bash
mkdir -p /path/to/checkpoints
tar -xf shortcut_libero.tar -C /path/to/checkpoints
tar -xf shortcut_calvin.tar -C /path/to/checkpoints
```

## Setup

Run commands from the repository root. Keep caches and outputs in scratch, not in the source checkout.

```bash
export PYTHONDONTWRITEBYTECODE=1
export TORCH_HOME=/path/to/scratch/torch
export XDG_CACHE_HOME=/path/to/scratch/xdg
```

For a clean CPU dependency target suitable for config checks and synthetic train/reload smokes:

```bash
python -m pip install --target /path/to/scratch/pydeps \
  torch torchvision numpy pyyaml h5py
export PYTHONPATH=/path/to/scratch/pydeps:${PYTHONPATH}
```

Full benchmark training/evaluation also needs the upstream benchmark packages:

1. LIBERO: <https://github.com/Lifelong-Robot-Learning/LIBERO>
2. CALVIN: <https://github.com/mees/calvin>
3. DINOv2: <https://github.com/facebookresearch/dinov2>

LIBERO eval additionally needs LIBERO/robosuite rendering set up. CALVIN eval additionally needs CALVIN `calvin_env`, CALVIN `calvin_models`, Hydra/OmegaConf, and `pyhash`.

The released benchmark configs use DINOv2 (`vision_backbone: dino`, `dino_model: dinov2_vitb14`) and require the DINOv2 checkout/cache described above. For local CPU-only smoke tests, use `vision_backbone: resnet18` with `pretrained_visual: false` or a scratch-only DINO stub.

## LIBERO Commands

The released LIBERO configs set `dataset_dir: LIBERO/libero/datasets`, which is resolved relative to the process working directory used for the command. `--data-root` only expands literal `<DATA_ROOT>` placeholders, and these LIBERO configs do not contain that placeholder. To keep official demos elsewhere, copy the selected YAML to a scratch location, set `dataset_dir` in that copy to the absolute directory that contains the suite folders such as `libero_spatial/`, `libero_object/`, `libero_goal/`, and `libero_10/`, and pass the copied config to `train`.

Inspect a selected config:

```bash
python -m shortcut_solvability.code.shortcut_policy.libero inspect-config \
  shortcut_solvability/configs/libero/libero_spatial_step13425.yaml
```

Train from official HDF5 demos:

```bash
python -m shortcut_solvability.code.shortcut_policy.libero train \
  shortcut_solvability/configs/libero/libero_spatial_step13425.yaml \
  --checkpoint-root /path/to/new_train_outputs \
  --output-root /path/to/outputs \
  --device cuda
```

Evaluate a packaged checkpoint:

```bash
python -m shortcut_solvability.code.shortcut_policy.libero eval \
  shortcut_solvability/configs/libero/libero_spatial_step13425.yaml \
  --checkpoint-root /path/to/checkpoints \
  --output-root /path/to/outputs \
  --step 13425 \
  --out-dir /path/to/outputs/libero_spatial_step13425/eval/step13425 \
  --device cuda
```

LIBERO task ids are the official suite task order from the LIBERO benchmark object. The policy conditions on that integer id, not on tokenized language. Action normalization is mean/std over the selected training HDF5 actions, including the gripper dimension, and the stats are stored in each checkpoint.

## CALVIN Data Preparation

Before training CALVIN, prepare the public split once per training split:

```bash
python -m shortcut_solvability.code.shortcut_policy.calvin prepare-data \
  /path/to/calvin_cache/task_D_D/training \
  --stats-max-steps 50000
```

This writes:

1. `states_actions_rel_and_abs.pt`, containing `episode_idx`, `gt_state`, `rel_actions`, and `actions` tensors packed from the per-step `episode_*.npz` files.
2. `normalization_stats.npz`, containing `action_mean_rel`, `action_std_rel`, `action_mean_abs`, `action_std_abs`, `gt_state_mean`, and `gt_state_std`.

Run the same command for `task_ABC_D/training` and `task_ABCD_D/training`. Training fails loudly if these files are missing. The default stats path matches the historical recipe: use the first `50000` step files for normalization stats.

## CALVIN Commands

Inspect a selected config:

```bash
python -m shortcut_solvability.code.shortcut_policy.calvin inspect-config \
  shortcut_solvability/configs/calvin/task_D_D_step40000.yaml
```

Train:

```bash
python -m shortcut_solvability.code.shortcut_policy.calvin train \
  shortcut_solvability/configs/calvin/task_D_D_step40000.yaml \
  --data-root /path/to/calvin_cache \
  --checkpoint-root /path/to/new_train_outputs \
  --output-root /path/to/outputs \
  --device cuda
```

Evaluate on the included official sequence list:

```bash
python -m shortcut_solvability.code.shortcut_policy.calvin eval \
  shortcut_solvability/configs/calvin/task_D_D_step40000.yaml \
  --data-root /path/to/calvin_cache \
  --checkpoint-root /path/to/checkpoints \
  --output-root /path/to/outputs \
  --step 40000 \
  --eval-sequences shortcut_solvability/results/calvin/official_1000_eval_sequences.json \
  --out-dir /path/to/outputs/task_D_D_step40000/eval/step40000 \
  --device cuda
```

CALVIN task vocabulary is the sorted set of the 34 discrete task labels in `training/lang_annotations/auto_lang_ann.npy`; the vocabulary is saved in each checkpoint and reused at eval. The policy uses the task id only. CALVIN action normalization applies mean/std to the first six continuous action dimensions; the gripper is trained as the seventh regression dimension and thresholded to `{-1, 1}` at eval. If proprio is enabled, it uses `[tcp6, gripper_width]`; the released final configs set `use_proprio: false`.

## Caveats

1. These commands are intended to reproduce the shortcut policy recipe when the external benchmark code, data, DINOv2 cache, and separately packaged checkpoints are available. The included historical CSV/JSON files remain the source for the paper numbers.
2. `ABC -> D` is included because the `1000`-sequence measured behavior exists, but its historical full official artifact manifest was incomplete. Preserve that caveat in root claim mappings.
3. This is not a universal policy trainer. It intentionally covers only the LIBERO/CALVIN DINOv2/task-id shortcut setup used for the audit.
