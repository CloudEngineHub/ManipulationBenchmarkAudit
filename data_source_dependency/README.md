# Data Source Dependency

## Purpose

This directory contains lightweight public evidence for the scripted-demo WidowX data-source-dependency diagnostic on the official SimplerEnv WidowX `4 x 24` grid.

## Contents

1. `configs/widowx_scripted_dsd_official_4x24.yaml`: sanitized policy recipe, task setup, dataset counts, and evaluation protocol.
2. `configs/train/*.yaml`: the four paper training configs, with public data, output, and DINOv2 paths supplied by environment variables.
3. `code/src/data_source_dependency/`: runnable scripted collection, saved-action replay, dataset validation, DINOv2-S+MLP training, and official SimplerEnv WidowX eval code.
4. `results/scripted_widowx/results.csv`: per-task success counts.
5. `results/scripted_widowx/trials.csv`: one row per official grid episode.
6. `results/scripted_widowx/aggregate_summary.json`: overall and per-task public summary.
7. `results/scripted_widowx/task_summaries/*.json`: per-task summary JSON files.
8. `artifact_layout.md`: required external download layout for datasets and selected checkpoints.

## Headline Result

The public files recompute stack `24/24`, carrot `23/24`, spoon `21/24`, eggplant `23/24`, and overall `91/96 = 94.79%`.

The carrot result uses the final yaw-canonicalized scripted dataset and checkpoint from 2026-05-21. Older carrot video/export counts are intentionally not used for this headline number.

## Environment

Use a scratch directory for all data, caches, checkpoints, and outputs. The examples below assume they are run from the repository root.

```bash
export DSD_WORK_ROOT=/path/to/dsd_work
export DSD_DATA_ROOT="$DSD_WORK_ROOT/data"
export DSD_OUTPUT_ROOT="$DSD_WORK_ROOT/runs"
export DSD_CACHE_ROOT="$DSD_WORK_ROOT/cache"
export DSD_DINO_HUB_DIR="$DSD_CACHE_ROOT/hub/facebookresearch_dinov2_main"
export PYTHONPATH="$PWD/data_source_dependency/code/src:${PYTHONPATH:-}"
export XDG_CACHE_HOME="$DSD_CACHE_ROOT/xdg"
export TORCH_HOME="$DSD_CACHE_ROOT/torch"
export MPLCONFIGDIR="$DSD_CACHE_ROOT/matplotlib"
export PYTHONPYCACHEPREFIX="$DSD_CACHE_ROOT/pycache"
mkdir -p "$DSD_DATA_ROOT" "$DSD_OUTPUT_ROOT" "$DSD_CACHE_ROOT/hub"
```

Install PyTorch/torchvision for the target CUDA stack, plus the Python packages in `code/requirements.txt`. The collector, saved-action replay, and official eval also require a working SimplerEnv WidowX installation with SAPIEN rendering. The DINOv2 backbone is loaded with `torch.hub.load(..., source="local")`; place a local `facebookresearch/dinov2` checkout at `$DSD_DINO_HUB_DIR`.

## Downloaded Data Path

After unpacking the external dataset and checkpoint downloads, the runnable layout should be:

```text
$DSD_DATA_ROOT/
  datasets/scripted_widowx_120_per_task/
    stack/episodes/*.npz
    carrot/episodes/*.npz
    spoon/episodes/*.npz
    eggplant/episodes/*.npz
  checkpoints/widowx_vits_mlp/
    stack/step6500.pt
    carrot/step10000.pt
    spoon/step10000.pt
    eggplant/step10000.pt
```

`carrot/episodes` must come from `scripted_widowx_carrot_jitter0025_yaw15_place0_open6_finishopen_tailpad5_5pergrid_yawcanon_20260521T2213Z`.

The dataset archives are laid out as `scripted_widowx_TASK/data/TASK/episodes/*.npz`. Extract them into the merged runnable layout with:

```bash
mkdir -p "$DSD_DATA_ROOT/datasets/scripted_widowx_120_per_task"
for task in stack carrot spoon eggplant; do
  tar -xf "scripted_widowx_${task}.tar" \
    -C "$DSD_DATA_ROOT/datasets/scripted_widowx_120_per_task" \
    --strip-components=2 \
    "scripted_widowx_${task}/data/${task}"
done
```

The checkpoint archive is laid out as `scripted_widowx_checkpoints/TASK/step*.pt`. Extract it with:

```bash
mkdir -p "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp"
tar -xf scripted_widowx.tar \
  -C "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp" \
  --strip-components=1 \
  scripted_widowx_checkpoints
```

## Validate And Replay

Validate the downloaded 120-demo-per-task dataset:

```bash
python -m data_source_dependency.widowx.validate_dataset \
  "$DSD_DATA_ROOT/datasets/scripted_widowx_120_per_task" \
  --tasks stack,carrot,spoon,eggplant \
  --chunk-size 5 \
  --pad-terminal-actions \
  --write-json "$DSD_OUTPUT_ROOT/dataset_summary.json"
```

Replay one saved successful demo per official grid id using the stored reset metadata:

```bash
python -m data_source_dependency.widowx.eval_replay \
  --dataset-root "$DSD_DATA_ROOT/datasets/scripted_widowx_120_per_task" \
  --tasks stack,carrot,spoon,eggplant \
  --episode-start 0 \
  --episode-end 24 \
  --use-saved-reset-metadata \
  --output-dir "$DSD_OUTPUT_ROOT/replay_official_grid"
```

Use `--all-demos` to replay all 480 downloaded demos.

## Train

The paper training configs use DINOv2-S, the 7D `actions_env` simulator action, dataset-standard normalization over the first 6 action dimensions, proprio enabled without proprio normalization, `chunk_size=5`, and `pad_terminal_actions=true`.

```bash
python -m data_source_dependency.widowx.train_vits_mlp \
  data_source_dependency/configs/train/train_stack_small_open6_tailpad5_padterm_20260519T0130Z.yaml
python -m data_source_dependency.widowx.train_vits_mlp \
  data_source_dependency/configs/train/train_carrot_yawcanon_padterm_20260521T2213Z.yaml
python -m data_source_dependency.widowx.train_vits_mlp \
  data_source_dependency/configs/train/train_spoon_small_open6_tailpad5_padterm_20260519T0502Z.yaml
python -m data_source_dependency.widowx.train_vits_mlp \
  data_source_dependency/configs/train/train_eggplant_small_open6_tailpad5_padterm_20260519T0502Z.yaml
```

For a quick wiring check, pass `--max-steps 1 --batch-size 2 --num-workers 0 --limit-episodes-per-task 1 --output-dir "$DSD_OUTPUT_ROOT/smoke_train_stack"` to the stack command. Full paper training used `max_steps=10000` and should be run on a CUDA GPU.

## Official Eval

Evaluate the selected checkpoints on the official grid:

```bash
python -m data_source_dependency.widowx.eval_official \
  --checkpoint "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp/stack/step6500.pt" \
  --tasks stack \
  --episode-start 0 \
  --episode-end 24 \
  --replan-steps 5 \
  --output-dir "$DSD_OUTPUT_ROOT/eval/stack_step6500_replan5"
python -m data_source_dependency.widowx.eval_official \
  --checkpoint "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp/carrot/step10000.pt" \
  --tasks carrot \
  --episode-start 0 \
  --episode-end 24 \
  --replan-steps 5 \
  --output-dir "$DSD_OUTPUT_ROOT/eval/carrot_step10000_replan5"
python -m data_source_dependency.widowx.eval_official \
  --checkpoint "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp/spoon/step10000.pt" \
  --tasks spoon \
  --episode-start 0 \
  --episode-end 24 \
  --replan-steps 5 \
  --output-dir "$DSD_OUTPUT_ROOT/eval/spoon_step10000_replan5"
python -m data_source_dependency.widowx.eval_official \
  --checkpoint "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp/eggplant/step10000.pt" \
  --tasks eggplant \
  --episode-start 0 \
  --episode-end 24 \
  --replan-steps 5 \
  --output-dir "$DSD_OUTPUT_ROOT/eval/eggplant_step10000_replan5"
```

The eval code honors `DSD_DINO_HUB_DIR`, so packaged checkpoints that remember the internal training cache can still run from a public scratch layout.

## Regenerate Scripted Demos

Regenerate the 120-demo-per-task scripted dataset in the same merged layout:

```bash
python -m data_source_dependency.collectors.scripted_widowx_stack --task-key stack --output-root "$DSD_DATA_ROOT/datasets" --run-tag scripted_widowx_120_per_task --target-successes-per-grid-id 5 --episode-start 0 --episode-end 24 --max-attempts-per-grid-id 60 --object-xy-jitter-m 0.0025 --object-yaw-jitter-deg 15 --place-clearance-m 0.0 --open-steps 6 --finish-after-open --terminal-pad-steps 5 --ignore-env-success-until-script-done --require-script-done-for-success
python -m data_source_dependency.collectors.scripted_widowx_stack --task-key carrot --output-root "$DSD_DATA_ROOT/datasets" --run-tag scripted_widowx_120_per_task --target-successes-per-grid-id 5 --episode-start 0 --episode-end 24 --max-attempts-per-grid-id 30 --object-xy-jitter-m 0.0025 --object-yaw-jitter-deg 15 --place-clearance-m 0.0 --open-steps 6 --finish-after-open --terminal-pad-steps 5 --ignore-env-success-until-script-done --require-script-done-for-success
python -m data_source_dependency.collectors.scripted_widowx_stack --task-key spoon --output-root "$DSD_DATA_ROOT/datasets" --run-tag scripted_widowx_120_per_task --target-successes-per-grid-id 5 --episode-start 0 --episode-end 24 --max-attempts-per-grid-id 30 --object-xy-jitter-m 0.0025 --object-yaw-jitter-deg 15 --place-clearance-m 0.0 --open-steps 6 --finish-after-open --terminal-pad-steps 5 --ignore-env-success-until-script-done --require-script-done-for-success
python -m data_source_dependency.collectors.scripted_widowx_stack --task-key eggplant --output-root "$DSD_DATA_ROOT/datasets" --run-tag scripted_widowx_120_per_task --target-successes-per-grid-id 5 --episode-start 0 --episode-end 24 --max-attempts-per-grid-id 30 --object-xy-jitter-m 0.0025 --object-yaw-jitter-deg 15 --place-clearance-m 0.0 --open-steps 6 --finish-after-open --terminal-pad-steps 5 --ignore-env-success-until-script-done --require-script-done-for-success
```

The non-stack tasks align the gripper yaw to the source object and canonicalize parallel-jaw yaw modulo pi before saving actions.

## Exclusions

Datasets, checkpoints, videos, full training/evaluation directories, caches, raw logs, and credential-bearing files are intentionally excluded.
