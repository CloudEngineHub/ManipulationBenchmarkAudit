# Data Source Dependency External Artifacts

## Purpose

This file specifies the external dataset and checkpoint layout expected by the runnable scripted-demo data-source-dependency release. The files are intentionally not committed to git.

## Required Download Layout

Unpack the external downloads under `$DSD_DATA_ROOT`:

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

Dataset source identity:

1. `stack`: `scripted_widowx_stack_jitter0025_yaw15_place0_open6_finishopen_tailpad5_5pergrid_20260518T2355Z`.
2. `carrot`: `scripted_widowx_carrot_jitter0025_yaw15_place0_open6_finishopen_tailpad5_5pergrid_yawcanon_20260521T2213Z`.
3. `spoon`: `scripted_widowx_spoon_jitter0025_yaw15_place0_open6_finishopen_tailpad5_5pergrid_20260519T0502Z`.
4. `eggplant`: `scripted_widowx_eggplant_jitter0025_yaw15_place0_open6_finishopen_tailpad5_5pergrid_20260519T0502Z`.

Checkpoint source identity:

1. `stack/step6500.pt`: `stack_scripted_jitter0025_yaw15_open6_tailpad5_padterm_proprio_chunk5_10000_20260519T0130Z`.
2. `carrot/step10000.pt`: `carrot_scripted_jitter0025_yaw15_open6_tailpad5_yawcanon_padterm_proprio_chunk5_10000_20260521T2213Z`.
3. `spoon/step10000.pt`: `spoon_scripted_jitter0025_yaw15_open6_tailpad5_padterm_proprio_chunk5_10000_20260519T0502Z`.
4. `eggplant/step10000.pt`: `eggplant_scripted_jitter0025_yaw15_open6_tailpad5_padterm_proprio_chunk5_10000_20260519T0502Z`.

The dataset package should expose exactly one positive-demo tree per task with `episodes/*.npz` directly below each task directory. The `carrot` tree must be the final yaw-canonicalized dataset.

The public dataset archives use the source-preserving layout `scripted_widowx_TASK/data/TASK/episodes/*.npz`. To produce the runnable merged layout:

```bash
mkdir -p "$DSD_DATA_ROOT/datasets/scripted_widowx_120_per_task"
for task in stack carrot spoon eggplant; do
  tar -xf "scripted_widowx_${task}.tar" \
    -C "$DSD_DATA_ROOT/datasets/scripted_widowx_120_per_task" \
    --strip-components=2 \
    "scripted_widowx_${task}/data/${task}"
done
```

The public checkpoint archive uses `scripted_widowx_checkpoints/TASK/step*.pt`. To produce the runnable checkpoint layout:

```bash
mkdir -p "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp"
tar -xf scripted_widowx.tar \
  -C "$DSD_DATA_ROOT/checkpoints/widowx_vits_mlp" \
  --strip-components=1 \
  scripted_widowx_checkpoints
```
