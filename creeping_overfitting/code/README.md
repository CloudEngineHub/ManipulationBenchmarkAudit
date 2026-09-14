# Creeping Overfitting Support Code

## Purpose

This directory contains selected first-party support scripts for the custom evaluation inputs released in `creeping_overfitting/`. These scripts document how the exact CALVIN, LIBERO, and SimplerEnv artifacts were generated, loaded, or validated without vendoring full third-party policy repositories or runnable model environments.

## Contents

1. `calvin/`: sequence-manifest generation, Protocol 1 reset-bank construction, and reset override helpers.
2. `libero/`: fresh init-state generation/merge helpers and episode-CSV validation.
3. `simplerenv/`: Protocol A-E config/runtime semantics, accepted asset staging, and the actual `widowx_protocol1.py` custom reset/instruction logic.
4. `validate_exact_inputs.py`: release-side validation for the exact input artifacts and selected LIBERO outcome rows.

## Runtime Notes

1. The CALVIN scripts are artifact-level support code. They require the external CALVIN evaluation stack (`calvin_agent`, Hydra, and OmegaConf) when regenerating reset banks; the copied first-party companion `calvin_sequence_manifest.py` is included in the same directory.
2. The LIBERO generation/merge scripts require an installed LIBERO stack plus Torch and tqdm. The release validator uses Torch to load each `.pruned_init` file and check the actual state counts against the manifests.
3. The SimplerEnv scripts document the accepted custom semantics. `stage_protocol_assets.py` imports the sibling `protocol_abcde_common.py`; that common module defaults `PROJECT_ROOT` to the original private project path unless `PROJECT_ROOT` or explicit config paths are provided. `stage_protocol_assets.py` also retains the original InternVLA-M1 asset staging path used during private packaging. The reusable accepted assets are released under `creeping_overfitting/artifacts/simplerenv/protocol_abcde/assets/`.
4. `widowx_protocol1.py` is the actual custom reset/instruction environment code and still depends on the SimplerEnv runtime packages (`sapien`, `transforms3d`, and `mani_skill2_real2sim`) for execution.
