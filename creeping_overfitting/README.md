# Creeping Overfitting

## Purpose

This directory contains lightweight public evidence for the creeping-overfitting diagnostic, covering distribution perturbation checks and fresh-test-sample checks.

## Contents

1. `results/simplerenv/fixed_grid_calibration/`: fixed-grid calibration per-episode rows, policy/task summaries, and validation report.
2. `results/simplerenv/distribution_overfitting/`: Protocol A-E per-episode rows, policy/condition summaries, test-set manifest, and validation report.
3. `results/calvin/`: CALVIN resampled-pose and fresh-sequence per-sequence rows, summary CSVs, confidence intervals, and combined summary JSON.
4. `results/libero/`: LIBERO Layer 2 fresh-init-state and selected official-calibration summary CSVs.
5. `results/calvin/manifests/` and `results/calvin/reset_banks/`: exact CALVIN official/fresh sequence manifests and the Protocol 1 reset bank used by the released rows.
6. `artifacts/libero/`: exact LIBERO fresh four-suite init-state tree used by the sample-overfitting rollouts.
7. `configs/` and `artifacts/simplerenv/`: the frozen SimplerEnv Protocol A-E config plus accepted custom block assets.
8. `code/`: selected support scripts/loaders for the custom CALVIN, LIBERO, and SimplerEnv exact-input artifacts.

## Notes

1. CALVIN fresh-sequence rows are included and recomputed here. Result provenance is verified at `Eval_Policies_CoRL` commit `eba7c0037294557427a7a854c91be56d3f2838ec`.
2. LIBERO Layer 2 now includes sanitized selected per-episode outcome rows for the final official-calibration and fresh-init-state runs. The fresh rows include the documented Spatial Forcing retry replacements for `libero_spatial/task_1` and `libero_goal/task_5`.
3. Large third-party repositories, weights, videos, full observations, action traces, logs, datasets, caches, and environments remain excluded. Small binary exact-input artifacts are included where they are the artifact identity itself.
4. `code/validate_exact_inputs.py` validates the released exact inputs without requiring the full policy/simulator environments.
