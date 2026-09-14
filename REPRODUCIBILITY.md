# Reproduction and Release Details

## Purpose

This guide documents the release layout, reproduction scope, validation coverage, downloads, and exclusions. For an introduction to the paper and its four diagnostics, see the [README](README.md).

Run the commands below from the repository root. Follow each diagnostic's linked README for its dependencies and data setup.

Third-party notices for selected helper code, custom asset files, and DINOv2-derived downloaded checkpoint payloads are recorded in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).

## Layout

```text
.
├── shortcut_solvability/
├── statistical_significance/
├── creeping_overfitting/
├── data_source_dependency/
├── leaderboards/
├── provenance/
├── scripts/
├── analysis/
├── CLAIMS.md
├── SHA256SUMS
├── public_manifest.json
├── LICENSE
└── README.md
```

## Included Diagnostics

1. `shortcut_solvability/`: LIBERO and CALVIN DINO+MLP/task-id shortcut-solvability summaries, configs, compact per-trial/per-sequence outcomes, and first-party training/evaluation code for the released shortcut policies.
2. `statistical_significance/`: LIBERO Goal five-policy `5k` shared-instance outcome rows, policy summaries, pairwise-disagreement summary, aggregate leaderboard significance-category CSVs, cutoff reference code, and the exact shared LIBERO Goal init-state tree.
3. `creeping_overfitting/`: SimplerEnv fixed-grid and Protocol A-E rows/summaries, CALVIN resampled-pose and fresh-sequence rows/summaries, LIBERO Layer 2 rows/summaries, exact CALVIN/LIBERO reset/init-state inputs, SimplerEnv Protocol A-E configs/assets, and support validation code.
4. `data_source_dependency/`: scripted-demo WidowX data-source-dependency summaries, official `4 x 24` grid trial outcomes, and first-party scripted collection, replay, dataset-validation, training, and official-eval code.
5. `leaderboards/`: copied public leaderboard CSV snapshots and the Sam official-protocol previous-SOTA exports used for the significance category tables.
6. `provenance/`: best-effort package/environment provenance and checkpoint identity manifests, with unrecoverable exact fields marked unknown.
7. `analysis/`: CPU regeneration path for selected frozen paper figures/tables and release-current analysis CSVs.

## Reproduction Levels

1. CPU claim and analysis regeneration is included in this release. Run `python scripts/recompute_claims.py` to recompute public headline numbers from released CSV/JSON/YAML files. For the selected frozen paper CSV/SVG/TeX artifacts, release-current CSVs, and the intentional `92`-row paper/current significance comparison, use the commands in [`analysis/README.md`](analysis/README.md).
2. Own-policy training code is included for `data_source_dependency/` and `shortcut_solvability/`. These commands require the diagnostic dependencies plus the named external datasets, checkpoints, benchmark packages, and DINOv2 caches. DSD has tested archive extraction commands, a real-DINO CPU optimizer/save/load smoke on a real stack NPZ, one real stack saved-demo replay, and one actual `step6500` stack checkpoint eval episode; this is not a full `96`-episode replay/eval reproduction. Shortcut has CPU optimizer/save/load/inference smokes, strict loading for all seven actual packaged checkpoints, exact original-vs-release `_encode` and full-forward equality for one actual LIBERO and one actual CALVIN checkpoint, and exact seeded augmentation equality. See [`data_source_dependency/README.md`](data_source_dependency/README.md) and [`shortcut_solvability/README.md`](shortcut_solvability/README.md) for runnable setup and archive extraction commands.
3. Third-party policy evaluations are released as artifacts plus the custom changes needed to identify the evaluated setup: compact rows/summaries, configs, adapters or patches, small custom assets, and exact reset/init-state files where those define the benchmark condition. Full external checkpoints and large datasets remain outside Git, either in upstream model/data locations or in the single Google Drive payload folder named below.

## External Payloads

Large payloads use one public Google Drive folder instead of being committed to Git: <https://drive.google.com/drive/folders/1ZYpEvdD1cf6JSLQiSu7hK0xahSNRvS9F>. The child folders are:

1. `datasets/`: <https://drive.google.com/drive/folders/1HGLUAxL4STXZ10091RFoha8ZrAn2WVQN>
2. `checkpoints/`: <https://drive.google.com/drive/folders/1S6M3l1FfFwUYHm5tsFXBs9p_K0BJsdqj>
3. `evaluation_inputs/`: <https://drive.google.com/drive/folders/1kUz0lVw-bgljb-oKRW8sQVCA83fajSqs>

These links allow anyone with the link to view and download the artifacts. Small result tables and source/config files stay in Git. `evaluation_inputs.tar` is a convenience mirror of the committed exact-input configs/manifests/assets for Drive-based downloads, and checkpoint archives include basic metadata next to selected weights that are intentionally outside Git.

## Validation

The lightweight aggregate checks are:

```bash
python scripts/recompute_claims.py
python scripts/validate_release.py --skip-exact-inputs
sha256sum -c SHA256SUMS
```

The exact-input check also validates Torch-loaded LIBERO `.pruned_init` payload contents, so run it in a CPU environment with NumPy and PyTorch available:

```bash
python creeping_overfitting/code/validate_exact_inputs.py
python scripts/validate_release.py
```

The selected paper/current analysis artifacts regenerate from committed files; use [`analysis/README.md`](analysis/README.md) for the required output directories, optional figure/table outputs, and the expected `compare-paper` status `1` with `92` labeled historical/current differences. [`creeping_overfitting/README.md`](creeping_overfitting/README.md), [`data_source_dependency/README.md`](data_source_dependency/README.md), and [`shortcut_solvability/README.md`](shortcut_solvability/README.md) describe diagnostic-specific runtime setup and limits.

`CLAIMS.md` maps each public claim to the files used for recomputation. `public_manifest.json` records the release policy, source candidates, artifact groups, validation status, and intentional exclusions.

## Exclusions

The Git repository intentionally excludes model weights, large datasets, rollout videos, full rollout directories, full observations, per-step action traces, simulator caches, conda environments, containers, raw logs, third-party source checkouts, Git metadata, browser state, credential files, and credential material. Selected datasets/checkpoints live in the public Drive payload folder. Checkpoint identity metadata is included in Git without binary payloads. The included small reset/init-state artifacts and custom assets are the exact files named by the validator when they define a public evaluation condition. Private paths, hostnames, job IDs, and W&B links are not release blockers by policy if credential-clean, but this package keeps them minimal.
