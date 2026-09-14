# Release Analysis

## Purpose

This directory contains the CPU-only analysis regeneration path for selected benchmark-audit paper values, figures, and tables. Put compact scripts, frozen paper plot inputs, and derived CSV outputs here when they are reproducible from files checked into this release. Do not put simulator outputs, private filesystem inputs, caches, bytecode, or historical experiment reruns here.

## Contents

1. `paper_inputs/` contains the eight compact CSV inputs used to regenerate the selected frozen paper plots: seven statistical-significance pie inputs and one SimplerEnv quantitative plot input.
2. `run_analysis.py` has three modes:
   1. `paper-inputs`: regenerate the exact paper-input significance pies, SimplerEnv score-panel data, confidence-interval CSV, SVG figures, and confidence-interval table from committed release files.
   2. `current`: regenerate release-current classifications, SimplerEnv quantitative plot data, confidence intervals, selected SVG figures, the confidence-interval table, and a labeled paper-versus-current bucket comparison table.
   3. `compare-paper`: compare release-current significance classifications with the frozen paper input CSVs, write row-level evidence, and exit nonzero when classifications differ.
3. `release_current_values/` contains committed CSV outputs produced by `current` mode.

The SimplerEnv SVG regenerates the score panel only; the rendered-environment image grid from the paper is not included in this release package.

## Dependencies

CSV and table regeneration use only the Python standard library. Figure rendering additionally requires `matplotlib`.

## Exact Paper Artifact Regeneration

Run from the repository root. This command uses only files checked into the release by default; `analysis/paper_inputs` is the default paper input directory.

```bash
ANALYSIS_TMP="${ANALYSIS_TMP:-../benchmark_release_analysis_scratch}"
mkdir -p "$ANALYSIS_TMP/cache" "$ANALYSIS_TMP/mplconfig"
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$ANALYSIS_TMP/cache" \
MPLCONFIGDIR="$ANALYSIS_TMP/mplconfig" \
python analysis/run_analysis.py paper-inputs \
  --root . \
  --output-dir "$ANALYSIS_TMP/paper_input_values" \
  --figures-dir "$ANALYSIS_TMP/paper_input_figures" \
  --tables-dir "$ANALYSIS_TMP/paper_input_tables"
```

Expected generated value files are `statistical_significance_pie_counts.csv`, `creeping_overfitting_simplerenv_plot_data.csv`, and `creeping_overfitting_confidence_intervals.csv`. Expected generated artifacts are `statistical_significance_pies_paper_inputs.svg`, `creeping_overfitting_simplerenv_paper_inputs.svg`, and `creeping_overfitting_confidence_intervals_paper_inputs.tex`.

For an external copy of the same paper CSV inputs, add `--paper-data-root /path/to/paper_inputs`. The directory must contain the same eight compact CSV filenames as `analysis/paper_inputs`.

## Release-Current Supplement

Regenerate the committed release-current CSVs and optional figures/tables:

```bash
ANALYSIS_TMP="${ANALYSIS_TMP:-../benchmark_release_analysis_scratch}"
mkdir -p "$ANALYSIS_TMP/cache" "$ANALYSIS_TMP/mplconfig"
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$ANALYSIS_TMP/cache" \
MPLCONFIGDIR="$ANALYSIS_TMP/mplconfig" \
python analysis/run_analysis.py current \
  --root . \
  --output-dir analysis/release_current_values \
  --figures-dir "$ANALYSIS_TMP/release_current_figures" \
  --tables-dir "$ANALYSIS_TMP/release_current_tables"
```

Regenerate into a separate directory and compare with the committed release-current CSVs:

```bash
ANALYSIS_TMP="${ANALYSIS_TMP:-../benchmark_release_analysis_scratch}"
mkdir -p "$ANALYSIS_TMP/cache" "$ANALYSIS_TMP/mplconfig"
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$ANALYSIS_TMP/cache" \
MPLCONFIGDIR="$ANALYSIS_TMP/mplconfig" \
python analysis/run_analysis.py current \
  --root . \
  --output-dir "$ANALYSIS_TMP/current_check" \
  --expected-dir analysis/release_current_values
```

`release_current_values/` contains these committed files:

1. `statistical_significance_pie_counts.csv`
2. `creeping_overfitting_simplerenv_plot_data.csv`
3. `creeping_overfitting_confidence_intervals.csv`
4. `statistical_significance_bucket_comparison.csv`

## Paper-Versus-Current Classification Audit

Run this command to write row-level evidence for the significance classification differences between the frozen paper inputs and the release-current public generator:

```bash
ANALYSIS_TMP="${ANALYSIS_TMP:-../benchmark_release_analysis_scratch}"
mkdir -p "$ANALYSIS_TMP/cache" "$ANALYSIS_TMP/mplconfig"
PYTHONDONTWRITEBYTECODE=1 \
XDG_CACHE_HOME="$ANALYSIS_TMP/cache" \
MPLCONFIGDIR="$ANALYSIS_TMP/mplconfig" \
python analysis/run_analysis.py compare-paper \
  --root . \
  --output-dir "$ANALYSIS_TMP/paper_comparison"
```

The command writes `paper_vs_release_significance_differences.csv` and `paper_vs_release_significance_summary.csv`. With the committed paper inputs and release-current generator, it exits with status `1` because `92` row classifications differ. The row sets and paper-name order match for all five pie inputs. The observed causes are `68` zero-or-negative-gain definition differences, `13` integer-count rounding differences, `6` threshold-equality differences, and `5` saturated or undefined cutoff differences.

The committed `statistical_significance_bucket_comparison.csv` labels the bucket order as `negative;not_significant;possibly_significant;certainly_significant`. The paper/current bucket counts are:

1. LIBERO: paper `302,164,167,156`; current `363,112,170,144`; total `789`.
2. CALVIN: paper `21,3,36,47`; current `24,0,36,47`; total `107`.
3. SimplerEnv: paper `13,23,62,24`; current `17,16,62,27`; total `122`.
4. RoboCasa: paper `4,0,10,16`; current `4,0,10,16`; total `30`.
5. RoboTwin 2.0: paper `2,0,3,14`; current `2,0,3,14`; total `19`.

## Interpretation Notes

1. Paper-input significance pies are generated from the frozen CSVs in `analysis/paper_inputs`. Release-current significance pies are generated from `statistical_significance/significance_categories/all_comparisons.csv`, which is regenerated by the public category code.
2. The package preserves both paths rather than redefining or relabeling either set of counts. `compare-paper` fails loudly for the material classification mismatch and writes row-level evidence.
3. Confidence intervals follow the paper convention: Newcombe-Wilson difference of proportions for SimplerEnv and LIBERO success-rate drops, paired row-level Wald intervals for CALVIN resampled-pose drops, and independent two-sample Wald intervals for CALVIN fresh-sequence drops.
