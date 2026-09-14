# What Are We Actually Benchmarking in Robot Manipulation?

**CoRL 2026 (to appear)**<br>
**IROS 2026 RGMCW workshop**

[Paper](https://arxiv.org/abs/2606.04233) · [Project website](https://ripl.github.io/manipulation_benchmark_audit/) · [Datasets and checkpoints](https://drive.google.com/drive/folders/1ZYpEvdD1cf6JSLQiSu7hK0xahSNRvS9F)

**TL;DR:** High scores on commonly used benchmarks (LIBERO, SimplerEnv) are not proof of broader manipulation capability.

## Findings

- **LIBERO:** A **0.09B** probe nears top scores without language encoding or robotics pretraining.
- **Significance:** Only **19.8% / 19.7%** of LIBERO / SimplerEnv gains are provably significant.
- **CALVIN:** Resampling block poses within the training range drops X-VLA from **4.17 to 3.14** tasks completed out of five.
- **SimplerEnv:** **22M** policies trained near the test reach **94.8%**, versus **95.8%** for **0.9B** X-VLA.
- **Most-reported benchmarks fail more diagnostics:** LIBERO, CALVIN, and SimplerEnv fare worse than RoboCasa and RoboTwin 2.0.

## Four diagnostics

- **Shortcut solvability:** High scores without the claimed capabilities.
- **Statistical significance:** Whether gains exceed evaluation noise.
- **Creeping overfitting:** Fitting narrow test conditions or fixed test samples.
- **Data-source dependence:** Generalization versus training close to the test.

## What's included

- [Shortcut solvability](shortcut_solvability/README.md): LIBERO/CALVIN training, evaluation, and results.
- [Statistical significance](statistical_significance/README.md): shared-test outcomes and leaderboard comparisons.
- [Creeping overfitting](creeping_overfitting/README.md): results, custom settings, assets, and starting states.
- [Data-source dependence](data_source_dependency/README.md): scripted WidowX collection, training, evaluation, and results.
- [Analysis](analysis/README.md): regenerate selected paper figures and tables on CPU.
- [Google Drive](https://drive.google.com/drive/folders/1ZYpEvdD1cf6JSLQiSu7hK0xahSNRvS9F): demonstration datasets, selected model weights, and evaluation inputs.
- [Reproduction details](REPRODUCIBILITY.md): setup, validation, and exclusions.
- [Claim-to-artifact guide](CLAIMS.md): evidence behind the results.

## Contact

- [Open an issue](https://github.com/ripl/ManipulationBenchmarkAudit/issues) or [email Tianchong Jiang](mailto:tianchongj@ttic.edu).
