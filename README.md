# What Are We Actually Benchmarking in Robot Manipulation?

**CoRL 2026 (to appear)**<br>
**IROS 2026 RGMCW workshop**

[Paper](https://arxiv.org/abs/2606.04233) · [Project website](https://ripl.github.io/manipulation_benchmark_audit/) · [Datasets and checkpoints](https://drive.google.com/drive/folders/1ZYpEvdD1cf6JSLQiSu7hK0xahSNRvS9F)

## About the paper

A high benchmark score is often treated as evidence of general manipulation capability. We examine whether that inference is justified through four diagnostics, applied to **LIBERO, CALVIN, SimplerEnv, RoboCasa, and RoboTwin 2.0**.

On LIBERO, a small probe approaches reported SOTA without a language encoder or large-scale robotics pretraining, and most reported gains cannot be established as statistically significant. On CALVIN, changing block poses within the training range lowers every tested policy's performance. These findings challenge what scores establish about capability; they do not establish that high-scoring policies lack capability.

## The four diagnostics

1. **Shortcut solvability:** Can a policy reach a comparable score while lacking the capabilities that score is taken to demonstrate?
2. **Statistical significance:** Does the reported evaluation evidence establish that an improvement is statistically significant?
3. **Creeping overfitting:** Have policies fitted the benchmark's narrow test distribution or its fixed test samples? The diagnostic distinguishes these two possibilities and keeps changes within the training distribution.
4. **Data-source dependence:** Can training data close to the test produce a score that would otherwise be interpreted as generalization across a train–test gap? In this audit, this diagnostic applies to SimplerEnv.

## What's included

| Diagnostic | Released materials |
|---|---|
| [Shortcut solvability](shortcut_solvability/README.md) | LIBERO/CALVIN shortcut-policy training and evaluation code, configs, and results. |
| [Statistical significance](statistical_significance/README.md) | Leaderboard comparisons, shared-instance rollout outcomes, and significance analysis. |
| [Creeping overfitting](creeping_overfitting/README.md) | Evaluation results, custom configs and assets, and exact reset and initial-state inputs. |
| [Data-source dependence](data_source_dependency/README.md) | Scripted WidowX demonstration collection, training and evaluation code, and results. |

We also include [leaderboard snapshots](leaderboards/) and [CPU analysis scripts](analysis/README.md) for regenerating selected paper figures and tables. The [claim-to-artifact guide](CLAIMS.md) connects the reported results to the released evidence.

## Using the release

To **explore the paper's results**, start with the diagnostic links above. To **regenerate figures and tables**, follow the [analysis guide](analysis/README.md).

To **train or evaluate our simple policies**, follow the [LIBERO/CALVIN shortcut guide](shortcut_solvability/README.md) or the [scripted WidowX guide](data_source_dependency/README.md). Both describe the required benchmark software and data.

The [Google Drive folder](https://drive.google.com/drive/folders/1ZYpEvdD1cf6JSLQiSu7hK0xahSNRvS9F) contains our scripted demonstration datasets and selected trained checkpoints, plus a convenience copy of the evaluation inputs. Download only the archives you need; code and compact result files are already in this repository.

For third-party policies, we provide evaluation artifacts and custom inputs; their official repositories supply the full software environments. Detailed reproduction scope, validation coverage, archive links, and exclusions are in [Reproduction and Release Details](REPRODUCIBILITY.md).

## Contact

Please [open an issue](https://github.com/ripl/ManipulationBenchmarkAudit/issues) or email [Tianchong Jiang](mailto:tianchongj@ttic.edu).
