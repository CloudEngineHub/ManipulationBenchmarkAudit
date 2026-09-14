# What Are We Actually Benchmarking in Robot Manipulation?

**CoRL 2026 (to appear)**<br>
**IROS 2026 RGMCW workshop**

[Paper](https://arxiv.org/abs/2606.04233) · [Project website](https://ripl.github.io/manipulation_benchmark_audit/) · [Datasets and checkpoints](https://drive.google.com/drive/folders/1ZYpEvdD1cf6JSLQiSu7hK0xahSNRvS9F)

**TL;DR:** We audit five robot manipulation benchmarks and show why high scores alone may not establish general manipulation capability.

## About the paper

Robot manipulation benchmarks test how well a robot control model, or policy, completes tasks. A high score is often treated as evidence of general manipulation capability. We test whether that conclusion is justified using four diagnostics on **LIBERO, CALVIN, SimplerEnv, RoboCasa, and RoboTwin 2.0**.

On LIBERO, a small model approaches the best reported scores without a language encoder or large-scale robotics pretraining. Most reported improvements cannot be shown to be statistically significant. On CALVIN, changing block positions and orientations within the training range lowers performance for every tested policy.

These results show why a high score alone may not establish general manipulation capability. They do not show that high-scoring policies lack that capability.

## The four diagnostics

1. **Shortcut solvability:** Can a policy reach a high score without the capabilities that score is taken to demonstrate?
2. **Statistical significance:** Does the reported evidence show that an improvement exceeds what evaluation noise could explain?
3. **Creeping overfitting:** Have policies become too tuned to a benchmark's narrow test conditions or its particular test examples?
4. **Data-source dependence:** Does a high score reflect generalization from different training conditions, or training data collected close to the test conditions?

## What's included

| Diagnostic | Released materials |
|---|---|
| [Shortcut solvability](shortcut_solvability/README.md) | LIBERO/CALVIN shortcut-policy training and evaluation code, settings, and results. |
| [Statistical significance](statistical_significance/README.md) | Leaderboard comparisons, outcomes for policies evaluated on the same test instances, and significance analysis. |
| [Creeping overfitting](creeping_overfitting/README.md) | Evaluation results, custom settings and assets, and files specifying the exact starting states used for evaluation. |
| [Data-source dependence](data_source_dependency/README.md) | Code for collecting scripted demonstrations with the simulated WidowX robot, training and evaluation code, and results. |

We also include [leaderboard snapshots](leaderboards/) and [CPU analysis scripts](analysis/README.md) for regenerating selected paper figures and tables. The [claim-to-artifact guide](CLAIMS.md) connects the reported results to the released evidence.

## Using the release

To **explore the paper's results**, start with the diagnostic links above. To **regenerate figures and tables**, follow the [analysis guide](analysis/README.md).

To **train or evaluate our simple policies**, follow the [LIBERO/CALVIN shortcut guide](shortcut_solvability/README.md) or the [scripted WidowX guide](data_source_dependency/README.md). Both describe the required benchmark software and data.

The [Google Drive folder](https://drive.google.com/drive/folders/1ZYpEvdD1cf6JSLQiSu7hK0xahSNRvS9F) contains our scripted demonstration datasets, selected trained model weights, and a copy of the evaluation inputs. Code and compact result files are in this repository.

For third-party policies, we provide evaluation results and custom inputs. Follow their official repositories to set up the software. See [Reproduction and Release Details](REPRODUCIBILITY.md) for download links, what can be reproduced, what we tested, and what is excluded.

## Contact

Please [open an issue](https://github.com/ripl/ManipulationBenchmarkAudit/issues) or email [Tianchong Jiang](mailto:tianchongj@ttic.edu).
