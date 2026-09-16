# MonStim Analyzer

**Reproducible EMG analysis and visualization for MonStim laboratory exports.**

MonStim Analyzer helps researchers import MonStim V3 recordings, review data quality, define latency windows, calculate EMG responses, visualize results, and export traceable outputs.

[:octicons-download-16: Download Windows beta](https://github.com/AEWorthy/MonStim-Analyzer/releases){ .md-button .md-button--primary }
[:octicons-rocket-16: Start your first analysis](https://worthy-lab.org/MonStim-Analyzer/user/getting_started.html){ .md-button }

## Project and developer

MonStim Analyzer is developed by [Andrew Worthy](https://worthy-lab.org/). Visit the [Worthy Lab website](https://worthy-lab.org/) for his CV, research background, and contact information.

## Before you import

MonStim directly supports CSV exports from the MonStim V3D and V3H LabVIEW acquisition formats. It does not infer arbitrary proprietary or custom acquisition layouts. If your data come from another system, use an approved importer add-on or [request an importer](https://github.com/AEWorthy/MonStim-Analyzer/issues/new?template=importer_request.yml).

## A typical workflow

1. Import an experiment and inspect a raw and filtered traces.
2. Confirm channel identity, timing, stimulus alignment, and latency windows.
3. Select the analysis profile and amplitude method appropriate for the protocol.
4. Review diagnostic notices, exclusions, plot contribution counts, and missing values.
5. Export results with the analysis settings needed to interpret them.

![MonStim Analyzer displaying synthetic H-reflex EMG overlays for two muscles, with M-wave and H-reflex timing windows.](assets/demo/session-emg.png)

## Documentation by need

- New to MonStim: [Getting started](https://worthy-lab.org/MonStim-Analyzer/user/getting_started.html) and [Importing experiments](user/importing_experiments.md).
- Looking for the main application reference: [Using MonStim Analyzer](user/using_monstim.md).
- Running analyses: [Analysis profiles](user/analysis_profiles.md), [Latency windows](user/latency_windows.md), and [Exporting results](user/exporting_results.md).
- Understanding a method: [Analysis methods](science/analysis_methods.md), [EMG processing](science/emg_processing.md), and [M-max estimation](science/mmax_estimation.md).
- Getting help: [Troubleshooting](user/troubleshooting.md), the in-app error-report tool, or an issue form.

## Citation

If MonStim contributes to academic work, please cite the software. The repository provides a machine-readable `CITATION.cff`, and the application offers **Help > Copy Citation**. See [How to cite MonStim](user/citing_monstim.md).
