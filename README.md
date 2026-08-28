
# MonStim Analyzer - EMG Analysis and Visualization Tool

**Version:** 0.6.0 (beta)

This repository contains the source code for the MonStim EMG Analyzer GUI
application and the supporting signal processing library. The tool provides
utilities for importing, processing and visualizing electromyography (EMG) data
collected with the custom 'MonStim-V3' LabView program created by William Goolsby for the Alvarez Lab at Emory.

## Documentation

- **[User Guide](docs/user/using_monstim.md)** - Complete usage workflow and feature reference
- **[Quickstart Guide](QUICKSTART.md)** - Express installation and overview
- **[Changelog](CHANGELOG.md)** - Version history and release notes
- **[EMG Processing](docs/science/emg_processing.md)** - Signal processing and transformations
- **[M-max Algorithm](docs/science/mmax_estimation.md)** - M-max detection and review
- **[Recording Exclusion](docs/user/recording_exclusion_editor.md)** - Managing data quality and exclusions
- **[Multi-Experiment Import](docs/user/importing_experiments.md)** - Batch import workflows
- **[UI Scaling Guide](docs/user/ui_scaling_guide.md)** - Display configuration for different screen sizes

## Quick Start

See the **[Quickstart Guide](QUICKSTART.md)** for express installation instructions and a brief program overview. For full usage instructions, see the **[User Guide](docs/user/using_monstim.md)** and the in-app **Help > Show Help** library.

## Developer Quick Note

- When running locally for development or to execute tests, activate the `monstim` conda environment first:

```pwsh
conda activate monstim
```

See `QUICKSTART.md` and the [Help Library](docs/user/index.md) for user documentation. Developer setup and testing instructions are maintained in the repository documentation.

## Conda vs Pip and PyQt pinning

- **PyQt (GUI) should be installed via Conda.** GUI packages are binary and platform-sensitive; Conda provides compatible builds for many platforms. Do not list `PySide6` in `requirements.txt` when you provide `pyqt` in `environment.yml`.
- To pin the exact PyQt version used by developers and CI, we pin the conda package in `environment.yml` (example: `pyqt=6.10.0`). This ensures consistent GUI binaries across machines.
- If you need a pip-only workflow, provide separate instructions and a `requirements-pip.txt` that pins `PySide6` explicitly (e.g. `PySide6==6.10.0`) and document that pip installs are not covered by the `monstim` conda environment.

CI and tooling notes:
- The repository's dependency consistency check understands the `pyqt` ↔ `PySide6` name mapping and will not fail when `pyqt` is provided by `environment.yml` and omitted from `requirements.txt`.
- Use `tools/sync_env_from_requirements.py` or the GitHub Actions workflow to keep `environment.yml` and `requirements.txt` in sync when changing pinned versions.
