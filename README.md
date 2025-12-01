
# MonStim Analyzer - EMG Analysis and Visualization Tool

**Version:** 0.5.0 (Full Release)

This is the first full release of MonStim Analyzer. All previous beta references have been removed. See CHANGELOG.md for details.

This repository contains the source code for the MonStim EMG Analyzer GUI
application and the supporting signal processing library. The tool provides
utilities for importing, processing and visualizing electromyography (EMG) data
collected with the custom 'MonStim-V3' LabView program created by William Goolsby for the Alvarez Lab at Emory.

## Documentation

- **[User Guide](docs/readme.md)** - Complete usage instructions and installation details
- **[Quickstart Guide](QUICKSTART.md)** - Express installation and overview
- **[Testing Guide](docs/testing.md)** - How to run the test suite (markers, golden fixtures, CI tips)
- **[Changelog](CHANGELOG.md)** - Version history and release notes
- **[EMG Processing](docs/Transform_EMG.md)** - Signal processing algorithms and analysis methods
- **[M-max Algorithm](docs/mmax_algorithm.md)** - Detailed technical documentation of M-max detection
- **[Recording Exclusion](docs/recording_exclusion_editor.md)** - Managing data quality and exclusions
- **[Multi-Experiment Import](docs/multi_experiment_import.md)** - Batch import workflows
- **[UI Scaling Guide](docs/ui_scaling_guide.md)** - Display configuration for different screen sizes

## Quick Start

See the **[Quickstart Guide](/QUICKSTART.md)** for express installation instructions and a brief program overview. For full installation and usage instructions, see the **[User Guide](docs/readme.md)**.

## Developer Quick Note

- When running locally for development or to execute tests, activate the `alv_lab` conda environment first:

```pwsh
conda activate alv_lab
```

See `QUICKSTART.md` and `docs/readme.md` for full developer instructions.
