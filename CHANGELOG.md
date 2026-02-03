# Changelog

All notable changes to the MonStim Analyzer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.5.1] - 2026-02-03

### Added
- **Experiment Index System**: New background indexing and caching system for faster experiment loading and improved metadata access.
- **Date Tracking for Annotations**: Added `date_added` and `date_modified` fields to all annotation models with migration to `DATA_VERSION = "2.1.0"`.
- **DataFrame Export Improvements**: Added descriptive filenames for CSV exports with experiment/dataset/session context.
- **Latency Window Clipboard Features**: New single-window clipboard and append/replace functionality for latency window management across experiments.
- **User Preference for Index Building**: Added preference to control automatic experiment index building behavior.
- **AUC Amplitude Calculation Method**: New "Area Under Curve" (AUC) method for calculating EMG amplitude, providing cumulative muscle activation measurement across analysis windows.

### Changed
- **Session Restoration**: Refactored session restoration to complete after experiment load, improving startup reliability and state management.
- **Progress Dialog**: Improved resizing behavior for large experiments with many datasets.
- **Channel Selection Persistence**: Enhanced channel selection memory and status messages across plotting operations.
- **Y-axis Scaling**: Improved linked plot Y-axis scaling for better data visualization.
- **OpenGL Acceleration**: Changed default OpenGL acceleration preference to disabled for better compatibility.
- **Error Handling**: Improved experiment rename validation, error logging, and user feedback throughout the application.
- **Thread Management**: Enhanced thread cancellation and cleanup logic in GUI operations for better resource management.
- **Plot Memory Management**: Improved memory handling and diagnostics for plotting operations.
- **Context Menu Positioning**: Enhanced fallback logic for context menu positioning on edge cases.
- **Export Timestamps**: Removed timestamps from export filenames for cleaner file naming.

### Fixed
- **Experiment Load Cancellation**: Fixed cleanup and state handling when users cancel experiment loading operations.
- **Session Restoration Cancellation**: Properly handle cancellation during session restoration to avoid inconsistent states.
- **Dataset Persistence**: Fixed issue where excluded datasets were not properly persisted in repositories.
- **Index Staleness Detection**: Improved detection logic for when experiment indices need rebuilding.
- **Dataset Count Checking**: Added proper validation in index staleness function to prevent errors.
- **Skipped Dataset Handling**: Now warns and reports datasets that are skipped during experiment load due to errors.
- **Raw Data Logging**: Fixed length logging in plot_data function for accurate diagnostics.
- **Context Menu Positioning**: Fixed edge cases where context menus could appear offscreen.

### Testing
- Added comprehensive tests for GUI state management and session restoration.
- Added tests for experiment indexing and staleness detection.
- Updated OpenGL acceleration preference tests to match new defaults.
- Improved test coverage for plotting and memory management.
- Fixed file handle cleanup issues in dark mode rendering tests.
- Added integration tests for math rendering in GUI dialogs.

### Dependencies
- Updated pytest to v9.0.2.
- Updated pip dependencies across multiple packages.
- CI: Bumped `actions/upload-artifact` from v5 to v6.
- CI: Bumped `actions/cache` from v4 to v5.
- Updated `environment.yml` with latest dependency versions.

### Configuration
- Added new latency window presets to `config.yml` for vibration experiments.
- Added data version 2.1.0 migration for date field additions.
- Updated YAML formatting for vibration_H_7 presets.
- Migrated Renovate config to `.github/` directory for better organization.

### Notes
- This is a maintenance release with significant improvements to experiment loading performance, state management, and stability.
- The new experiment indexing system provides faster loading times for large experiments with many datasets.
- Users can control index building behavior through Program Preferences.
- Date tracking in annotations enables better data provenance and audit trails.


## [0.5.0] - 2025-12-03

### Added
- Windows distributable and installer artifacts: `MonStim Analyzer v0.5.0.exe` and `MonStim_Analyzer_v0.5.0-WIN.zip` (packaging managed via `win-main.spec`).
- Comprehensive Quickstart and user-facing documentation packaged with the release (`docs/readme.md`, `QUICKSTART.md`).
- Improved release QA artifacts included in the `build/` directory for easier verification of installers and packaging.
- High-DPI LaTeX math rendering in the Help/About window and a math image cache (migrated to user directory) to support improved help dialog math rendering and offline images.
- `env_recreate.ps1` helper script for local environment recreation and other environment tooling moved/added to `tools/`.

### Changed
- Incremented GUI/application version to `0.5.0` (`monstim_gui/version.py`) and prepared repository for a full, non-beta release.
- Documentation overhaul: expanded Quickstart, installation instructions, and the M-max algorithm documentation in `docs/`.
- Packaging: updated PyInstaller spec (`win-main.spec`) to include documentation and QUICKSTART files in the distributable and to use a consistent EXE/DIST naming convention.
- Data handling: retained `DATA_VERSION = "2.0.1"` while ensuring migration hooks remain available for future changes.
- Migrated GUI framework from **PyQt6** to **PySide6** (add fallback handling in CI and dependency manifests).
- Refactored help dialog and math rendering pipeline: removed MathJax/WebEngine, improved Markdown math rendering and image URI handling, and added explicit cache clearing API.
- Refactored zoom handling and close-event cleanup logic in multiple GUI dialogs.
- Improved experiment plotter, CSV importer, and related plotting/backend import organization; moved matplotlib backend setup where needed.
- CI and packaging updates: switch CI to use `environment.ci.yml` with `mamba`/`conda` changes, add PySide6 fallback, reorganize Renovate/Dependabot configs, and adjust GitHub Actions setup.
- Refactored imports and moved some tooling (e.g. `setup.py` to `tools/`) for repository organization.
- Limited `CommandInvoker` history size and increase default command history limit to 100 to reduce unbounded growth.

### Fixed
- Various stability and UI issues discovered during release QA: improved startup reliability, settings persistence, and plot refresh behavior across several widgets.
- Resolved packaging path and data inclusion issues so documentation and sample files are present in the distributable.
- Closed resource/file-handle leaks surfaced in tests (fixes for `test_dark_mode_produces_different_images` and other tests).
- Explicit cleanup added before forced garbage collection in tests to avoid object growth during test runs.
- Improved error handling for plotting configuration and safer backend fallback behavior.

### Testing
- Expanded pytest-based automated tests and golden fixtures under the `tests/` tree to improve coverage and regression protection.
- Test configuration and CI-related updates included to better exercise packaging and GUI smoke checks (`pytest.ini`, `pyproject.toml`).
- Added and updated integration tests and GUI-related tests (math rendering, integration end-to-end), and improved test robustness for GC/object growth and performance tests.
- Updated CI to use `environment.ci.yml` and modernized test environment setup.

### Dependencies
- Small dependency bumps and housekeeping in `requirements.txt`/`pyproject.toml` to keep CI and packaging stable.
- Continued reliance on PyQtGraph as the supported plotting backend (matplotlib plotters were removed in earlier releases).
- Bumped core dependencies: Python → 3.14, NumPy → 2.3.5, SciPy → 1.16.3, pytest → 9.x and multiple `requirements.txt`/`environment.yml` updates.
- Renovate and Dependabot tooling/config updates to better manage automated dependency upgrades.

### Status
- This is the first full release (no longer beta).

### Notes
- See `docs/readme.md` and `QUICKSTART.md` for installation, usage, and packaging details. Visit the repository releases page for downloadable artifacts and checksums.


## [0.4.3] - 2025-09-23

### Changed
- Improved experiment/session switching behavior with clearer UI status updates
- Renamed and polished Analysis Profile UI labels for clarity
- Refined margins and label alignment across several GUI widgets
- Data selection widget refresh and experiment refresh logic improvements

### Added
- Data Curation Manager dialog and related commands
- Manage Recordings button in the data selection widget
- Lightweight metadata methods for repositories to speed up UI listing
- Pytest-based test infrastructure and initial golden fixtures

### Fixed
- Minor repository and loader adjustments for stability

### Dependencies
- Bumped Markdown to 3.9
- Updated matplotlib and pytest via dependency groups
- CI: Bumped actions/setup-python from v5 to v6

Note: This is a patch release with incremental UX and stability improvements over 0.4.2.


## [0.4.2] - 2025-09-02

### Added
- **Recording Exclusion Editor**: New bulk exclusion tool for filtering recordings based on stimulus amplitude and other criteria at session, dataset, or experiment level
- **Dataset Metadata Editor**: Dialog for editing dataset metadata (date, animal ID, condition) with automatic folder renaming support
- **Enhanced Tooltips**: Comprehensive tooltips throughout the user interface for better usability
- **OpenGL Hardware Acceleration**: User preference option for enabling OpenGL rendering with performance optimizations
- **Program Preferences**: New, unified program 'Settings' dialog with display, UI scaling, OpenGL, and data tracking settings.

### Changed
- **Plotting Performance**: Major performance improvements with auto-downsampling, clip-to-view optimization, OpenGL support, and PyQtGraph enhancements
- **Plotting Backend Migration**: Completely removed deprecated matplotlib plotters in favor of PyQtGraph-based plotting
- **UI/UX Improvements**: 
  - Enhanced crosshair and tooltip positioning in PyQtGraph plotter for more precise data visualization
  - Improved UI spacing and layout across multiple dialogs and widgets
  - Enhanced latency window dialog layout and usability
  - Refined plot options interface
- **Data Import Process**: Improved validation to allow users to fix/force chosen names during dataset validation
- **M-max System**: Updated M-max detection algorithms and improved documentation.
- **Legend and Error Bar Handling**: Improved legend handling and error band plotting in PyQtGraph plotters
- **Latency Window Plotting**: Now plots latency windows once per channel on EMG plots for cleaner, faster visualization

### Fixed
- **Qt Error Handling**: Resolved silent Qt painter errors during UI resizing and moving
- **QSettings Persistence**: Fixed issues preventing proper setting persistence across sessions
- **Plot Updates**: Removed redundant display_plot calls and improved plot refresh efficiency
- **Data Selection**: Enhanced CircleDelegate to check painter activity before drawing to prevent errors
- **Layout Consistency**: Improved consistency across different UI components and dialogs

### Technical Improvements
- Added bulk recording exclusion command with full undo/redo support
- Refactored dataset name validation for flexible handling of non-standard names
- Enhanced logging throughout the application
- Improved error handling in PlotController for user-friendly messages
- Better handling of HDF5 file operations after folder renames

### Dependencies
- Updated pandas in the scientific-computing group
- Updated GitHub Actions (actions/checkout from v4 to v5)

### Breaking Changes
- **Matplotlib Plotters Removed**: Completely removed deprecated matplotlib-based plotting modules in favor of PyQtGraph. This improves performance but removes matplotlib as a dependency.


## [0.4.1] - 2025-07-05

### Features
- EMG data import and processing
- Multi-experiment batch import
- Advanced M-max detection algorithms
- Interactive plotting with PyQtGraph
- Data export capabilities
- Analysis profiles for different experimental conditions
- Recording exclusion management
- Undo/redo functionality

### Known Issues
- Mac builds not currently available
- First launch may trigger security warnings (unsigned software)

---

For detailed technical information, see the [User Guide](docs/readme.md) and other documentation in the `docs/` directory.




