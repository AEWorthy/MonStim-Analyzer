# Changelog

All notable changes to the MonStim Analyzer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## [0.4.1] - Previous Release

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
