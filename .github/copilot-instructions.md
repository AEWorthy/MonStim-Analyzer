# Copilot Instructions for MonStim Analyzer

## 🚨 ABSOLUTELY REQUIRED: Activate the `alv_lab` Conda Environment First
You MUST activate the `alv_lab` environment **before running ANY code, tests, imports, migrations, or package installs**. Failing to do this causes missing dependency errors (e.g. `pytest` not found, GUI backend issues, version mismatches).

Always do this first in every fresh terminal session:

```powershell
conda activate alv_lab
```

If the environment is missing:
1. Create it from `docs/environment.yml` (preferred) or install from `requirements.txt`.
2. Re-open VS Code or reload the window so the interpreter is detected.

Never run `python`, `pytest`, or `pyinstaller` until `conda info --envs` shows `alv_lab` with an asterisk.

If automation (like Copilot) attempts to run tests without activation, STOP and activate first. Treat unactivated runs as invalid.

## Architecture Overview

**Two-Package Structure**: The codebase is split into domain logic (`monstim_signals`) and GUI logic (`monstim_gui`). Never import PySide6 in `monstim_signals` - it must remain GUI-agnostic.

**Hierarchical Data Model**: `Experiment` → `Dataset` → `Session` → `Recording`. This hierarchy drives all UI organization and data operations. Each level has its own repository class for persistence and annotation overlay system.

**Command Pattern**: All user actions use Command objects executed through `CommandInvoker` for undo/redo support. See `monstim_gui/commands.py` for patterns like `ExcludeRecordingCommand`.

## Critical Patterns

### Error Handling
- Use `UnableToPlotError` for recoverable plotting failures - it shows user-friendly messages via `handle_unable_to_plot_error()`
- Always catch `UnableToPlotError` separately before general exceptions in plotting code
- Don't wrap `UnableToPlotError` in generic exception handlers - preserve the original error type

### Data Access
- Domain objects are accessed via Repository pattern: `SessionRepository(path).load()`
- Annotations are stored separately as JSON overlays - never modify original data files
- Use `session.recordings_filtered` (cached property) for processed data, not raw arrays
- Data state is managed through annotation objects (`SessionAnnot`, `DatasetAnnot`, etc.)

### M-max Algorithm
- **Individual Level**: Advanced multi-approach plateau detection in `plateau.py` (`get_avg_mmax()`)
- **Aggregation Levels**: Mean-based aggregation in `dataset.py` and `experiment.py` for population-level normalization
- **Key Principle**: Use mean M-max from all sessions/datasets to provide proper H/M ratio normalization
- **Selection Logic**: Prefer maximum approach, fall back through 95th percentile → top 20% mean → corrected mean
- **Validation**: Selected M-max should be ≤ validation_tolerance of plateau mean (configurable, default 105%)

### Manager Architecture
- `DataManager`: Handles import/export and data loading workflows
- `PlotController`: Orchestrates plotting with hook system for extensibility
- `ReportManager`: Generates analysis reports
- GUI components delegate business logic to these managers

## Development Workflows

### Running the Application
ALWAYS activate `alv_lab` first. Do not skip this step.

```powershell
conda activate alv_lab  # REQUIRED
python main.py --debug  # Development run with console logging
```

If `alv_lab` environment is not available, ensure all dependencies from `requirements.txt` are installed in your active environment.

### Testing Components
ALWAYS ensure the environment is active:

```powershell
conda activate alv_lab  # REQUIRED before any test
pytest -q               # Run full test suite
```

Selective interactive helpers:
```python
from monstim_signals.testing import test_session_object, test_dataset_object
```

If a test unexpectedly fails due to ImportError / missing dependency, first confirm environment activation.

### Building Release
```powershell
# Update version numbers first in monstim_gui/version.py
# Delete config-user.yml file if it exists
pyinstaller --clean win-main.spec  # Creates executable with PyInstaller
```

The PyInstaller spec file is configured with debug/release toggles - check comments in `win-main.spec` for build configuration options.

## Essential File Patterns

### Command Implementation
```python
class YourCommand(Command):
    def __init__(self, gui, ...):
        self.command_name = "Your Action Name"  # Shows in undo menu
        
    def execute(self):
        # Apply changes to domain objects
        # Update GUI state via self.gui methods
        
    def undo(self):
        # Reverse the changes exactly
```

### Domain Object Extensions
- Add methods to Session/Dataset/Experiment classes in `monstim_signals/domain/`
- Use `@cached_property` for expensive computations that can be invalidated
- Reset caches via `reset_all_caches()` when underlying data changes

### Plotting Integration
- Plotting classes are in `monstim_signals/plotting/` and use PyQtGraph
- Each domain level has its own plotter: `SessionPlotterPyQtGraph`, etc.
- Always check for `canvas=None` and raise `UnableToPlotError` appropriately

## Configuration and State

### Analysis Profiles
Stored in `docs/analysis_profiles/` as YAML files. Profiles configure default parameters for different experimental conditions.

### Application State
- Session restoration handled by `application_state.py`
- User preferences managed through `ConfigRepository`
- Undo/redo state maintained in `CommandInvoker.history`

## Data Import Expectations

**Directory Structure**: CSV files must follow specific naming conventions:
- Experiment folders contain Dataset folders
- Dataset folders: `[YYMMDD] [AnimalID] [Condition]`
- Session files use default MonStim naming

**Conversion Pipeline**: CSV → HDF5 + JSON metadata. Import handled by `csv_importer.py` with threading for large datasets.

## Common Gotchas

- Never call plotting methods without checking `channel_indices` - empty list causes `UnableToPlotError`
- Use absolute paths for all file operations - relative paths cause navigation issues
- GUI updates must happen on main thread - use `QApplication.setOverrideCursor()` for long operations
- Repository `save()` calls should follow domain object modifications for persistence

## Extension Points

- Add new plot types by extending existing plotter classes
- New commands integrate automatically with undo/redo system
- Hook system in `PlotController` allows custom pre/post-plot operations
- Manager classes can be extended for new data workflows
