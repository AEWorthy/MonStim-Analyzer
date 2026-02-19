### Legacy vs Active Tests

- Markers:
    - unit: fast, isolated
    - integration: cross-component flows
    - slow: perf/large data
    - legacy: quarantined, pending modernization

- Modernized and active (default run):
    - test_command_pattern_system.py (integration)
    - test_data_curation_commands.py (integration)
    - test_dataset_inclusion_and_delete.py (integration)
    - test_integration_end_to_end.py (integration, slow)
    - test_error_handling_recovery.py (integration)
    - test_plotting_error_handling.py (unit)
    - test_signal_processing.py (unit)
    - test_mmax_algorithm_errors.py (unit)
    - test_integrity_copy_move.py (integration)
    - test_golden_import.py (integration, slow)
    - test_gui_state_management.py (integration)
    - test_domain_model_real_data.py (integration, slow)
    - test_domain_annotations.py (integration, slow)
    - test_domain_hierarchy_and_caches.py (unit)
    - test_domain_mmax_selection.py (unit)

- Legacy/retired:
    - test_domain_business_logic.py (monolithic; replaced by focused tests and now skipped; safe to delete in cleanup)
### Running Locally

By default, active tests run (legacy excluded by default via pytest.ini). To run everything active:
`pytest -q`

If you need to include any legacy markers in the future:
`pytest -q -m "legacy"`
### Status (as of 2025-09-23)

- Active set: green (including integration and error-handling suites)
- Legacy set: 1 module quarantined (domain monolith)
# MonStim Analyzer - Test Suite Summary

## Overview
This test suite validates domain logic, import/export workflows, and GUI-adjacent behaviors for MonStim Analyzer. The suite is organized with pytest markers and a legacy quarantine to keep daily runs stable while we refactor older, drifted tests.

## Marker strategy and default behavior
- Markers are defined in `pytest.ini` and the default run excludes legacy: `addopts = -q -m "not legacy"`.
- Available markers:
    - `unit`: Fast, isolated unit tests
    - `integration`: Cross-module/integration tests
    - `slow`: Performance/large data
    - `legacy`: Out-of-date tests kept for reference; excluded by default

Common runs:
- Default stable run (recommended): just run `pytest` (uses the default `-m "not legacy"`).
- Only legacy tests: `pytest -m legacy`
- Everything including legacy: `pytest -m "legacy or not legacy"`
- Specific class/test: standard pytest selectors still apply, e.g. `pytest tests/test_signal_processing.py::test_butter_filter`

## Legacy quarantine (to be refactored)
The following drifted module is quarantined under the `legacy` marker so the default run remains green. It’ll be split and updated incrementally.
- `tests/test_domain_business_logic.py` (monolithic; being split by concern)

Recently modernized (now part of default suite):
- `tests/test_command_pattern_system.py` (aligned to monstim_gui.commands)
- `tests/test_data_curation_commands.py`
- `tests/test_dataset_inclusion_and_delete.py`
- `tests/test_integration_end_to_end.py`
- `tests/test_error_handling_recovery.py`

Status: All non-legacy tests pass locally in prior runs; new domain tests added. The monolithic domain test is now skipped and slated for deletion.

## Test Files and Coverage (non-legacy focus)

### Commands and inclusion/exclusion tests
These are aligned with the current command APIs and run by default. Real-data subtests are skipped automatically when golden paths aren’t present.

### Dataset inclusion/exclusion and deletion
Note: Currently marked `legacy`; will be updated to the current `Command` and repository APIs.

### 3. `test_integrity_copy_move.py` (2 tests)
**Purpose**: Validates data integrity during copy and move operations with nested file structures.

**Test Coverage**:
- `test_copy_preserves_all_content_and_metadata`: Creates nested dataset, copies, validates integrity
- `test_move_preserves_all_content_and_metadata`: Creates nested dataset, moves, validates integrity

**Key Validations**:
- All nested files and directories preserved during operations
- File content integrity maintained (using SHA-256 hashes)
- Metadata files properly created and updated
- Source and destination have identical structure and content

### 4. `test_golden_import.py`
**Purpose**: Comprehensive testing of CSV import pipeline using curated golden fixtures.

**Test Coverage**:
- `test_import_structure_and_integrity`: Validates import creates correct HDF5/JSON structure
- `test_invalid_csv_handling`: Tests graceful handling of malformed CSV files
- `test_domain_object_consistency`: Verifies domain objects load correctly after import
- `test_import_performance_baseline`: Performance regression testing with timing checks
- `test_threading_consistency`: Validates multi-threaded import produces consistent results

**Key Validations**:
- CSV files correctly converted to `.raw.h5`/`.meta.json`/`.annot.json` triples
- Domain objects (Experiment/Dataset/Session/Recording) load with correct hierarchy
- Invalid CSV files handled gracefully without crashing import process
- Import performance stays within reasonable bounds (< 0.5s per CSV)
- Single-threaded and multi-threaded imports produce identical results

## Golden Fixtures Structure

### Location: `tests/fixtures/golden/`
Professional test fixture organization with comprehensive documentation.

**Structure**:
```
tests/fixtures/golden/
├── README.md                           # Comprehensive documentation
├── 240808 C571.4 pre-dec mcurves_long-/  # Dataset 1 (v3h format)
├── 240815 C572.5 pre-dec mcurves_long-/  # Dataset 2 (v3d format)  
├── [Third Dataset]/                    # Dataset 3 (v3h format)
└── invalid/                           # Negative test cases
    ├── Empty Dataset/
    │   └── WT00-0000.csv              # Empty file
    ├── Malformed Dataset/
    │   └── WT01-0000.csv              # Missing headers
    └── Wrong Naming/
        └── BADNAME.csv                # Invalid filename pattern
```

**Coverage**:
- Real-world CSV datasets across formats (v3h and v3d)
- Negative cases e.g., empty files, malformed headers, invalid naming

## Test Infrastructure

### Key Components:
- **Isolated environments**: Each test uses temporary directories with automatic cleanup
- **Headless operation**: GUI components stubbed for automated testing
- **Path override**: Tests use isolated output paths to prevent interference
- **Performance monitoring**: Baseline performance checks to catch regressions
- **Threading validation**: Ensures concurrent operations produce consistent results

### Test Utilities:
- `helpers.py`: Utilities for generating test data and validating file integrity
- `conftest.py`: Pytest fixtures for environment setup and GUI mocking
- Golden fixture scanning functions for structure validation

## Validation Scope

### Data Integrity:
- ✅ File content preservation during copy/move operations
- ✅ Metadata consistency across operations
- ✅ Directory structure preservation
- ✅ Hash-based content verification

### Command Pattern:
- ✅ Execute/undo cycle for all reversible operations
- ✅ Proper warning behavior for irreversible operations
- ✅ Command state management and cleanup
- ✅ GUI integration without coupling

### Import Pipeline:
- ✅ CSV to HDF5 conversion accuracy
- ✅ Format detection (v3h vs v3d)
- ✅ Error handling for malformed inputs
- ✅ Multi-threading consistency
- ✅ Performance regression detection

### Domain Objects:
- ✅ Experiment/Dataset/Session/Recording hierarchy
- ✅ Annotation overlay system
- ✅ Repository pattern implementation
- ✅ Cache invalidation and state management

## Running Tests

### Prerequisites:
```bash
# Activate the conda environment
conda activate monstim
```

### Individual Test Files:
Use standard pytest selection, e.g. `pytest tests/test_integrity_copy_move.py -v`.

### Complete Test Suite:
Default (stable): `pytest` (excludes legacy)

Include legacy as well: `pytest -m "legacy or not legacy"`

### Expected Results:
- All non-legacy tests should pass consistently
- Runtime depends on environment and data fixtures
- No warnings or errors in the default run

## Maintenance Notes

### Adding New Tests:
1. Follow existing patterns for fixtures and temporary directories
2. Use `helpers.py` utilities for common operations
3. Ensure proper cleanup in test teardown
4. Add negative test cases where appropriate

### Golden Fixtures:
- CSV files are curated and should not be modified without careful consideration
- Add new datasets to `tests/fixtures/golden/` following naming conventions
- Update `README.md` when adding new fixture types
- Maintain both positive and negative test cases

### Performance Monitoring:
- Baseline performance checks help catch regressions
- Adjust timing thresholds based on expected hardware performance
- Consider pytest-benchmark integration for more sophisticated performance testing

## Integration with CI/CD

CI should run the default stable suite (non-legacy) for fast feedback, and schedule the legacy suite periodically or on-demand:
- PR/CI default: `pytest` (non-legacy only)
- Nightly or pre-release: `pytest -m "legacy or not legacy"`

This keeps CI fast while preserving coverage for areas undergoing refactoring.

## Golden channel counts assurance

- Added `tests/test_golden_channel_counts.py` to validate that all sessions imported from `tests/fixtures/golden` have at least 2 channels, verified via both `.meta.json` (`num_channels`) and `.raw.h5` dataset shapes.
- Markers: `integration`, `slow` (runs a full import of golden fixtures into a temp directory; never touches `data/`).
- Current status: PASS on development branch (as of 2025-09-23), no channel-count issues found.