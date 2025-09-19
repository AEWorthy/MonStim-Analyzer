# MonStim Analyzer - Test Suite Summary

## Overview
This comprehensive test suite validates the data curation commands, domain object integrity, and CSV import pipeline for the MonStim Analyzer application. All tests are designed to run in isolation with proper cleanup and provide robust coverage of the core functionality.

## Test Files and Coverage

### 1. `test_data_curation_commands.py` (5 tests)
**Purpose**: Validates the command pattern implementation for experiment and dataset operations.

**Test Coverage**:
- `test_create_and_undo_experiment`: Creates experiment, verifies filesystem structure, tests undo
- `test_rename_experiment_and_undo`: Renames experiment, checks metadata updates, validates undo
- `test_move_dataset_and_undo`: Moves dataset between experiments, verifies structure, tests undo
- `test_copy_dataset_and_undo`: Copies dataset, ensures independent copies, validates undo
- `test_delete_experiment_is_irreversible`: Tests irreversible delete with warning on undo attempt

**Key Validations**:
- Filesystem operations create correct directory structures
- Annotation files (`.annot.json`) are properly created and updated
- Command undo functionality works correctly for reversible operations
- Irreversible operations properly warn users when undo is attempted

### 2. `test_dataset_inclusion_and_delete.py` (2 tests)
**Purpose**: Tests dataset inclusion/exclusion and deletion functionality.

**Test Coverage**:
- `test_toggle_dataset_inclusion_and_undo`: Tests include/exclude toggling with full undo support
- `test_delete_dataset_command_irreversible`: Validates irreversible dataset deletion

**Key Validations**:
- Dataset exclusion state tracked in `experiment.annot.json` under `excluded_datasets`
- Include/exclude operations are fully reversible with proper undo behavior
- Dataset deletion is irreversible and warns on undo attempts

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

### 4. `test_golden_import.py` (5 tests)
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
- **80+ CSV files** across multiple datasets
- **Multiple format support**: v3h and v3d CSV formats
- **Negative test cases**: Empty files, malformed headers, invalid naming
- **Real-world data**: Actual experimental data for comprehensive validation

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
conda activate alv_lab
```

### Individual Test Files:
```bash
# Data curation commands
python -m pytest tests/test_data_curation_commands.py -v

# Dataset inclusion/exclusion and deletion
python -m pytest tests/test_dataset_inclusion_and_delete.py -v

# Copy/move integrity validation
python -m pytest tests/test_integrity_copy_move.py -v

# Golden CSV import pipeline
python -m pytest tests/test_golden_import.py -v
```

### Complete Test Suite:
```bash
# Run all tests together
python -m pytest tests/test_*.py -v
```

### Expected Results:
- **14 tests total**
- **All tests should pass**
- **Typical runtime**: ~6-10 seconds
- **No warnings or errors**

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

These tests are designed to run in automated environments:
- **Headless operation**: No GUI dependencies
- **Isolated execution**: No external file system dependencies
- **Deterministic results**: Consistent across different environments
- **Comprehensive coverage**: Validates both happy path and error conditions

The test suite provides confidence that data curation operations work correctly and can be safely deployed in production environments.