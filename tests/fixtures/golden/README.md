# Golden Experiment Fixtures

This folder contains a curated, deterministic dataset of raw MonStim CSV files used to validate the CSV import pipeline and repository loaders end-to-end.

## Structure

### Valid Test Data
- `240808 C322.2RE pre-dec mcurves_long+/` — Dataset with v3d format CSVs
- `240815 C309.3 post-dec mcurve_long+/` — Dataset with v3d format CSVs  
- `250917 C554.5 post-cut vibes/` — Dataset with v3h format CSVs for comprehensive coverage

Each dataset contains:
- CSVs with names like `WT21-0000.csv` where:
  - `WT21` is the 4-character session ID
  - `0000` is the 4-digit recording ID
- Optional `.STM` sidecars (ignored by importer)

### Invalid Test Data
- `invalid/` — Contains malformed inputs for negative testing:
  - `Empty Dataset/` — Intentionally empty or placeholder-only dataset
  - `Malformed Dataset/` — CSVs with missing sections or non-numeric data in [DATA]
  - `Wrong Naming/` — CSVs with invalid filename patterns

## Test Coverage

### Happy Path Tests
- Import validation: File structure and content integrity
- Domain loading: Experiment/Dataset/Session/Recording count verification
- Format detection: Both v3h and v3d CSV variants

### Negative Tests
- Malformed CSV handling (empty files, missing headers)
- Invalid filename patterns
- Error logging and graceful degradation

### Performance Tests
- Import timing with benchmark reporting (optional pytest-benchmark)
- Memory usage validation for larger datasets

## Why Keep This In-Repo?

- **Deterministic**: Same inputs every time, no randomness or external dependencies
- **Compact**: Few KB total, doesn't bloat repository
- **Realistic**: Exercises actual CSV formats and directory structures from real experiments
- **Version controlled**: Changes to test data are tracked and reviewable

## Usage Notes

- Importer auto-detects format (v3h/v3d) by scanning first few lines
- Session/recording IDs inferred from CSV filename structure
- Tests run in isolated temporary directories, no impact on real data
- Both threaded and single-threaded import modes tested
- Golden datasets are expected to contain ≥ 2 channels; enforced by `tests/test_golden_channel_counts.py`
