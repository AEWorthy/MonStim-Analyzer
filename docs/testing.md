# Testing Guide (Developers)

This project uses pytest with a curated set of golden fixtures to provide fast, deterministic, and hermetic tests across domain logic, import/export workflows, and GUI-adjacent behaviors.

## Environment

- Recommended: the `monstim` conda environment

```powershell
conda activate monstim
```

- If `monstim` isn’t available, install directly:

```powershell
python -m pip install -r requirements.txt
```

GUI tests are configured to run headlessly via `tests/conftest.py`; no extra flags are required.

## Running tests

- Default stable run (excludes legacy):

```powershell
pytest
```

- Include legacy tests as well:

```powershell
pytest -m "legacy or not legacy"
```

- Examples:
  - Single file: `pytest tests/test_signal_processing.py -q`
  - Single test: `pytest tests/test_signal_processing.py::test_butter_filter -q`
  - Only integration: `pytest -m integration`
  - Only unit: `pytest -m unit`

Markers are defined in `pytest.ini`:
- `unit` — fast, isolated tests
- `integration` — cross-component flows
- `slow` — performance or large data
- `legacy` — quarantined tests excluded by default (`addopts = -q -m "not legacy"`)

## Data policy: golden-only

- Do NOT use the repository `data/` folder in tests.
- All real-data tests must use curated CSVs under `tests/fixtures/golden/`.
- Tests import CSVs into a temporary directory using:

  ```python
  from monstim_signals.io.csv_importer import import_experiment
  import_experiment(golden_root, tmp_out_dir, overwrite=True, max_workers=1)
  ```

- The golden set includes both valid datasets and an `invalid/` folder for negative tests (malformed files, wrong naming, and empty datasets). Tests must not modify fixtures in-place; treat them as read-only inputs.

Cleanup: If a local run creates a top-level `.pytest-tmp-golden-check` folder, a session-scoped autouse fixture removes it after tests. The path is also ignored by git.

### Channel count guarantee

Golden datasets are expected to contain at least 2 channels per session/recording. The test `tests/test_golden_channel_counts.py` enforces this by verifying both `.meta.json` (`num_channels >= 2`) and `.raw.h5` (`raw.shape[1] >= 2`).

## Writing tests

- Prefer unit tests for pure-domain logic and small utilities; add `@pytest.mark.unit`.
- Use integration tests for importer/repository flows and command pattern interactions; add `@pytest.mark.integration` (and `@pytest.mark.slow` if it imports all golden data).
- Use `tmp_path`/`tmp_path_factory` for any filesystem writes; never write under the repo tree.
- For importer tests, always import into a temp dir; never mutate `tests/fixtures/golden/`.
- For GUI-adjacent logic, keep PyQt imports out of `monstim_signals` (domain must remain GUI-agnostic).
- When renaming channels, note that duplicate names are rejected. Domain code raises `ValueError` preemptively; there are tests for this at both domain and command layers.

## Known quarantined tests

- `tests/test_domain_business_logic.py` — legacy monolith, replaced by focused tests; kept under the `legacy` marker for reference.

## Troubleshooting

-- Ensure you’re in the right environment (conda `monstim`) so scientific and PyQt dependencies resolve.
- If a test appears to use data from `data/`, refactor it to import from golden fixtures into a temp dir.
- For plotting-related failures, follow the error-handling policy: `UnableToPlotError` should be handled explicitly in plotting paths.
