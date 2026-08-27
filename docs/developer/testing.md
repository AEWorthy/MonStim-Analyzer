# Testing

## Purpose

Tests should demonstrate the behavior a change protects, not merely execute code. Use the `monstim` environment for every project command.

## Run tests

```powershell
# Focused test while developing
conda run -n monstim python -m pytest tests/gui/test_help_navigation.py -q -p no:cacheprovider --basetemp=.pytest_tmp_help

# Default suite (legacy tests remain excluded by pytest configuration)
conda run -n monstim python -m pytest

# Include legacy-marked tests when intentionally checking them
conda run -n monstim python -m pytest -m "legacy or not legacy"
```

Use `-k <expression>` or a precise node ID to narrow a failure. For GUI tests, keep the offscreen configuration used by the test suite and avoid relying on timing or a visible desktop.

## Test design

- Put pure signal and domain behavior in focused unit or domain tests.
- Exercise repository, import/export, command, and hierarchy behavior with integration tests.
- Use `tmp_path` or pytest-managed temporary locations for every write.
- Treat files under `tests/fixtures` as read-only inputs.
- Verify an undo/redo round trip for every undoable mutation.
- For a bug fix, add the smallest regression test that fails before the fix.

## Windows notes

Cloud-synced worktrees can leave temporary test folders locked. A cleanup failure is an environment problem, not proof that tests passed or failed. Use a unique `--basetemp`, close file handles, and report cleanup failures separately from test results.

## Before handoff

```powershell
conda run -n monstim ruff check <touched Python paths>
git diff --check
```

Run broader tests when the change crosses domain, persistence, or GUI boundaries.

## Related topics

- [Contributing](contributing.md)
- [Commands and undo](command_testing_strategy.md)
