# Annotation data versions

## Purpose

Annotation JSON overlays evolve independently of the application version. `monstim_signals/version.py` defines the current `DATA_VERSION`; `monstim_signals/io/data_migrations.py` upgrades older annotation dictionaries before domain dataclasses are created.

## Stored overlays

| Level | Overlay file |
| --- | --- |
| Recording | `<stem>.annot.json` |
| Session | `session.annot.json` |
| Dataset | `dataset.annot.json` |
| Experiment | `experiment.annot.json` |

Each overlay includes `data_version`. Raw signal files are treated as immutable acquisition data; do not retrofit an annotation migration into raw data.

## Add a migration

1. Define the target version in `monstim_signals/version.py`.
2. Add an ordered `MigrationStep` in `data_migrations.py` from the immediately preceding version.
3. Make the step tolerate missing legacy fields and set the new `data_version`.
4. Keep the migration idempotent and preserve unrelated fields unless the migration explicitly replaces a legacy schema.
5. Add tests in `tests/io/test_data_migrations.py` for the forward path, idempotence, and expected failure cases.

The migration API supports dry runs and returns a report. Newer stored data must raise a clear future-version error rather than being silently downgraded.

## Review checklist

- Confirm the migration chain reaches the new version with no gaps.
- Decide whether a derived cache or catalog needs invalidation after the semantic change.
- Confirm repositories persist successful migrations only after the complete upgrade succeeds.
- Document externally visible changes in release notes and relevant user help.

## Related topics

- [Architecture](architecture.md)
- [Testing](testing.md)
