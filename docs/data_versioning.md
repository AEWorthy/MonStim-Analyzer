# Data Versioning & Annotation Migration

MonStim Analyzer stores user edits in lightweight JSON *annotation overlay* files:

- Recording: `<stem>.annot.json`
- Session: `session.annot.json`
- Dataset: `dataset.annot.json`
- Experiment: `experiment.annot.json`

Each file contains a `data_version` key describing the schema/semantics of the
annotation payload. This is **independent** from the Python package version.

Current data version: defined in `monstim_signals/version.py` as `DATA_VERSION`.

## Why Version?
As the application evolves we may rename keys, introduce new optional fields, or
normalize legacy structures. Older annotation files must remain loadable.
Instead of embedding many conditional branches in repository code, we apply a
**forward migration pipeline** that upgrades an old annotation dict to the
current schema *before* it is converted into its dataclass (`SessionAnnot`,
`DatasetAnnot`, etc.).

## Migration Framework
Implemented in `monstim_signals/io/data_migrations.py`:

- `MigrationStep`: describes `from_version -> to_version` and the function.
- `MIGRATIONS`: ordered list of steps; new steps are appended only.
- `migrate_annotation_dict(raw: dict) -> MigrationReport`: main entry point.
- `FutureVersionError`: raised if stored version is newer than the code knows.
- `UnknownVersionError`: raised for malformed / unrecognized versions.

Missing or empty `data_version` is treated as legacy `'1.0.0'`.

### Idempotency
Each migration function must be safe to run multiple times without further
changes after the first application. This keeps recovery simple if a partially
migrated file is encountered.

### Example Step
```
# 1.0.0 -> 2.0.0
if 'is_completed' not in data:
    data['is_completed'] = False
# rename legacy key
if 'excluded_session_ids' in data and 'excluded_sessions' not in data:
    data['excluded_sessions'] = data.pop('excluded_session_ids')
```

## Adding a New Migration
1. Decide new target version (e.g. bump `DATA_VERSION` from `2.0.0` to `2.1.0`).
2. Implement function:
   ```python
   def migrate_2_0_0_to_2_1_0(data: dict) -> dict:
       if 'new_field' not in data:
           data['new_field'] = []
       data['data_version'] = '2.1.0'
       return data
   ```
3. Append to `MIGRATIONS`:
   ```python
   MIGRATIONS.append(MigrationStep('2.0.0', '2.1.0', migrate_2_0_0_to_2_1_0))
   ```
4. Update `DATA_VERSION = "2.1.0"` in `version.py`.
5. Add / update tests in `tests/test_data_migrations.py`.
6. Document changes in `CHANGELOG.md` under a *Data Versioning* subsection.

## Repository Integration
Repositories (`repositories.py`) call `migrate_annotation_dict` immediately
after reading a JSON annotation file and before instantiating dataclasses. If
a migration occurs, the upgraded JSON is saved back to disk, so subsequent loads
are already up to date.

## Error Handling
- If an annotation has a future version (e.g., `3.0.0` but code only knows `2.x`), loading aborts with a clear error: user should upgrade their software.
- Unknown or malformed versions log a warning and skip migration but still try parsing; this favors availability over strict failure.

## Other Versioned Concerns (Future Work)
| Area | Rationale | Suggested Approach |
|------|-----------|-------------------|
| M-max algorithm parameters | If detection heuristics change, cached `m_max_values` may become stale | Add `mmax_algo_version` field to session/dataset annotations; invalidate caches if mismatch |
| Analysis profiles (`docs/analysis_profiles/*.yml`) | Parameter structure could evolve | Add `profile_version` inside YAML; provide translation functions on load |
| Exported reports | Structural changes to report JSON/CSV | Embed `report_schema_version` at top-level |

## Quick FAQ
**Q: Do we store a migration history inside the file?** No; only the final
`data_version` is stored. History is recoverable from VCS if needed.

**Q: Can we downgrade?** Downgrades are not supported; use source control or
backup copies if you need older formats.

**Q: Should we migrate raw signal HDF5 files?** Not presently; raw acquisition
is assumed immutable. Any future transformations should write *new* datasets or
metadata keys with their own version tags.

## Testing Strategy
- Unit tests target migration of synthetic legacy dicts.
- Integration tests indirectly validate by loading actual repository objects.
- When adding a migration, add a fixture representing the pre-migration shape.

## Maintenance Checklist
- [ ] Update `DATA_VERSION`
- [ ] Add migration function + registry step
- [ ] Add / update tests
- [ ] Update docs (this file + CHANGELOG)
- [ ] Consider cache invalidation impacts (e.g., m_max)

