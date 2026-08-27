# Architecture

## Purpose

MonStim separates scientific/domain work from the Qt user interface. Preserve that separation so calculations remain testable and the interface can refresh safely as selected data changes.

## Main components

| Component | Owns | Key locations |
| --- | --- | --- |
| Domain | Experiment, dataset, session, recording behavior; transforms and aggregation | `monstim_signals/domain`, `monstim_signals/transform` |
| Persistence | CSV import, HDF5/JSON storage, annotation migration, catalog support | `monstim_signals/io` |
| GUI | Selection state, menus, dialogs, plotting, and presentation | `monstim_gui` |
| Commands | Undoable mutations and their history | `monstim_gui/commands.py` |
| Configuration | Default profile, overrides, and typed resolution | `monstim_signals/core/configuration.py`, `docs/resources` |

## Ownership and persistence

The hierarchy is experiment > dataset > session > recording. Annotation overlays carry non-destructive edits. A session owns latency windows; dataset and experiment actions fan one chosen window set out to affected sessions. Repositories, not widgets, are responsible for writing persistent domain state.

Application state such as window geometry, selected IDs, and preferences is separate from analysis data and is stored through Qt QSettings. See [Application settings](qsettings_management.md).

## Change flow

1. A GUI action gathers user intent.
2. A command or manager applies an approved domain change.
3. The repository persists the changed annotations and updates derived catalog data when needed.
4. The GUI refreshes selection, notices, plots, and undo/redo availability.

Avoid putting domain mutations directly in widgets. Avoid making a GUI component the source of truth for a persisted value.

## Related topics

- [Commands and undo](command_testing_strategy.md)
- [Annotation data versions](data_versioning.md)
- [Testing](testing.md)
