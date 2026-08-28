# Application settings

## Purpose

Qt QSettings stores application state that should survive restarts. It is not the source of scientific data, annotations, or analysis profiles. The implementation is `monstim_gui/core/application_state.py`.

## What is stored

| Group | Examples |
| --- | --- |
| `SessionRestore` | selected experiment, dataset, session, and profile |
| `LastSelection` | most recently selected profile |
| `LastPaths` | import and export folders |
| `RecentFiles` | recent experiment identifiers |
| `ProgramPreferences` | restoration, path tracking, profile tracking, and loading preferences |

The Qt organization and application names determine the platform-specific storage location. Changing either effectively creates a different settings store, so treat it as a migration decision.

## Change a setting safely

1. Add a clearly named key under the appropriate group.
2. Read it through `ApplicationState` with an explicit default.
3. Respect the user preference that enables or disables the related tracking.
4. Add a focused test using mocked QSettings.
5. Document a user-visible preference in the relevant user help.

## Settings migrations

`SETTINGS_VERSION` is currently defined in `application_state.py`. Increase it only when key names or value shapes become incompatible, then add the corresponding logic to `_migrate_settings()`. Test a prior-version store, a clean store, and a store newer than the running application.

## Recovery and diagnostics

`ApplicationState.get_settings_diagnostics()` exposes stored groups for troubleshooting. `clear_all_tracked_data()` removes tracked state while retaining preferences; `clear_all_settings()` removes the complete settings store. Keep recovery actions explicit and supportable—do not clear settings automatically to work around a startup issue.

## Related topics

- [Architecture](architecture.md)
- [Testing](testing.md)
