# QSettings Management and Best Practices

## Overview

MonStim Analyzer uses Qt's `QSettings` to persist user preferences and application state across sessions. This document explains how settings work, how to manage them, and answers common questions about version upgrades.

## Storage Location

**Windows (Current Platform):**
- Format: Native (Windows Registry)
- Location: `HKEY_CURRENT_USER\Software\WorthyLab\MonStim Analyzer`
- Organization: `WorthyLab`
- Application: `MonStim Analyzer`

The storage location is determined by the organization name and application name set in [main.py](../main.py#L132-L133).

## Settings Structure

Settings are organized into categories using `/` as a separator:

```
SettingsVersion                         # Version tracking for migrations (v0.5.3+)
SessionRestore/
  ├─ experiment                         # Last opened experiment ID
  ├─ dataset                            # Last opened dataset ID  
  ├─ session                            # Last opened session ID
  └─ profile                            # Last selected analysis profile
LastSelection/
  └─ profile                            # Last selected profile (always saved)
LastPaths/
  ├─ import_directory                   # Last import location
  └─ export_directory                   # Last export location
ProgramPreferences/
  ├─ track_session_restoration          # Enable/disable session restoration
  ├─ track_import_export_paths          # Enable/disable path memory
  ├─ track_recent_files                 # Enable/disable recent files
  ├─ track_analysis_profiles            # Enable/disable profile tracking
  ├─ use_opengl_acceleration            # OpenGL rendering preference
  ├─ use_lazy_open_h5                   # Lazy HDF5 loading preference
  ├─ enable_parallel_loading            # Parallel dataset loading
  └─ build_index_on_load                # Rebuild indexes on load
RecentFiles/
  └─ experiments                        # List of recent experiment paths
RecentProfiles/
  └─ names                              # List of recent profile names
```

### How to Troubleshoot/Manage Settings

Use the settings management utility:

```powershell
conda activate alv_lab
python tools/settings_manager.py inspect
```

This will show:
- All current settings with values
- Whether `SessionRestore/*` keys exist
- Whether session restoration tracking is enabled
- Storage location and format

## Settings Version and Migration

As of version 0.5.3, MonStim Analyzer tracks settings version using the `SettingsVersion` key. This enables automatic migration when settings structure changes between releases.

### How Migration Works

1. On startup, `ApplicationState.reinitialize_settings()` checks the stored version
2. If stored version < current version, `_migrate_settings()` is called automatically
3. Migration logic can rename keys, convert data formats, or remove obsolete settings
4. New version number is saved after successful migration

### Adding a New Migration

When you need to make breaking changes to settings structure:

1. Increment `SETTINGS_VERSION` in [monstim_gui/core/application_state.py](../monstim_gui/core/application_state.py)
2. Add migration logic in `_migrate_settings()`:

```python
def _migrate_settings(self, from_version: int, to_version: int):
    """Perform migration of settings between versions."""
    
    # Example: Migrating from v1 to v2
    if from_version < 2:
        # Rename a key
        old_val = self.settings.value("OldKeyName")
        if old_val is not None:
            self.settings.setValue("NewKeyName", old_val)
            self.settings.remove("OldKeyName")
        
        # Convert data format
        old_list = self.settings.value("OldList", [], type=list)
        new_dict = {item: True for item in old_list}
        self.settings.setValue("NewDict", new_dict)
    
    logging.info(f"Settings migration from v{from_version} to v{to_version} complete")
```

3. Test the migration by:
   - Backing up current settings: `python tools/settings_manager.py backup --file backup.json`
   - Manually setting version to old: Settings → SettingsVersion = 1
   - Restarting app and verifying migration happens
   - Restoring backup if needed: `python tools/settings_manager.py restore --file backup.json`


## Common Settings Manager Operations

### Inspect Current Settings
```powershell
python tools/settings_manager.py inspect
```

### Check Version and Migration Status  
```powershell
python tools/settings_manager.py version
```

### Manually Trigger Migration
```powershell
python tools/settings_manager.py migrate
```

### Backup Settings to JSON
```powershell
python tools/settings_manager.py backup --file my_backup.json
```

### Restore Settings from Backup
```powershell
python tools/settings_manager.py restore --file my_backup.json
```

### Clear All Settings (Nuclear Option)
```powershell
python tools/settings_manager.py clear
```

### Programmatic Access in Code

```python
from monstim_gui.core.application_state import app_state

# Get diagnostics
diagnostics = app_state.get_settings_diagnostics()
print(f"Settings version: {diagnostics['version']}")
print(f"Session restore enabled: {diagnostics['session_restore_enabled']}")
print(f"Last experiment: {diagnostics['session_restore'].get('SessionRestore/experiment')}")

# Clear only user data (preserve preferences)
app_state.clear_all_tracked_data()

# Clear everything (including preferences)
app_state.clear_all_settings()
```

## Troubleshooting

### Problem: Last-opened experiment not restoring

**Check 1**: Is session restoration enabled?
```powershell
python tools/settings_manager.py inspect
# Look for: ProgramPreferences/track_session_restoration = true
```

**Check 2**: Do SessionRestore keys exist?
```powershell
python tools/settings_manager.py inspect
# Look for: SessionRestore/experiment, SessionRestore/dataset, etc.
```

**Fix**: If tracking is disabled, enable it in application preferences or manually:
```python
from monstim_gui.core.application_state import app_state
app_state.set_setting("track_session_restoration", True)
```

### Problem: Settings seem corrupted or causing crashes

**Solution**: Backup and clear settings, then restart:
```powershell
python tools/settings_manager.py backup --file emergency_backup.json
python tools/settings_manager.py clear
# Restart application - fresh settings will be created
```

### Problem: Want to transfer settings to another computer

**Solution**: Use backup/restore:
```powershell
# On old computer:
python tools/settings_manager.py backup --file settings_export.json

# Copy settings_export.json to new computer, then:
python tools/settings_manager.py restore --file settings_export.json
```

## Best Practices for Developers

1. **Never change org/app names** in [main.py](../main.py) without a migration plan - this changes storage location.

2. **Increment SETTINGS_VERSION** when making breaking changes to settings structure

3. **Use descriptive key names** with category prefixes: `Category/subcategory/setting_name`

4. **Always provide defaults** when reading settings:
   ```python
   value = app_state.get_preference("my_setting", default_value=True)
   ```

5. **Document new settings** in this file when adding them

6. **Test migration logic** thoroughly before release - settings corruption is hard to recover from

7. **Log settings operations** at DEBUG level for troubleshooting

8. **Respect user preferences** - don't override settings without user action

## References

- Qt Documentation: [QSettings Class](https://doc.qt.io/qt-6/qsettings.html)
- Implementation: [monstim_gui/core/application_state.py](../monstim_gui/core/application_state.py)
- Initialization: [main.py](../main.py#L132-L147)
- Management Utility: [tools/settings_manager.py](../tools/settings_manager.py)
