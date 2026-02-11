"""
QSettings Management and Migration Utility

This tool helps inspect, migrate, and manage QSettings across application versions.
Run with: python tools/settings_manager.py [command]

Commands:
  inspect    - Show all current settings with values
  version    - Check settings version and migration status
  migrate    - Migrate settings from old version (if needed)
  clear      - Clear all settings (prompts for confirmation)
  backup     - Export settings to JSON file
  restore    - Import settings from JSON backup
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path so we can import from monstim_gui
sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtCore import QCoreApplication, QSettings  # noqa: E402

# Constants
SETTINGS_VERSION_KEY = "SettingsVersion"
CURRENT_SETTINGS_VERSION = 1  # Increment when making breaking changes


def initialize_app():
    """Initialize QApplication with correct org/app names."""
    app = QCoreApplication(sys.argv)
    app.setOrganizationName("WorthyLab")
    app.setApplicationName("MonStim Analyzer")
    return app, QSettings()


def inspect_settings(settings: QSettings):
    """Display all current settings with their values."""
    print("\n" + "=" * 80)
    print("QSettings Inspection")
    print("=" * 80)
    print(f"Organization: {settings.organizationName()}")
    print(f"Application:  {settings.applicationName()}")
    print(f"Storage:      {settings.fileName()}")
    print(f"Format:       {'Native (Registry)' if settings.format() == QSettings.Format.NativeFormat else 'INI'}")

    all_keys = settings.allKeys()
    print(f"\nTotal Keys:   {len(all_keys)}")

    if not all_keys:
        print("\n⚠️  No settings found! This might be the first run or settings were cleared.")
        return

    # Group keys by category
    categories = {}
    for key in sorted(all_keys):
        category = key.split("/")[0] if "/" in key else "Root"
        if category not in categories:
            categories[category] = []
        categories[category].append(key)

    print("\n" + "-" * 80)
    print("Settings by Category:")
    print("-" * 80)

    for category, keys in sorted(categories.items()):
        print(f"\n[{category}] ({len(keys)} keys)")
        for key in keys:
            value = settings.value(key)
            # Truncate long values
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            print(f"  {key:50s} = {value_str}")

    # Check for session restoration specifically
    print("\n" + "-" * 80)
    print("Session Restoration Status:")
    print("-" * 80)
    session_restore_keys = [k for k in all_keys if k.startswith("SessionRestore/")]
    if session_restore_keys:
        print("✓ Session restoration data found:")
        for key in session_restore_keys:
            print(f"  {key} = {settings.value(key)}")
    else:
        print("⚠️  No SessionRestore/* keys found!")
        print("   This means last-opened experiment won't be restored on startup.")

        # Check if tracking is disabled
        track_pref = settings.value("ProgramPreferences/track_session_restoration", True, type=bool)
        if not track_pref:
            print("   → Session restoration tracking is DISABLED in preferences")
        else:
            print("   → Session restoration tracking is enabled, but no session saved yet")
            print("      (Try loading an experiment and the key should appear)")


def check_version(settings: QSettings):
    """Check settings version and determine if migration is needed."""
    print("\n" + "=" * 80)
    print("Settings Version Check")
    print("=" * 80)

    stored_version = settings.value(SETTINGS_VERSION_KEY, None, type=int)

    if stored_version is None:
        print("⚠️  No version key found (legacy settings or first run)")
        print(f"   Current version: {CURRENT_SETTINGS_VERSION}")
        print("   Recommendation: Run 'migrate' to set version tracking")
    elif stored_version == CURRENT_SETTINGS_VERSION:
        print(f"✓ Settings are up to date (version {CURRENT_SETTINGS_VERSION})")
    elif stored_version < CURRENT_SETTINGS_VERSION:
        print("⚠️  Settings are outdated!")
        print(f"   Stored version:  {stored_version}")
        print(f"   Current version: {CURRENT_SETTINGS_VERSION}")
        print("   Recommendation: Run 'migrate' to update settings")
    else:
        print("❌ Settings version is NEWER than expected!")
        print(f"   Stored version:  {stored_version}")
        print(f"   Current version: {CURRENT_SETTINGS_VERSION}")
        print("   This may indicate running an older app version with newer settings.")


def migrate_settings(settings: QSettings):
    """Migrate settings from old version to current version."""
    print("\n" + "=" * 80)
    print("Settings Migration")
    print("=" * 80)

    stored_version = settings.value(SETTINGS_VERSION_KEY, None, type=int)

    if stored_version == CURRENT_SETTINGS_VERSION:
        print(f"✓ Settings already at version {CURRENT_SETTINGS_VERSION}, no migration needed")
        return

    print(f"Migrating from version {stored_version or 'unversioned'} to {CURRENT_SETTINGS_VERSION}...")

    # Migration logic for each version bump
    if stored_version is None or stored_version < 1:
        print("\n  → Migration to v1:")
        print("     - Adding version tracking")
        print("     - No data changes needed (all existing keys preserved)")
        # Future migrations would go here, e.g.:
        # - Rename keys
        # - Convert data formats
        # - Remove obsolete keys

    # Set new version
    settings.setValue(SETTINGS_VERSION_KEY, CURRENT_SETTINGS_VERSION)
    settings.sync()

    print(f"\n✓ Migration complete! Settings now at version {CURRENT_SETTINGS_VERSION}")


def clear_settings(settings: QSettings, force=False):
    """Clear all settings (with confirmation)."""
    print("\n" + "=" * 80)
    print("Clear All Settings")
    print("=" * 80)

    all_keys = settings.allKeys()
    if not all_keys:
        print("No settings to clear.")
        return

    print(f"This will delete {len(all_keys)} settings keys from:")
    print(f"  {settings.fileName()}")

    if not force:
        response = input("\nAre you sure? Type 'YES' to confirm: ")
        if response != "YES":
            print("Cancelled.")
            return

    settings.clear()
    settings.sync()
    print(f"\n✓ All settings cleared ({len(all_keys)} keys deleted)")


def backup_settings(settings: QSettings, output_file: str):
    """Export settings to a JSON file."""
    print("\n" + "=" * 80)
    print("Backup Settings")
    print("=" * 80)

    all_keys = settings.allKeys()
    if not all_keys:
        print("⚠️  No settings to backup!")
        return

    backup_data = {
        "metadata": {
            "organization": settings.organizationName(),
            "application": settings.applicationName(),
            "version": settings.value(SETTINGS_VERSION_KEY, None),
            "total_keys": len(all_keys),
        },
        "settings": {},
    }

    for key in all_keys:
        value = settings.value(key)
        # Convert Qt types to JSON-serializable types
        if isinstance(value, (list, tuple)):
            value = list(value)
        elif hasattr(value, "__class__") and "PySide6" in value.__class__.__module__:
            value = str(value)
        backup_data["settings"][key] = value

    output_path = Path(output_file)
    output_path.write_text(json.dumps(backup_data, indent=2), encoding="utf-8")

    print(f"✓ Backed up {len(all_keys)} settings to: {output_path}")


def restore_settings(settings: QSettings, input_file: str, force=False):
    """Import settings from a JSON backup."""
    print("\n" + "=" * 80)
    print("Restore Settings")
    print("=" * 80)

    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    try:
        backup_data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to read backup file: {e}")
        return

    metadata = backup_data.get("metadata", {})
    backup_settings = backup_data.get("settings", {})

    print(f"Backup from: {metadata.get('organization')} / {metadata.get('application')}")
    print(f"Keys to restore: {len(backup_settings)}")

    current_keys = settings.allKeys()
    if current_keys and not force:
        print(f"\n⚠️  Current settings has {len(current_keys)} keys that will be overwritten!")
        response = input("Continue? Type 'YES' to confirm: ")
        if response != "YES":
            print("Cancelled.")
            return

    # Clear existing and restore from backup
    settings.clear()
    for key, value in backup_settings.items():
        settings.setValue(key, value)
    settings.sync()

    print(f"\n✓ Restored {len(backup_settings)} settings from backup")


def main():
    parser = argparse.ArgumentParser(
        description="QSettings management and migration utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command", choices=["inspect", "version", "migrate", "clear", "backup", "restore"], help="Command to execute"
    )
    parser.add_argument("--file", help="File path for backup/restore operations")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")

    args = parser.parse_args()

    # Initialize Qt application
    app, settings = initialize_app()

    # Execute command
    if args.command == "inspect":
        inspect_settings(settings)
    elif args.command == "version":
        check_version(settings)
    elif args.command == "migrate":
        migrate_settings(settings)
    elif args.command == "clear":
        clear_settings(settings, args.force)
    elif args.command == "backup":
        if not args.file:
            args.file = "qsettings_backup.json"
        backup_settings(settings, args.file)
    elif args.command == "restore":
        if not args.file:
            print("❌ --file argument required for restore command")
            return 1
        restore_settings(settings, args.file, args.force)

    return 0


if __name__ == "__main__":
    sys.exit(main())
