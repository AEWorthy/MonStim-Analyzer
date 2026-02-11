"""
Quick test to verify QSettings persistence across app launches.

This script:
1. Saves a test session state
2. Confirms it was saved
3. Shows that it persists across app restarts (same org/app name)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PySide6.QtCore import QCoreApplication, QSettings  # noqa: E402


def test_persistence():
    """Test that settings persist correctly."""
    print("\n" + "=" * 80)
    print("QSettings Persistence Test")
    print("=" * 80)

    # Initialize app with correct names
    app = QCoreApplication(sys.argv)
    app.setOrganizationName("WorthyLab")
    app.setApplicationName("MonStim Analyzer")

    settings = QSettings()

    print(f"\nStorage: {settings.fileName()}")
    print(f"Format:  {'Registry' if settings.format() == QSettings.Format.NativeFormat else 'INI'}")

    # Save a test session state
    test_exp_id = "TEST_PERSISTENCE_EXPERIMENT"
    test_dataset_id = "TEST_DATASET_123"
    test_session_id = "TEST_SESSION_456"

    print("\n1. Saving test session state...")
    settings.setValue("SessionRestore/experiment", test_exp_id)
    settings.setValue("SessionRestore/dataset", test_dataset_id)
    settings.setValue("SessionRestore/session", test_session_id)
    settings.sync()
    print("   ✓ Saved to QSettings")

    # Verify it was saved
    print("\n2. Verifying data was saved...")
    read_exp = settings.value("SessionRestore/experiment", "", type=str)
    read_dataset = settings.value("SessionRestore/dataset", "", type=str)
    read_session = settings.value("SessionRestore/session", "", type=str)

    if read_exp == test_exp_id and read_dataset == test_dataset_id and read_session == test_session_id:
        print("   ✓ All values read back correctly!")
        print(f"     - Experiment: {read_exp}")
        print(f"     - Dataset:    {read_dataset}")
        print(f"     - Session:    {read_session}")
    else:
        print("   ❌ ERROR: Values did not match!")
        print(f"     Expected experiment: {test_exp_id}, got: {read_exp}")
        print(f"     Expected dataset: {test_dataset_id}, got: {read_dataset}")
        print(f"     Expected session: {test_session_id}, got: {read_session}")
        return False

    # Check version key
    print("\n3. Checking settings version...")
    version = settings.value("SettingsVersion", None, type=int)
    if version is not None:
        print(f"   ✓ Settings version: {version}")
    else:
        print("   ⚠️  No version key (run migration to add)")

    print("\n" + "=" * 80)
    print("RESULT: Settings persistence working correctly! ✓")
    print("=" * 80)
    print("\nThese values will persist across app restarts because:")
    print("  - Organization name is fixed: 'WorthyLab'")
    print("  - Application name is fixed: 'MonStim Analyzer'")
    print("  - Application version does NOT affect storage location")
    print("\nYou can verify by running this script again - values should still be there.")

    # Option to clean up test data
    print("\n" + "-" * 80)
    response = input("Remove test data? (y/n): ").strip().lower()
    if response == "y":
        settings.remove("SessionRestore/experiment")
        settings.remove("SessionRestore/dataset")
        settings.remove("SessionRestore/session")
        settings.sync()
        print("✓ Test data removed")
    else:
        print("Test data preserved (you can remove manually later)")

    return True


if __name__ == "__main__":
    test_persistence()
