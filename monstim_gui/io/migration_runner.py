"""Background annotation migration runner.

Runs scan + migrations in a QThread and emits progress/status updates.
"""

import logging
from pathlib import Path
from typing import List

from PySide6.QtCore import QThread, Signal

from monstim_signals.io.data_migrations import migrate_annotation_dict, scan_annotation_versions


class MigrationRunner(QThread):
    finished = Signal(int)  # number of files migrated
    error = Signal(str)
    progress = Signal(int)
    status_update = Signal(str)
    canceled = Signal()  # Emitted when canceled

    def __init__(self, experiment_path: str):
        super().__init__()
        self.experiment_path = experiment_path
        self._cancel_requested = False

    def request_cancel(self):
        """Request graceful cancellation of migration."""
        logging.info("Cancellation requested for migration runner")
        self._cancel_requested = True

    def run(self) -> None:
        try:
            exp_path = Path(self.experiment_path)
            self.status_update.emit("Scanning annotations for migrations...")

            results = scan_annotation_versions(exp_path)
            to_migrate: List[dict] = [r for r in results if r.get("needs_migration")]
            total = len(to_migrate)
            if not total:
                self.progress.emit(100)
                self.status_update.emit("No migrations required.")
                self.finished.emit(0)
                return

            migrated = 0
            self.status_update.emit(f"Applying migrations for {total} files...")

            # Apply migrations: open each file, migrate in-place, and write back.
            for i, item in enumerate(to_migrate, start=1):
                # Check for cancellation
                if self._cancel_requested:
                    logging.info(f"Migration canceled after {migrated}/{total} files")
                    self.canceled.emit()
                    return

                try:
                    path = Path(item["path"]) if "path" in item else None
                    if not path:
                        continue
                    import json

                    try:
                        data = json.loads(path.read_text())
                    except Exception:
                        logging.exception("Failed to read annotation: %s", path)
                        continue

                    report = migrate_annotation_dict(data, in_place=True, strict_version=False)
                    if report.changed:
                        try:
                            path.write_text(json.dumps(data, indent=2))
                            migrated += 1
                        except Exception:
                            logging.exception("Failed to write migrated annotation: %s", path)
                    pct = 10 + int(90 * (i / total))
                    self.progress.emit(pct)
                except Exception:
                    logging.exception("Migration failed for item: %s", item.get("path"))
                    # Continue migrating others
                    continue

            self.progress.emit(100)
            self.status_update.emit(f"Migrations complete: {migrated}/{total} files updated.")
            self.finished.emit(migrated)
        except Exception as e:
            logging.error(f"Migration runner failed: {e}")
            self.error.emit(str(e))


class MigrationScanThread(QThread):
    has_work = Signal(bool, int)  # (needs_migration, count)
    error = Signal(str)

    def __init__(self, experiment_path: str):
        super().__init__()
        self.experiment_path = experiment_path
        self._cancel_requested = False

    def request_cancel(self):
        """Request graceful cancellation of migration scan."""
        logging.debug("Cancellation requested for migration scan")
        self._cancel_requested = True

    def run(self) -> None:
        try:
            if self._cancel_requested:
                return
            results = scan_annotation_versions(Path(self.experiment_path))
            if self._cancel_requested:
                return
            to_migrate = [r for r in results if r.get("needs_migration")]
            self.has_work.emit(bool(to_migrate), len(to_migrate))
        except Exception as e:
            self.error.emit(str(e))
