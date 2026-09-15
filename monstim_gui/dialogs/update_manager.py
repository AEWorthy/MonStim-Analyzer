"""Non-blocking UI for signed MonStim application updates."""

from __future__ import annotations

import os

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QCheckBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout

from monstim_gui.core.application_state import app_state
from monstim_gui.updates import (
    UpdateError,
    UpdateRelease,
    available_update,
    download_update,
    fetch_update_catalog,
    launch_staged_update,
    stage_update,
)


class UpdateCheckThread(QThread):
    complete = Signal(object)
    failed = Signal(str)

    def run(self) -> None:
        try:
            self.complete.emit(available_update(fetch_update_catalog()))
        except UpdateError as exc:
            self.failed.emit(str(exc))


class UpdateDownloadThread(QThread):
    complete = Signal(str)
    failed = Signal(str)

    def __init__(self, release: UpdateRelease, parent=None):
        super().__init__(parent)
        self.release = release

    def run(self) -> None:
        archive = None
        try:
            archive = download_update(self.release)
            stage_update(self.release, archive)
            self.complete.emit(self.release.version)
        except UpdateError as exc:
            self.failed.emit(str(exc))
        finally:
            if archive is not None:
                archive.unlink(missing_ok=True)


class UpdateManagerDialog(QDialog):
    """Show release trust details before any executable is downloaded or activated."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.release: UpdateRelease | None = None
        self.setWindowTitle("MonStim Updates")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        self.status = QLabel("Check the signed official update catalog for a newer Windows beta.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.auto_download = QCheckBox("Automatically download verified compatible updates", self)
        self.auto_download.setChecked(app_state.settings.value("Updates/auto_download", False, type=bool))
        self.auto_download.toggled.connect(lambda enabled: app_state.settings.setValue("Updates/auto_download", enabled))
        layout.addWidget(self.auto_download)
        self.notes = QPushButton("Open release notes", self)
        self.notes.setEnabled(False)
        self.notes.clicked.connect(self.open_notes)
        layout.addWidget(self.notes)
        buttons = QHBoxLayout()
        self.check = QPushButton("Check now", self)
        self.check.clicked.connect(self.check_now)
        buttons.addWidget(self.check)
        self.install = QPushButton("Download and install on restart", self)
        self.install.setEnabled(False)
        self.install.clicked.connect(self.download_and_install)
        buttons.addWidget(self.install)
        close = QPushButton("Close", self)
        close.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def check_now(self) -> None:
        self.check.setEnabled(False)
        self.status.setText("Checking the signed official update catalog…")
        self.check_thread = UpdateCheckThread(self)
        self.check_thread.complete.connect(self.check_complete)
        self.check_thread.failed.connect(self.check_failed)
        self.check_thread.start()

    def check_complete(self, release: UpdateRelease | None) -> None:
        self.check.setEnabled(True)
        self.release = release
        if release is None:
            self.status.setText("This MonStim installation is up to date for the beta update channel.")
            return
        self.status.setText(
            f"MonStim {release.version} is available. It will be verified before staging and will not alter research data, settings, or add-ons."
        )
        self.notes.setEnabled(True)
        self.install.setEnabled(True)

    def check_failed(self, message: str) -> None:
        self.check.setEnabled(True)
        self.status.setText(message)

    def open_notes(self) -> None:
        if self.release:
            QDesktopServices.openUrl(QUrl(self.release.notes_url))

    def download_and_install(self) -> None:
        if self.release is None:
            return
        accepted = QMessageBox.question(
            self,
            "Install verified update?",
            "MonStim will download and verify the official release, stage it beside the current application, then restart. "
            "Research data, settings, and add-ons will not be changed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if accepted != QMessageBox.StandardButton.Yes:
            return
        self.install.setEnabled(False)
        self.status.setText("Downloading and verifying update…")
        self.download_thread = UpdateDownloadThread(self.release, self)
        self.download_thread.complete.connect(self.download_complete)
        self.download_thread.failed.connect(self.download_failed)
        self.download_thread.start()

    def download_complete(self, version: str) -> None:
        try:
            launch_staged_update(version, wait_pid=os.getpid())
        except UpdateError as exc:
            self.status.setText(f"Update staged safely, but could not restart automatically: {exc}")
            return
        self.status.setText("Verified update staged. MonStim will now restart.")
        self.accept()
        self.parent().close() if self.parent() is not None else None

    def download_failed(self, message: str) -> None:
        self.install.setEnabled(True)
        self.status.setText(message)
