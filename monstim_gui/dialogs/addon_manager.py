"""User-facing installer and diagnostics for official importer add-ons."""

from __future__ import annotations

import os
from pathlib import Path

from packaging.version import Version
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QCheckBox, QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget, QMessageBox, QPushButton, QVBoxLayout

from monstim_gui.core.application_state import app_state
from monstim_gui.plugins import PluginError, diagnostic_report, download_pack, fetch_official_catalog, install_pack, installed_manifests, plugin_root


class AddonManagerDialog(QDialog):
    """Install local official packs and expose actionable compatibility details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add-on Manager")
        self.resize(680, 410)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Importer add-ons execute code. Install only packs obtained from the official MonStim GitHub Releases page. "
                "MonStim checks the pack structure, compatibility, and declared runtime requirements before installation."
            )
        )
        displayed_plugin_root = "[User addons/plugins location]" if os.environ.get("MONSTIM_DOCS_SCREENSHOT") else str(plugin_root())
        self.location_label = QLabel(f"Installed add-ons: <code>{displayed_plugin_root}</code>")
        self.location_label.setTextFormat(Qt.TextFormat.RichText)
        self.location_label.setWordWrap(True)
        layout.addWidget(self.location_label)
        self.auto_updates = QCheckBox("Automatically install compatible updates to already-installed official add-ons", self)
        self.auto_updates.setChecked(app_state.settings.value("Addons/auto_install_updates", False, type=bool))
        self.auto_updates.toggled.connect(lambda enabled: app_state.settings.setValue("Addons/auto_install_updates", enabled))
        layout.addWidget(self.auto_updates)
        self.list_widget = QListWidget(self)
        layout.addWidget(self.list_widget)
        buttons = QHBoxLayout()
        install = QPushButton("Install Add-on ZIP…", self)
        install.clicked.connect(self.install_local)
        buttons.addWidget(install)
        check = QPushButton("Check Official Updates", self)
        check.clicked.connect(self.check_official_updates)
        buttons.addWidget(check)
        copy_diagnostics = QPushButton("Copy Diagnostics", self)
        copy_diagnostics.clicked.connect(lambda: QGuiApplication.clipboard().setText(diagnostic_report()))
        buttons.addWidget(copy_diagnostics)
        close = QPushButton("Close", self)
        close.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.refresh()

    def check_official_updates(self):
        try:
            catalog = fetch_official_catalog()
        except PluginError as exc:
            QMessageBox.information(self, "Official add-ons unavailable", str(exc))
            return
        available = catalog.get("plugins", [])
        if not available:
            QMessageBox.information(self, "Official add-ons", "No official importer packs are currently available for this MonStim release.")
            return
        installed = {manifest.id: manifest.version for manifest, _path, compatible in installed_manifests() if compatible}
        candidates = []
        for item in available:
            try:
                if {"id", "name", "version", "asset_url", "sha256"} <= item.keys() and (
                    item["id"] not in installed or Version(item["version"]) > Version(installed[item["id"]])
                ):
                    candidates.append(item)
            except Exception:
                continue
        if not candidates:
            QMessageBox.information(self, "Official add-ons", "Installed official add-ons are up to date for this catalog.")
            return
        labels = [f"{item['name']} {item['version']}" for item in candidates]
        label, accepted = QInputDialog.getItem(self, "Official add-on update", "Available pack:", labels, 0, False)
        if not accepted:
            return
        item = candidates[labels.index(label)]
        self.install_official(item)

    def install_official(self, item: dict):
        confirm = QMessageBox.question(
            self,
            "Install official add-on?",
            f"Install {item['name']} {item['version']} from the signed MonStim catalog? Review its release notes before continuing.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        downloaded = None
        try:
            downloaded = download_pack(item["asset_url"], item["sha256"])
            manifest = install_pack(downloaded)
        except (PluginError, OSError) as exc:
            resolution = "Check your connection, update MonStim, choose a compatible pack, or contact support."
            QMessageBox.warning(self, "Official add-on not installed", f"{exc}\n\n{resolution}")
            return
        finally:
            if downloaded is not None:
                downloaded.unlink(missing_ok=True)
        QMessageBox.information(self, "Official add-on installed", f"{manifest.name} {manifest.version} is ready to use.")
        self.refresh()

    def refresh(self):
        self.list_widget.clear()
        records = installed_manifests()
        if not records:
            self.list_widget.addItem("No importer add-ons are installed.")
            return
        for manifest, _path, compatible in records:
            status = "Ready" if compatible else "Incompatible"
            self.list_widget.addItem(f"{manifest.name} {manifest.version} — {status}\n{manifest.description}")

    def install_local(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Install MonStim Importer Add-on", str(plugin_root()), "MonStim Add-on (*.zip)")
        if not filename:
            return
        confirm = QMessageBox.warning(
            self,
            "Install executable add-on?",
            "This add-on contains Python code and will run inside MonStim. Continue only if you downloaded it from the official "
            "MonStim release page.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            manifest = install_pack(Path(filename))
        except PluginError as exc:
            QMessageBox.warning(
                self, "Add-on not installed", f"{exc}\n\nResolution: update MonStim, choose a compatible add-on release, or contact MonStim support."
            )
            return
        QMessageBox.information(self, "Add-on installed", f"{manifest.name} {manifest.version} is ready to use.")
        self.refresh()
