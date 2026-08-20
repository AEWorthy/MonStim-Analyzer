"""SQLite-backed, rebuildable catalog for experiment discovery and cached metadata.

The catalog is deliberately non-authoritative: the experiment folder's HDF5 and
JSON files remain the source of truth.  It avoids the repeated directory walks
and large JSON index deserializations that made large experiment opens slow.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_FILENAME = ".monstim-cache.sqlite"
CATALOG_SCHEMA_VERSION = 1
RAW_SUFFIX = ".raw.h5"


@dataclass(frozen=True)
class CatalogRecording:
    """The persisted metadata needed to construct a recording repository."""

    stem: Path
    raw_path: Path
    meta_json: str
    annot_json: str | None
    primary_stim_v: float | None


def recording_stem(raw_path: Path) -> Path:
    """Return the source stem for ``<stem>.raw.h5`` without suffix ambiguity."""
    name = raw_path.name
    if not name.endswith(RAW_SUFFIX):
        raise ValueError(f"Expected a {RAW_SUFFIX} recording path, got {raw_path}")
    return raw_path.with_name(name[: -len(RAW_SUFFIX)])


def catalog_path(experiment_path: Path) -> Path:
    return experiment_path / CATALOG_FILENAME


def _file_fingerprint(path: Path) -> tuple[int | None, float | None]:
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime
    except FileNotFoundError:
        return None, None


def _read_text_if_exists(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else None
    except OSError:
        logger.warning("Could not read catalog source %s", path, exc_info=True)
        return None


def _primary_stim_voltage(meta_text: str) -> float | None:
    try:
        meta = json.loads(meta_text)
        primary = meta.get("primary_stim")
        if isinstance(primary, dict):
            value = primary.get("stim_v")
        elif isinstance(primary, int) and primary > 0:
            clusters = meta.get("stim_clusters", [])
            value = clusters[primary - 1].get("stim_v") if primary <= len(clusters) else None
        else:
            value = None
        return float(value) if value is not None else None
    except IndexError, TypeError, ValueError, json.JSONDecodeError:
        return None


class ExperimentCatalog:
    """Read-only access to a single experiment's SQLite catalog."""

    def __init__(self, experiment_path: Path):
        self.experiment_path = experiment_path.resolve()
        self.path = catalog_path(self.experiment_path)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def is_usable(self) -> bool:
        if not self.path.is_file():
            return False
        try:
            with self.connect() as connection:
                row = connection.execute("SELECT value FROM catalog_meta WHERE key = 'schema_version'").fetchone()
                root = connection.execute("SELECT value FROM catalog_meta WHERE key = 'experiment_path'").fetchone()
            return (
                row is not None and row["value"] == str(CATALOG_SCHEMA_VERSION) and root is not None and Path(root["value"]) == self.experiment_path
            )
        except sqlite3.Error:
            logger.warning("Catalog %s is unreadable and will be rebuilt", self.path, exc_info=True)
            return False

    def dataset_paths(self) -> list[Path]:
        with self.connect() as connection:
            rows = connection.execute("SELECT path FROM datasets ORDER BY sort_name, id").fetchall()
        return [Path(row["path"]) for row in rows]

    def session_paths(self, dataset_path: Path) -> list[Path]:
        with self.connect() as connection:
            rows = connection.execute("SELECT path FROM sessions WHERE dataset_path = ? ORDER BY sort_name, id", (str(dataset_path),)).fetchall()
        return [Path(row["path"]) for row in rows]

    def recordings(self, session_path: Path) -> list[CatalogRecording]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT stem, raw_path, meta_json, annot_json, primary_stim_v
                FROM recordings WHERE session_path = ?
                ORDER BY primary_stim_v IS NULL, primary_stim_v, sort_name, stem
                """,
                (str(session_path),),
            ).fetchall()
        return [
            CatalogRecording(
                stem=Path(row["stem"]),
                raw_path=Path(row["raw_path"]),
                meta_json=row["meta_json"],
                annot_json=row["annot_json"],
                primary_stim_v=row["primary_stim_v"],
            )
            for row in rows
        ]


def _initialize(connection: sqlite3.Connection, experiment_path: Path) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE datasets (
            path TEXT PRIMARY KEY, id TEXT NOT NULL, sort_name TEXT NOT NULL,
            annot_json TEXT, annot_size INTEGER, annot_mtime REAL
        );
        CREATE TABLE sessions (
            path TEXT PRIMARY KEY, dataset_path TEXT NOT NULL, id TEXT NOT NULL, sort_name TEXT NOT NULL,
            annot_json TEXT, annot_size INTEGER, annot_mtime REAL
        );
        CREATE INDEX sessions_by_dataset ON sessions(dataset_path, sort_name, id);
        CREATE TABLE recordings (
            stem TEXT PRIMARY KEY, session_path TEXT NOT NULL, raw_path TEXT NOT NULL,
            sort_name TEXT NOT NULL, meta_json TEXT NOT NULL, annot_json TEXT,
            raw_size INTEGER, raw_mtime REAL, meta_size INTEGER, meta_mtime REAL,
            annot_size INTEGER, annot_mtime REAL, primary_stim_v REAL
        );
        CREATE INDEX recordings_by_session ON recordings(session_path, primary_stim_v, sort_name);
        """
    )
    connection.executemany(
        "INSERT INTO catalog_meta(key, value) VALUES (?, ?)",
        (("schema_version", str(CATALOG_SCHEMA_VERSION)), ("experiment_path", str(experiment_path))),
    )


def build_catalog(experiment_path: Path, progress_callback=None) -> ExperimentCatalog:
    """Build the complete catalog atomically from authoritative experiment files."""
    experiment_path = experiment_path.resolve()
    destination = catalog_path(experiment_path)
    temporary = destination.with_suffix(destination.suffix + ".building")
    if temporary.exists():
        temporary.unlink()

    dataset_paths = sorted((path for path in experiment_path.iterdir() if path.is_dir()), key=lambda path: (path.name.casefold(), path.name))
    completed = False
    connection = sqlite3.connect(temporary)
    try:
        with connection:
            _initialize(connection, experiment_path)
            for dataset_index, dataset_path in enumerate(dataset_paths, start=1):
                dataset_annot = dataset_path / "dataset.annot.json"
                dataset_size, dataset_mtime = _file_fingerprint(dataset_annot)
                connection.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        str(dataset_path),
                        dataset_path.name,
                        dataset_path.name.casefold(),
                        _read_text_if_exists(dataset_annot),
                        dataset_size,
                        dataset_mtime,
                    ),
                )
                session_paths = sorted((path for path in dataset_path.iterdir() if path.is_dir()), key=lambda path: (path.name.casefold(), path.name))
                for session_path in session_paths:
                    session_annot = session_path / "session.annot.json"
                    session_size, session_mtime = _file_fingerprint(session_annot)
                    connection.execute(
                        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            str(session_path),
                            str(dataset_path),
                            session_path.name,
                            session_path.name.casefold(),
                            _read_text_if_exists(session_annot),
                            session_size,
                            session_mtime,
                        ),
                    )
                    for raw_path in sorted(session_path.glob(f"*{RAW_SUFFIX}"), key=lambda path: (path.name.casefold(), path.name)):
                        stem = recording_stem(raw_path)
                        meta_path = stem.with_suffix(".meta.json")
                        annot_path = stem.with_suffix(".annot.json")
                        meta_text = _read_text_if_exists(meta_path)
                        if meta_text is None:
                            logger.warning("Skipping recording without metadata: %s", raw_path)
                            continue
                        raw_size, raw_mtime = _file_fingerprint(raw_path)
                        meta_size, meta_mtime = _file_fingerprint(meta_path)
                        annot_size, annot_mtime = _file_fingerprint(annot_path)
                        connection.execute(
                            "INSERT INTO recordings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                str(stem),
                                str(session_path),
                                str(raw_path),
                                stem.name.casefold(),
                                meta_text,
                                _read_text_if_exists(annot_path),
                                raw_size,
                                raw_mtime,
                                meta_size,
                                meta_mtime,
                                annot_size,
                                annot_mtime,
                                _primary_stim_voltage(meta_text),
                            ),
                        )
                if progress_callback is not None:
                    progress_callback("catalog", dataset_index, len(dataset_paths), dataset_path.name)
        completed = True
    finally:
        connection.close()
        if temporary.exists():
            if completed:
                os.replace(temporary, destination)
            else:
                temporary.unlink()

    # The JSON index is obsolete after a successful catalog build.
    legacy_index = experiment_path / ".index.json"
    if legacy_index.exists():
        legacy_index.unlink()
    return ExperimentCatalog(experiment_path)


def ensure_catalog(experiment_path: Path, progress_callback=None) -> ExperimentCatalog:
    """Return a usable catalog, rebuilding only when it is absent or invalid."""
    catalog = ExperimentCatalog(experiment_path.resolve())
    return catalog if catalog.is_usable() else build_catalog(catalog.experiment_path, progress_callback)


def refresh_recording_annotation(stem: Path) -> None:
    """Synchronize one saved recording annotation without rescanning its experiment."""
    stem = stem.resolve()
    experiment_path = stem.parent.parent.parent
    catalog = ExperimentCatalog(experiment_path)
    if not catalog.is_usable():
        return
    annotation_path = stem.with_suffix(".annot.json")
    annotation_size, annotation_mtime = _file_fingerprint(annotation_path)
    with catalog.connect() as connection:
        connection.execute(
            "UPDATE recordings SET annot_json = ?, annot_size = ?, annot_mtime = ? WHERE stem = ?",
            (_read_text_if_exists(annotation_path), annotation_size, annotation_mtime, str(stem)),
        )


def refresh_session_annotation(session_path: Path) -> None:
    """Synchronize one saved session annotation without rebuilding the catalog."""
    refresh_session_annotations([session_path])


def refresh_session_annotations(session_paths: list[Path]) -> None:
    """Synchronize saved session annotations in one catalog transaction."""
    if not session_paths:
        return
    resolved_paths = [session_path.resolve() for session_path in session_paths]
    catalog = ExperimentCatalog(resolved_paths[0].parent.parent)
    if not catalog.is_usable():
        return
    updates = []
    for session_path in resolved_paths:
        if session_path.parent.parent != catalog.experiment_path:
            raise ValueError("All sessions in a catalog refresh must belong to one experiment")
        annotation_path = session_path / "session.annot.json"
        annotation_size, annotation_mtime = _file_fingerprint(annotation_path)
        updates.append((_read_text_if_exists(annotation_path), annotation_size, annotation_mtime, str(session_path)))
    with catalog.connect() as connection:
        connection.executemany("UPDATE sessions SET annot_json = ?, annot_size = ?, annot_mtime = ? WHERE path = ?", updates)


def refresh_dataset_annotation(dataset_path: Path) -> None:
    """Synchronize one saved dataset annotation without rebuilding the catalog."""
    dataset_path = dataset_path.resolve()
    catalog = ExperimentCatalog(dataset_path.parent)
    if not catalog.is_usable():
        return
    annotation_path = dataset_path / "dataset.annot.json"
    annotation_size, annotation_mtime = _file_fingerprint(annotation_path)
    with catalog.connect() as connection:
        connection.execute(
            "UPDATE datasets SET annot_json = ?, annot_size = ?, annot_mtime = ? WHERE path = ?",
            (_read_text_if_exists(annotation_path), annotation_size, annotation_mtime, str(dataset_path)),
        )


def relocate_catalog_paths(experiment_path: Path, old_prefix: Path, new_prefix: Path) -> bool:
    """Update catalog paths after a filesystem rename without a full rebuild.

    ``old_prefix`` may no longer exist; it identifies the absolute paths stored
    before the rename. The operation is one SQLite transaction and scales with
    affected rows rather than requiring a full source traversal.
    """
    catalog = ExperimentCatalog(experiment_path)
    if not catalog.is_usable():
        return False
    old_text = str(old_prefix.resolve())
    new_text = str(new_prefix.resolve())
    if old_text == new_text:
        return True
    prefix_like = f"{old_text}%"
    with catalog.connect() as connection:
        for table, columns in (
            ("datasets", ("path",)),
            ("sessions", ("path", "dataset_path")),
            ("recordings", ("stem", "session_path", "raw_path")),
        ):
            for column in columns:
                connection.execute(
                    f"UPDATE {table} SET {column} = REPLACE({column}, ?, ?) WHERE {column} LIKE ?",
                    (old_text, new_text, prefix_like),
                )
        connection.execute("UPDATE catalog_meta SET value = ? WHERE key = 'experiment_path'", (str(catalog.experiment_path),))
    return True
