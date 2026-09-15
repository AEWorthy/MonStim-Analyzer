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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_FILENAME = ".monstim-cache.sqlite"
CATALOG_SCHEMA_VERSION = 2
RAW_SUFFIX = ".raw.h5"
DIRECTORY_SIGNATURE_KEY = "directory_signature"


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


def _directory_signature(experiment_path: Path) -> str:
    """Return a stable signature for the dataset/session directory hierarchy."""
    directories = []
    for dataset_path in experiment_path.iterdir():
        if not dataset_path.is_dir():
            continue
        directories.append(dataset_path.relative_to(experiment_path).as_posix())
        directories.extend(session_path.relative_to(experiment_path).as_posix() for session_path in dataset_path.iterdir() if session_path.is_dir())
    return json.dumps(sorted(directories), separators=(",", ":"))


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

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a catalog connection and release its file handle on exit."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

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

    def is_stale(self) -> bool:
        """Return whether source directories changed since this catalog was built."""
        if not self.is_usable():
            return True
        try:
            with self.connect() as connection:
                row = connection.execute(
                    "SELECT value FROM catalog_meta WHERE key = ?",
                    (DIRECTORY_SIGNATURE_KEY,),
                ).fetchone()
            return row is None or row["value"] != _directory_signature(self.experiment_path)
        except OSError, sqlite3.Error:
            logger.warning("Could not validate source directories for %s; rebuilding", self.path, exc_info=True)
            return True

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
        (
            ("schema_version", str(CATALOG_SCHEMA_VERSION)),
            ("experiment_path", str(experiment_path)),
            (DIRECTORY_SIGNATURE_KEY, _directory_signature(experiment_path)),
        ),
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
    """Return a current catalog, rebuilding when source directories changed."""
    catalog = ExperimentCatalog(experiment_path.resolve())
    if catalog.is_usable() and not catalog.is_stale():
        return catalog
    if catalog.is_usable():
        logger.info("Source directory structure changed for %s; rebuilding catalog", catalog.experiment_path)
    return build_catalog(catalog.experiment_path, progress_callback)


def invalidate_catalog(experiment_path: Path) -> None:
    """Remove the cached catalog so the next open rebuilds it from disk.

    The catalog is only a cache.  Removing it after a structural filesystem
    change is safer than trying to maintain every relationship in SQLite,
    especially for bulk operations and partial failures.
    """
    path = catalog_path(Path(experiment_path).resolve())
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"), path.with_suffix(path.suffix + ".building")):
        try:
            candidate.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning("Could not invalidate catalog sidecar %s", candidate, exc_info=True)


def invalidate_catalogs(*experiment_paths: Path) -> None:
    """Invalidate each distinct experiment catalog involved in a mutation."""
    seen: set[Path] = set()
    for experiment_path in experiment_paths:
        resolved = Path(experiment_path).resolve()
        if resolved not in seen:
            invalidate_catalog(resolved)
            seen.add(resolved)


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
    refresh_dataset_annotations([dataset_path])


def refresh_dataset_annotations(dataset_paths: list[Path]) -> None:
    """Synchronize saved dataset annotations in one catalog transaction."""
    if not dataset_paths:
        return
    resolved_paths = [dataset_path.resolve() for dataset_path in dataset_paths]
    catalog = ExperimentCatalog(resolved_paths[0].parent)
    if not catalog.is_usable():
        return
    updates = []
    for dataset_path in resolved_paths:
        if dataset_path.parent != catalog.experiment_path:
            raise ValueError("All datasets in a catalog refresh must belong to one experiment")
        annotation_path = dataset_path / "dataset.annot.json"
        annotation_size, annotation_mtime = _file_fingerprint(annotation_path)
        updates.append((_read_text_if_exists(annotation_path), annotation_size, annotation_mtime, str(dataset_path)))
    with catalog.connect() as connection:
        connection.executemany("UPDATE datasets SET annot_json = ?, annot_size = ?, annot_mtime = ? WHERE path = ?", updates)


def relocate_catalog_paths(experiment_path: Path, old_prefix: Path, new_prefix: Path) -> bool:
    """Update catalog paths after a filesystem rename without a full rebuild.

    ``old_prefix`` may no longer exist; it identifies the absolute paths stored
    before the rename. The operation is one SQLite transaction and scales with
    affected rows rather than requiring a full source traversal.
    """
    catalog = ExperimentCatalog(experiment_path)
    # A dataset/session rename leaves the catalog root unchanged, so the
    # normal usability check is sufficient.  An experiment rename moves the
    # catalog sidecar itself, however, and its stored root still names the old
    # experiment until this transaction updates it.  Accept precisely that
    # transitional state; do not treat an unrelated or corrupt cache as valid.
    if not catalog.path.is_file():
        return False
    old_text = str(old_prefix.resolve())
    new_text = str(new_prefix.resolve())
    if old_text == new_text:
        return True
    try:
        with catalog.connect() as connection:
            schema = connection.execute("SELECT value FROM catalog_meta WHERE key = 'schema_version'").fetchone()
            root = connection.execute("SELECT value FROM catalog_meta WHERE key = 'experiment_path'").fetchone()
    except sqlite3.Error:
        logger.warning("Catalog %s is unreadable and cannot be relocated", catalog.path, exc_info=True)
        return False
    if schema is None or schema["value"] != str(CATALOG_SCHEMA_VERSION) or root is None:
        return False
    stored_root = Path(root["value"])
    if stored_root != catalog.experiment_path and stored_root != Path(old_text):
        return False
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
        connection.execute(
            "INSERT INTO catalog_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (DIRECTORY_SIGNATURE_KEY, _directory_signature(catalog.experiment_path)),
        )
        connection.execute("UPDATE catalog_meta SET value = ? WHERE key = 'experiment_path'", (str(catalog.experiment_path),))
    return True


def transfer_catalog_dataset(
    source_experiment: Path,
    destination_experiment: Path,
    source_dataset: Path,
    destination_dataset: Path,
    *,
    remove_source: bool = True,
) -> bool:
    """Transfer or copy one dataset's cached rows between experiment catalogs.

    The dataset has already moved on disk when this function is called.  Both
    catalogs retain the pre-move row data, so this copies the affected rows to
    the destination with their new absolute paths and then removes them from
    the source.  It does not inspect recordings or open HDF5 files.

    A cache is non-authoritative: callers must invalidate both catalogs if
    this returns ``False`` or raises after either catalog was modified.
    """
    source_catalog = ExperimentCatalog(source_experiment)
    destination_catalog = ExperimentCatalog(destination_experiment)
    if not source_catalog.is_usable() or not destination_catalog.is_usable():
        return False

    source_text = str(source_dataset.resolve())
    destination_text = str(destination_dataset.resolve())
    source_like = f"{source_text}%"

    try:
        with source_catalog.connect() as connection:
            dataset_rows = connection.execute("SELECT * FROM datasets WHERE path = ?", (source_text,)).fetchall()
            session_rows = connection.execute(
                "SELECT * FROM sessions WHERE dataset_path = ? OR path LIKE ?",
                (source_text, source_like),
            ).fetchall()
            recording_rows = connection.execute(
                "SELECT * FROM recordings WHERE session_path LIKE ? OR raw_path LIKE ?",
                (source_like, source_like),
            ).fetchall()
        if len(dataset_rows) != 1:
            return False

        def relocated(row: sqlite3.Row, columns: tuple[str, ...]) -> tuple:
            return tuple(
                value.replace(source_text, destination_text, 1) if column in columns and isinstance(value, str) else value
                for column, value in zip(row.keys(), row, strict=True)
            )

        with destination_catalog.connect() as connection:
            # A stale destination cache can retain a row for a dataset that
            # no longer exists on disk. The pre-move filesystem conflict check
            # guarantees this prefix is available for the transferred dataset.
            destination_like = f"{destination_text}%"
            connection.execute(
                "DELETE FROM recordings WHERE session_path LIKE ? OR raw_path LIKE ?",
                (destination_like, destination_like),
            )
            connection.execute("DELETE FROM sessions WHERE dataset_path = ? OR path LIKE ?", (destination_text, f"{destination_text}%"))
            connection.execute("DELETE FROM datasets WHERE path = ?", (destination_text,))
            connection.executemany("INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)", [relocated(row, ("path",)) for row in dataset_rows])
            connection.executemany(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [relocated(row, ("path", "dataset_path")) for row in session_rows],
            )
            connection.executemany(
                "INSERT INTO recordings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [relocated(row, ("stem", "session_path", "raw_path")) for row in recording_rows],
            )
            connection.execute(
                "INSERT INTO catalog_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (DIRECTORY_SIGNATURE_KEY, _directory_signature(destination_catalog.experiment_path)),
            )

        if remove_source:
            with source_catalog.connect() as connection:
                connection.execute("DELETE FROM recordings WHERE session_path LIKE ? OR raw_path LIKE ?", (source_like, source_like))
                connection.execute("DELETE FROM sessions WHERE dataset_path = ? OR path LIKE ?", (source_text, source_like))
                connection.execute("DELETE FROM datasets WHERE path = ?", (source_text,))
                connection.execute(
                    "INSERT INTO catalog_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (DIRECTORY_SIGNATURE_KEY, _directory_signature(source_catalog.experiment_path)),
                )
        return True
    except OSError, sqlite3.Error:
        logger.warning("Could not transfer catalog rows for %s", source_dataset, exc_info=True)
        return False


def copy_catalog_dataset(
    source_experiment: Path,
    destination_experiment: Path,
    source_dataset: Path,
    destination_dataset: Path,
) -> bool:
    """Add a copied dataset to an existing destination catalog without a rebuild.

    The copy's recording metadata is identical to its source.  Transfer the
    already-cached rows, then replace the copied dataset annotation because a
    same-experiment duplicate can adjust its display metadata on disk.
    """
    if not transfer_catalog_dataset(source_experiment, destination_experiment, source_dataset, destination_dataset, remove_source=False):
        return False

    destination_dataset = destination_dataset.resolve()
    catalog = ExperimentCatalog(destination_experiment)
    try:
        annotation_path = destination_dataset / "dataset.annot.json"
        annotation_size, annotation_mtime = _file_fingerprint(annotation_path)
        with catalog.connect() as connection:
            connection.execute(
                "UPDATE datasets SET id = ?, sort_name = ?, annot_json = ?, annot_size = ?, annot_mtime = ? WHERE path = ?",
                (
                    destination_dataset.name,
                    destination_dataset.name.casefold(),
                    _read_text_if_exists(annotation_path),
                    annotation_size,
                    annotation_mtime,
                    str(destination_dataset),
                ),
            )
        return True
    except OSError, sqlite3.Error:
        logger.warning("Could not update copied catalog row for %s", destination_dataset, exc_info=True)
        return False


def remove_catalog_dataset(experiment_path: Path, dataset_path: Path) -> bool:
    """Remove one deleted dataset's rows from a usable catalog transactionally."""
    experiment_path = experiment_path.resolve()
    dataset_path = dataset_path.resolve()
    catalog = ExperimentCatalog(experiment_path)
    if not catalog.is_usable():
        return False
    if dataset_path.parent != experiment_path:
        raise ValueError(f"Dataset {dataset_path} does not belong to experiment {experiment_path}")

    dataset_text = str(dataset_path)
    dataset_like = f"{dataset_text}%"
    try:
        with catalog.connect() as connection:
            connection.execute("DELETE FROM recordings WHERE session_path LIKE ? OR raw_path LIKE ?", (dataset_like, dataset_like))
            connection.execute("DELETE FROM sessions WHERE dataset_path = ? OR path LIKE ?", (dataset_text, dataset_like))
            connection.execute("DELETE FROM datasets WHERE path = ?", (dataset_text,))
            connection.execute(
                "INSERT INTO catalog_meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (DIRECTORY_SIGNATURE_KEY, _directory_signature(experiment_path)),
            )
        return True
    except OSError, sqlite3.Error:
        logger.warning("Could not remove catalog rows for %s", dataset_path, exc_info=True)
        return False
