# monstim_signals/io/repositories.py
import concurrent.futures
import datetime
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import h5py

from monstim_signals.core import (
    DatasetAnnot,
    ExperimentAnnot,
    RecordingAnnot,
    RecordingMeta,
    SessionAnnot,
)
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.recording import Recording
from monstim_signals.domain.session import Session
from monstim_signals.io.data_migrations import (
    FutureVersionError,
    UnknownVersionError,
    migrate_annotation_dict,
    scan_annotation_versions,
)
from monstim_signals.io.experiment_index import find_session_index


class RecordingRepository:
    """
    Knows how to load/save exactly one recording:
    <stem>.raw.h5, <stem>.meta.json, <stem>.annot.json
    where `stem` is a Path without extension, e.g. Path(".../AA00_0000").
    """

    def __init__(self, stem: Path):
        """
        `stem` = Path to the filename prefix, without suffix.
        e.g., if the files are:
            AA00_0000.raw.h5
            AA00_0000.meta.json
            AA00_0000.annot.json
        then stem = Path("/path/to/AA00_0000").
        """
        self.stem = stem
        self.raw_h5 = stem.with_suffix(".raw.h5")
        self.meta_js = stem.with_suffix(".meta.json")
        self.annot_js = stem.with_suffix(".annot.json")

    def update_path(self, new_stem: Path) -> None:
        """
        Update the repository to point to a new stem.
        This is useful if the recording files move.
        """
        self.stem = new_stem
        self.raw_h5 = new_stem.with_suffix(".raw.h5")
        self.meta_js = new_stem.with_suffix(".meta.json")
        self.annot_js = new_stem.with_suffix(".annot.json")

    def load(
        self, config=None, *, strict_version: bool = False, lazy_open_h5: Optional[bool] = None, allow_write: bool = True
    ) -> "Recording":
        # 1) Load meta JSON (immutable, record‐time facts)
        meta_dict = json.loads(self.meta_js.read_text())
        meta = RecordingMeta.from_dict(meta_dict)

        # 2) Load or create annot JSON (user edits)
        if self.annot_js.exists():
            try:
                text = self.annot_js.read_text()
                if not text.strip():
                    logging.warning(f"Annotation file '{self.annot_js}' is empty. Recreating empty annotation.")
                    annot_dict = asdict(RecordingAnnot.create_empty())
                    if allow_write:
                        self.annot_js.write_text(json.dumps(annot_dict, indent=2))
                else:
                    try:
                        annot_dict = json.loads(text)
                    except json.JSONDecodeError as e:
                        size = None
                        try:
                            size = self.annot_js.stat().st_size
                        except Exception:
                            logging.exception(f"Failed to get size of annotation file '{self.annot_js}'")
                            pass
                        logging.error(f"Failed to decode JSON in annotation file '{self.annot_js}' (size={size}): {e}")
                        # Move corrupt file aside and create a fresh annotation file
                        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                        corrupt_path = self.annot_js.with_name(f"{self.annot_js.name}.corrupt-{ts}")
                        try:
                            if allow_write:
                                self.annot_js.rename(corrupt_path)
                                logging.warning(f"Moved corrupt annotation to {corrupt_path}. Creating new empty annotation.")
                        except Exception:
                            logging.exception(f"Failed to move corrupt annotation file {self.annot_js}")
                        annot_dict = asdict(RecordingAnnot.create_empty())
                        if allow_write:
                            self.annot_js.write_text(json.dumps(annot_dict, indent=2))

                try:
                    report = migrate_annotation_dict(annot_dict, strict_version=strict_version)
                    if report.changed and allow_write:
                        logging.debug(
                            f"Recording annotation migrated {report.original_version}->{report.final_version} for {self.annot_js.name}"
                        )
                        # Persist migrated version immediately
                        self.annot_js.write_text(json.dumps(annot_dict, indent=2))
                except FutureVersionError as e:
                    logging.error(str(e))
                    raise
                except UnknownVersionError as e:
                    logging.warning(f"Unknown version for {self.annot_js}: {e}. Proceeding without migration.")
                annot = RecordingAnnot.from_dict(annot_dict)
            except Exception:
                logging.exception(f"Unexpected error while reading annotation file {self.annot_js}")
                raise
        else:
            logging.warning(f"Annotation file '{self.annot_js}' not found. Using a new empty annotation in-memory.")
            annot = RecordingAnnot.create_empty()
            if allow_write:
                self.annot_js.write_text(json.dumps(asdict(annot), indent=2))

        # 3) Optionally avoid opening the HDF5 file here to speed up initial loads.
        #    If the config specifies `lazy_open_h5=True` we will not open the
        #    file here and instead rely on the `meta.num_samples` field if it is
        #    present. This drastically reduces per-recording overhead when many
        #    recordings are present and raw data will only be accessed later.
        # Determine lazy_open behavior: prefer explicit argument; otherwise default
        # to the original eager behavior (do not lazy-open) to preserve backwards compatibility.
        lazy_open = bool(lazy_open_h5) if lazy_open_h5 is not None else False

        raw_dataset = None
        if not lazy_open:
            # default behavior: open dataset now (maintains existing behavior)
            h5file = h5py.File(self.raw_h5, "r")
            raw_dataset = h5file["raw"]  # type: ignore
            # Patch in num_samples from the raw array shape
            meta.num_samples = int(raw_dataset.shape[0])  # (#samples × #channels)
        else:
            # If we choose laziness and the meta contains num_samples, rely on it.
            # Otherwise open the file temporarily to extract num_samples.
            if meta.num_samples is None:
                # Rare case: read the HDF5 just to set num_samples then close
                tmp_h5 = h5py.File(self.raw_h5, "r")
                try:
                    raw_ds_tmp = tmp_h5["raw"]
                    meta.num_samples = int(raw_ds_tmp.shape[0])
                finally:
                    tmp_h5.close()

        # 5) Build the domain object. If `raw_dataset` is None we pass a placeholder
        #    which will be reopened lazily later when `rec.raw_view(...)` is called.
        recording = Recording(meta=meta, annot=annot, raw=raw_dataset, repo=self, config=config)
        return recording

    def save(self, recording: Recording) -> None:
        """
        Only rewrite the annot JSON (we assume meta/raw never change).
        This is called when the user edits the recording's annotation.
        """
        try:
            recording.annot.date_modified = datetime.datetime.now().isoformat(timespec="seconds")
        except Exception:
            logging.debug("Failed to set date_modified on RecordingAnnot", exc_info=True)
        self.annot_js.write_text(json.dumps(asdict(recording.annot), indent=2))

    def rename(self, new_stem: Path, attempts: int = 3, wait: float = 0.5) -> None:
        """Rename recording files to a new stem atomically with retries on Windows locks.

        Args:
            new_stem: Path to new stem (no suffix).
        """
        import errno
        import gc
        import time

        if new_stem.exists():
            raise FileExistsError(f"Target recording stem already exists: {new_stem}")

        for attempt in range(attempts):
            try:
                # Rename file-by-file if they exist
                if self.raw_h5.exists():
                    self.raw_h5.rename(new_stem.with_suffix(".raw.h5"))
                if self.meta_js.exists():
                    self.meta_js.rename(new_stem.with_suffix(".meta.json"))
                if self.annot_js.exists():
                    self.annot_js.rename(new_stem.with_suffix(".annot.json"))
                # Update in-memory paths
                self.update_path(new_stem)
                # Refresh experiment index after recording rename
                try:
                    exp_root = new_stem.parent.parent.parent  # experiment/dataset/session/recording-stem
                    from .experiment_index import ensure_fresh_index

                    ensure_fresh_index(exp_root.name, exp_root)
                except Exception:
                    logging.debug("Index refresh after recording rename failed (non-fatal).", exc_info=True)
                return
            except OSError as e:
                if getattr(e, "errno", None) == errno.EACCES and attempt < attempts - 1:
                    logging.warning(f"Access denied renaming recording (attempt {attempt+1}), retrying...")
                    time.sleep(wait)
                    gc.collect()
                    continue
                raise

    @staticmethod
    def discover_in_folder(folder: Path) -> Iterator["RecordingRepository"]:
        """
        Given a folder Path, yield a RecordingRepository for each *.raw.h5 found.
        E.g. if folder contains:
            AA00_0000.raw.h5, AA00_0000.meta.json, AA00_0000.annot.json,
            AA00_0001.raw.h5, AA00_0001.meta.json, AA00_0001.annot.json
        then this yields:
            RecordingRepository(Path("folder/AA00_0000"))
            RecordingRepository(Path("folder/AA00_0001"))
        """
        for raw_h5 in folder.glob("*.raw.h5"):
            stem = raw_h5.with_suffix("")  # drop ".raw.h5" → Path("folder/AA00_0000")
            yield RecordingRepository(stem=stem)


class SessionRepository:
    """
    Knows how to load/save one session (i.e. a folder of recordings at various stimuli).
    A session folder must contain multiple <stem>.raw.h5/.meta.json/.annot.json.
    """

    def __init__(self, folder: Path):
        """
        `folder` is a Path to a session‐level directory, e.g. Path("/data/ExperimentRoot/Dataset_01/AA00").
        """
        self.folder = folder
        self.session_id = folder.name  # e.g. "AA00"
        self.session_js = folder / "session.annot.json"

    def update_path(self, new_folder: Path) -> None:
        """
        Update the repository to point to a new folder.
        This is useful if the session root folder changes.
        """
        self.folder = new_folder
        self.session_js = new_folder / "session.annot.json"
        # Ensure the session_id reflects the new folder name
        self.session_id = new_folder.name

    def load(
        self, config=None, *, strict_version: bool = False, allow_write: bool = True, load_recordings: bool = True
    ) -> "Session":
        # Guard: folder must exist
        if not self.folder.exists():
            raise FileNotFoundError(f"Session folder not found: {self.folder}")

        recordings = []
        # 1) Discover all recordings in this folder
        # Prefer index-based discovery to avoid heavy directory scans
        recording_repos: list[RecordingRepository] = []
        try:
            exp_root = self.folder.parent.parent  # Experiment/dataset/session
            sess_idx = find_session_index(exp_root, self.folder)
            if sess_idx is not None and sess_idx.recordings:
                for r in sess_idx.recordings:
                    p = Path(r.path)
                    stem = p.with_suffix("") if p.is_file() else p  # tolerate file or folder style
                    recording_repos.append(RecordingRepository(stem=stem))
            else:
                recording_repos = list(RecordingRepository.discover_in_folder(self.folder))
        except Exception:
            logging.debug("Index-based discovery failed; falling back to folder scan.", exc_info=True)
            recording_repos = list(RecordingRepository.discover_in_folder(self.folder))
        if load_recordings:
            # Pass through lazy_open_h5 from config if present; prefer explicit key in config
            lazy_from_cfg = None
            try:
                if config is not None and "lazy_open_h5" in config:
                    lazy_from_cfg = bool(config.get("lazy_open_h5"))
            except Exception:
                lazy_from_cfg = None

            recordings = [
                repo.load(config=config, lazy_open_h5=lazy_from_cfg, allow_write=allow_write) for repo in recording_repos
            ]
            # 2) Sort by the primary StimCluster’s stim_v
            recordings.sort(key=lambda r: r.meta.primary_stim.stim_v)
        else:
            # Lightweight discovery: build minimal Recording objects without opening HDF5,
            # so sessions are considered valid and pass strict domain checks.
            # Heavy raw data access remains deferred.
            try:
                self._pending_recording_stems = [r.stem for r in recording_repos]
            except Exception:
                logging.debug("Failed to record pending recording stems (non-fatal).", exc_info=True)

            try:
                # Force lazy_open_h5 to True for lightweight construction
                recordings = [repo.load(config=config, lazy_open_h5=True, allow_write=allow_write) for repo in recording_repos]
                recordings.sort(key=lambda r: r.meta.primary_stim.stim_v)
            except Exception:
                logging.debug("Lightweight recording construction failed; leaving session empty.", exc_info=True)

        # 3) Load or create session annotation JSON
        if self.session_js.exists():
            try:
                text = self.session_js.read_text()
                if not text.strip():
                    logging.warning(
                        f"Session annotation file '{self.session_js}' is empty. Creating empty session annotation."
                    )
                    session_annot_dict = asdict(SessionAnnot.create_empty())
                    if allow_write:
                        self.session_js.write_text(json.dumps(session_annot_dict, indent=2))
                else:
                    try:
                        session_annot_dict = json.loads(text)
                    except json.JSONDecodeError as e:
                        size = None
                        try:
                            size = self.session_js.stat().st_size
                        except Exception:
                            logging.exception(f"Failed to get size of session annotation file '{self.session_js}'")
                            pass
                        logging.error(f"Failed to decode JSON in session annotation '{self.session_js}' (size={size}): {e}")
                        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                        corrupt_path = self.session_js.with_name(f"{self.session_js.name}.corrupt-{ts}")
                        try:
                            if allow_write:
                                self.session_js.rename(corrupt_path)
                                logging.warning(
                                    f"Moved corrupt session annotation to {corrupt_path}. Creating new empty annotation."
                                )
                        except Exception:
                            logging.exception(f"Failed to move corrupt session annotation file {self.session_js}")
                        session_annot_dict = asdict(SessionAnnot.create_empty())
                        if allow_write:
                            self.session_js.write_text(json.dumps(session_annot_dict, indent=2))

                try:
                    report = migrate_annotation_dict(session_annot_dict, strict_version=strict_version)
                    if report.changed:
                        logging.debug(
                            f"Session annotation migrated {report.original_version}->{report.final_version} for {self.session_js.name}"
                        )
                        # Persist migrations regardless of allow_write; schema updates must be saved
                        self.session_js.write_text(json.dumps(session_annot_dict, indent=2))
                except FutureVersionError as e:
                    logging.error(str(e))
                    raise
                except UnknownVersionError as e:
                    logging.warning(f"Unknown version for {self.session_js}: {e}. Proceeding without migration.")
                session_annot = SessionAnnot.from_dict(session_annot_dict)
            except Exception:
                logging.exception(f"Unexpected error while reading session annotation {self.session_js}")
                raise
        else:  # If no session.annot.json, initialize a brand‐new one
            if recordings:
                logging.debug(
                    f"Session annotation file '{self.session_js}' not found. Using first recording's meta to create a new one."
                )
                session_annot = SessionAnnot.from_meta(recordings[0].meta)
            else:
                logging.warning(f"Session annotation file '{self.session_js}' not found. Creating a new empty one.")
                session_annot = SessionAnnot.create_empty()
            if allow_write:
                self.session_js.write_text(json.dumps(asdict(session_annot), indent=2))

        # 4) Build a Session domain object
        session = Session(
            session_id=self.session_id,
            recordings=recordings,
            annot=session_annot,
            repo=self,
            config=config,
        )
        return session

    def materialize_recordings(self, session: "Session", config=None, *, allow_write: bool = False) -> "Session":
        """Load recordings into an existing session on-demand.

        Uses any cached pending stems discovered during load; falls back to folder scan.
        """
        try:
            stems = getattr(self, "_pending_recording_stems", None)
            if stems is None:
                stems = [r.stem for r in RecordingRepository.discover_in_folder(self.folder)]

            # Honor lazy_open_h5 from config to avoid opening data at selection time if desired
            lazy_from_cfg = None
            try:
                if config is not None and "lazy_open_h5" in config:
                    lazy_from_cfg = bool(config.get("lazy_open_h5"))
            except Exception:
                lazy_from_cfg = None

            recs = [
                RecordingRepository(stem).load(config=config, lazy_open_h5=lazy_from_cfg, allow_write=allow_write)
                for stem in stems
            ]
            try:
                recs.sort(key=lambda r: r.meta.primary_stim.stim_v)
            except Exception:
                # If any recording is missing primary_stim, keep original order
                logging.debug("Could not sort recordings by stim_v; leaving original order.", exc_info=True)

            # Session.recordings is a derived, filtered view (no setter). Update internal storage.
            session._all_recordings = recs
            # Refresh session parameters and caches to reflect newly available recordings
            try:
                session._load_session_parameters()
                session._initialize_annotations()
                session.update_latency_window_parameters()
                session.reset_recordings_cache()
            except Exception:
                logging.debug("Post-materialization session refresh encountered an issue.", exc_info=True)
        except Exception:
            logging.exception("Failed to materialize recordings for session %s", self.session_id)
        return session

    def save(self, session: Session) -> None:
        try:
            session.annot.date_modified = datetime.datetime.now().isoformat(timespec="seconds")
        except Exception:
            logging.debug("Failed to set date_modified on SessionAnnot", exc_info=True)
        self.session_js.write_text(json.dumps(asdict(session.annot), indent=2))
        # Save ALL recordings including excluded ones to persist their state
        for rec in session._all_recordings:
            rec.repo.save(rec)

    def rename(self, new_folder: Path, attempts: int = 3, wait: float = 0.5) -> None:
        """Rename the session folder, retrying on transient Windows locks."""
        import errno
        import gc
        import time

        if new_folder.exists():
            raise FileExistsError(f"Target session folder already exists: {new_folder}")

        for attempt in range(attempts):
            try:
                self.folder.rename(new_folder)
                self.update_path(new_folder)
                # Refresh experiment index after session rename
                try:
                    exp_root = new_folder.parent.parent
                    from .experiment_index import ensure_fresh_index

                    ensure_fresh_index(exp_root.name, exp_root)
                except Exception:
                    logging.debug("Index refresh after session rename failed (non-fatal).", exc_info=True)
                return
            except OSError as e:
                if getattr(e, "errno", None) == errno.EACCES and attempt < attempts - 1:
                    logging.warning(f"Access denied renaming session (attempt {attempt+1}), retrying...")
                    time.sleep(wait)
                    gc.collect()
                    continue
                raise

    @staticmethod
    def discover_in_folder(folder: Path) -> Iterator["SessionRepository"]:
        """
        Given a folder Path, yield a SessionRepository for each session subfolder.
        E.g. if folder contains:
            AA00, AA01, AA02
        then this yields:
            SessionRepository(Path("folder/AA00"))
            SessionRepository(Path("folder/AA01"))
            SessionRepository(Path("folder/AA02"))
        """
        for entry in folder.iterdir():
            # Only consider directories as session candidates; skip files like dataset.annot.json
            if not entry.is_dir():
                continue

            # Session folder with explicit session annotation
            if (entry / "session.annot.json").exists():
                logging.debug(f"Discovered session: {entry.name}")
                yield SessionRepository(entry)
                continue

            # Session folder inferred by presence of raw recordings
            if any(entry.glob("*.raw.h5")):
                logging.debug(f"Discovered session without annot: {entry.name}")
                yield SessionRepository(entry)  # still yield, but no session.annot.json
                continue

            # Directory present but no recognizable session contents; warn so users can fix layout
            logging.warning(f"Invalid session directory (no session.annot.json nor *.raw.h5): {entry}")


class DatasetRepository:
    """
    Knows how to load/save one dataset (all sessions from one animal).
    A dataset folder contains multiple subfolders, each a session.
    """

    def __init__(self, folder: Path):
        """
        `folder` might be Path("/data/ExperimentRoot/Dataset_01").
        """
        self.folder = folder
        self.dataset_id = folder.name  # e.g. "Dataset_01" or "240829 C328.1 post-dec mcurve_long-"
        self.dataset_js = folder / "dataset.annot.json"

    def update_path(self, new_folder: Path) -> None:
        """
        Update the repository to point to a new folder.
        This is useful if the dataset root folder changes.
        """
        self.folder = new_folder
        self.dataset_js = new_folder / "dataset.annot.json"

    def load(
        self,
        config=None,
        *,
        strict_version: bool = False,
        lazy_open_h5: Optional[bool] = None,
        allow_write: bool = True,
        load_recordings: bool = False,
    ) -> "Dataset":
        # 1) Discover valid session folders (those with annot or any *.raw.h5)
        session_repos = list(SessionRepository.discover_in_folder(self.folder))

        # 2) Load each Session from discovered repos
        sessions = []
        for repo in session_repos:
            try:
                sess = repo.load(
                    config=config,
                    strict_version=strict_version,
                    allow_write=allow_write,
                    load_recordings=load_recordings,
                )
                sessions.append(sess)
            except ValueError as e:
                # Surface a clear message and continue loading other sessions
                logging.error(
                    "Skipping session '%s' due to missing recordings or invalid contents: %s",
                    repo.session_id,
                    e,
                )
                logging.error("Session folder incomplete: %s", repo.folder)
                continue
            except Exception:
                logging.exception("Failed to load session '%s' at %s", repo.session_id, repo.folder)
                continue

        # 3) Load or create dataset annotation JSON
        if self.dataset_js.exists():
            try:
                text = self.dataset_js.read_text()
                if not text.strip():
                    logging.warning(
                        f"Dataset annotation file '{self.dataset_js}' is empty. Creating empty dataset annotation."
                    )
                    dataset_annot_dict = asdict(DatasetAnnot.from_ds_name(self.dataset_id))
                    if allow_write:
                        self.dataset_js.write_text(json.dumps(dataset_annot_dict, indent=2))
                else:
                    try:
                        dataset_annot_dict = json.loads(text)
                    except json.JSONDecodeError as e:
                        size = None
                        try:
                            size = self.dataset_js.stat().st_size
                        except Exception:
                            logging.exception(f"Failed to get size of dataset annotation file '{self.dataset_js}'")
                            pass
                        logging.error(f"Failed to decode JSON in dataset annotation '{self.dataset_js}' (size={size}): {e}")
                        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                        corrupt_path = self.dataset_js.with_name(f"{self.dataset_js.name}.corrupt-{ts}")
                        try:
                            if allow_write:
                                self.dataset_js.rename(corrupt_path)
                                logging.warning(
                                    f"Moved corrupt dataset annotation to {corrupt_path}. Creating new empty annotation."
                                )
                        except Exception:
                            logging.exception(f"Failed to move corrupt dataset annotation file {self.dataset_js}")
                        dataset_annot_dict = asdict(DatasetAnnot.from_ds_name(self.dataset_id))
                        if allow_write:
                            self.dataset_js.write_text(json.dumps(dataset_annot_dict, indent=2))

                try:
                    report = migrate_annotation_dict(dataset_annot_dict, strict_version=strict_version)
                    if report.changed:
                        logging.debug(
                            f"Dataset annotation migrated {report.original_version}->{report.final_version} for {self.dataset_js.name}"
                        )
                        # Persist migrations regardless of allow_write; schema updates must be saved
                        self.dataset_js.write_text(json.dumps(dataset_annot_dict, indent=2))
                except FutureVersionError as e:
                    logging.error(str(e))
                    raise
                except UnknownVersionError as e:
                    logging.warning(f"Unknown version for {self.dataset_js}: {e}. Proceeding without migration.")
                dataset_annot = DatasetAnnot.from_dict(dataset_annot_dict)
            except Exception:
                logging.exception(f"Unexpected error while reading dataset annotation {self.dataset_js}")
                raise
        else:  # If no session.annot.json, initialize a brand‐new one
            logging.info(
                f"Session annotation file '{self.dataset_js}' not found. Using the dataset name to create a new one (in-memory)."
            )
            dataset_annot = DatasetAnnot.from_ds_name(self.dataset_id)
            if allow_write:
                self.dataset_js.write_text(json.dumps(asdict(dataset_annot), indent=2))

        # 4) Build a Dataset domain object
        dataset = Dataset(
            dataset_id=self.dataset_id,
            sessions=sessions,
            annot=dataset_annot,
            repo=self,
            config=config,
        )
        return dataset

    def save(self, dataset: Dataset) -> None:
        """
        Save all sessions in this dataset.
        (If I want dataset‐level annotations in the future, write them here.)
        This is called when the user edits any session's recordings.
        """
        try:
            dataset.annot.date_modified = datetime.datetime.now().isoformat(timespec="seconds")
        except Exception:
            logging.debug("Failed to set date_modified on DatasetAnnot", exc_info=True)
        self.dataset_js.write_text(json.dumps(asdict(dataset.annot), indent=2))
        # Save ALL sessions including excluded ones to persist their state
        for session in dataset._all_sessions:
            session.repo.save(session)

    def rename(self, new_folder: Path, dataset=None, attempts: int = 3, wait: float = 0.5) -> None:
        """Rename the dataset folder, retrying on transient Windows locks.

        If `dataset` (a Dataset domain object) is provided, update its child session and
        recording repository objects in-memory so callers do not need to update them.
        """
        import errno
        import gc
        import time

        if new_folder.exists():
            raise FileExistsError(f"Target dataset folder already exists: {new_folder}")

        for attempt in range(attempts):
            try:
                self.folder.rename(new_folder)
                # Update repo internals
                self.update_path(new_folder)
                # Update dataset id
                self.dataset_id = new_folder.name

                # If a Dataset domain object was provided, update its child repos
                if dataset is not None:
                    try:
                        dataset.id = new_folder.name
                        # For each session in the dataset, update the session repo paths
                        for session in dataset.get_all_sessions(include_excluded=True):
                            if session.repo:
                                # New session folder path under the renamed dataset
                                new_session_path = self.folder / session.repo.folder.name
                                session.repo.update_path(new_session_path)

                                # Update recordings within the session
                                for recording in session.recordings:
                                    if recording.repo:
                                        # recording.repo.stem is a Path to the stem (filename without suffix)
                                        old_stem = recording.repo.stem
                                        new_stem = new_session_path / old_stem.name
                                        recording.repo.update_path(new_stem)
                    except Exception:
                        # If in-memory updates fail, log but do not prevent the rename (filesystem succeeded)
                        logging.exception("Failed to update in-memory child repo objects after dataset rename.")

                return
            except OSError as e:
                if getattr(e, "errno", None) == errno.EACCES and attempt < attempts - 1:
                    logging.warning(f"Access denied renaming dataset (attempt {attempt+1}), retrying...")
                    time.sleep(wait)
                    gc.collect()
                    continue
                raise
        # Refresh experiment index after dataset rename
        try:
            exp_root = new_folder.parent
            from .experiment_index import ensure_fresh_index

            ensure_fresh_index(exp_root.name, exp_root)
        except Exception:
            logging.debug("Index refresh after dataset rename failed (non-fatal).", exc_info=True)

    def get_metadata(self) -> dict:
        """
        Get lightweight metadata about the dataset without loading heavy session/recording data.
        Returns basic info like session count, names, completion status, etc.
        """
        try:
            # Get dataset annotation
            if self.dataset_js.exists():
                try:
                    text = self.dataset_js.read_text()
                    if not text.strip():
                        logging.warning(f"Dataset annotation file '{self.dataset_js}' is empty. Using defaults.")
                        dataset_annot = DatasetAnnot.from_ds_name(self.dataset_id)
                    else:
                        try:
                            annot_dict = json.loads(text)
                            dataset_annot = DatasetAnnot.from_dict(annot_dict)
                        except json.JSONDecodeError as e:
                            size = None
                            try:
                                size = self.dataset_js.stat().st_size
                            except Exception:
                                logging.exception(f"Failed to get size of dataset annotation file '{self.dataset_js}'")
                                pass
                            logging.error(
                                f"Failed to decode JSON in dataset annotation '{self.dataset_js}' (size={size}): {e}"
                            )
                            ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                            corrupt_path = self.dataset_js.with_name(f"{self.dataset_js.name}.corrupt-{ts}")
                            try:
                                self.dataset_js.rename(corrupt_path)
                                logging.warning(f"Moved corrupt dataset annotation to {corrupt_path}. Using defaults.")
                            except Exception:
                                logging.exception(f"Failed to move corrupt dataset annotation file {self.dataset_js}")
                            dataset_annot = DatasetAnnot.from_ds_name(self.dataset_id)
                except Exception:
                    logging.exception(f"Unexpected error while reading dataset annotation {self.dataset_js}")
                    dataset_annot = DatasetAnnot.from_ds_name(self.dataset_id)
            else:
                dataset_annot = DatasetAnnot.from_ds_name(self.dataset_id)

            # Get session folders without loading them
            session_folders = [p for p in self.folder.iterdir() if p.is_dir()]
            session_names = [folder.name for folder in session_folders]

            # Construct formatted name like the Dataset domain object does
            if dataset_annot.date and dataset_annot.animal_id and dataset_annot.condition:
                formatted_name = f"{dataset_annot.date} {dataset_annot.animal_id} {dataset_annot.condition}"
            else:
                formatted_name = self.dataset_id

            return {
                "id": self.dataset_id,
                "path": str(self.folder),
                "formatted_name": formatted_name,
                "session_count": len(session_folders),
                "session_names": session_names,
                "is_completed": dataset_annot.is_completed,
                "excluded_sessions": dataset_annot.excluded_sessions,
                "data_version": dataset_annot.data_version,
                "animal_id": dataset_annot.animal_id,
                "date": dataset_annot.date,
                "condition": dataset_annot.condition,
                "date_added": dataset_annot.date_added,
                "date_modified": dataset_annot.date_modified,
            }

        except Exception as e:
            logging.error(f"Failed to get metadata for dataset {self.dataset_id}: {e}")
            return {
                "id": self.dataset_id,
                "path": str(self.folder),
                "formatted_name": self.dataset_id,
                "session_count": 0,
                "session_names": [],
                "is_completed": False,
                "excluded_sessions": [],
                "data_version": "unknown",
                "animal_id": "unknown",
                "date": "unknown",
                "condition": "unknown",
                "error": str(e),
            }


class ExperimentRepository:
    """
    Knows how to load/save an entire experiment (all animals).
    The root folder contains multiple subfolders, each a dataset (animal).
    """

    def __init__(self, folder: Path):
        """
        `folder` might be Path("/data/ExperimentRoot").
        """
        self.folder = folder
        self.expt_js = folder / "experiment.annot.json"

    def update_path(self, new_folder: Path) -> None:
        """
        Update the repository to point to a new folder.
        This is useful if the experiment root folder changes.
        """
        self.folder = new_folder
        self.expt_js = new_folder / "experiment.annot.json"

    def load(
        self,
        config=None,
        *,
        strict_version: bool = False,
        preflight_scan: bool = False,
        progress_callback=None,
        lazy_open_h5: Optional[bool] = None,
        load_workers: Optional[int] = None,
        allow_write: bool = True,
        load_recordings: bool = False,
    ) -> "Experiment":
        """Load the full experiment.

        Args:
            config: Optional config object passed through to domain objects.
            strict_version: If True, raise on unknown / future annotation versions.
            preflight_scan: If True perform migration scan (mostly used by tests / CLI flows).
            progress_callback: Optional callable invoked as each dataset is about to be loaded.
                Signature: callback(level:str, index:int, total:int, name:str) where:
                    level == "dataset" currently (future: could add session granularity).
                The callback is invoked BEFORE the dataset is actually loaded so the UI can
                display a responsive progress bar even for very large datasets.
        """
        # Prefer index-based discovery to avoid repeated filesystem scans
        try:
            # Only read an existing index; do not trigger rebuilds here.
            from .experiment_index import is_index_stale, load_experiment_index

            lazy_idx = load_experiment_index(self.folder)
            if lazy_idx is not None and not is_index_stale(lazy_idx):
                dataset_folders = [Path(ds.path) for ds in lazy_idx.datasets]
            else:
                raise RuntimeError("Index missing or stale")
        except Exception:
            logging.debug("Falling back to filesystem discovery for datasets.", exc_info=True)
            dataset_folders = [p for p in self.folder.iterdir() if p.is_dir()]
        if preflight_scan:
            try:
                scan_results = scan_annotation_versions(
                    self.folder,
                    include_levels=("experiment", "dataset", "session", "recording"),
                )
                outdated = [r for r in scan_results if r.get("needs_migration")]
                if outdated:
                    logging.info(
                        "Preflight migration scan: %d annotation files need migration (example: %s -> %s)",
                        len(outdated),
                        outdated[0]["path"],
                        outdated[0]["planned_steps"],
                    )
                else:
                    logging.info("Preflight migration scan: all annotation files current.")
            except Exception as e:  # pragma: no cover
                logging.warning(f"Preflight scan failed (non-fatal): {e}")
        datasets = []
        total_datasets = len(dataset_folders)

        # Read concurrency settings from explicit function args. We do not
        # treat lazy_open_h5/load_workers as config-file keys here; they should
        # be passed explicitly by the caller (e.g. the GUI using QSettings).
        max_workers = int(load_workers) if load_workers is not None else 1
        parallel_allowed = max_workers > 1 and bool(lazy_open_h5)

        if parallel_allowed:
            # Use a ThreadPoolExecutor to load datasets concurrently; limit
            # concurrency to not exhaust system resources. We only allow
            # parallel loads when `lazy_open_h5` is True, so h5py handles are
            # not opened concurrently across threads (reduces thread-safety risk).
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_map = {}
                for idx, ds_folder in enumerate(dataset_folders, start=1):
                    # Call the progress callback just before scheduling load
                    if progress_callback is not None:
                        try:
                            progress_callback(level="dataset", index=idx, total=total_datasets, name=ds_folder.name)
                        except InterruptedError:
                            # User canceled - propagate immediately
                            logging.info(f"Dataset loading interrupted at {idx}/{total_datasets} (parallel mode)")
                            # Cancel remaining futures
                            for fut in future_map.keys():
                                fut.cancel()
                            ex.shutdown(wait=False)
                            raise
                        except Exception:  # pragma: no cover - defensive
                            logging.debug("Progress callback errored (ignored)", exc_info=True)
                    future = ex.submit(
                        DatasetRepository(ds_folder).load,
                        config=config,
                        strict_version=strict_version,
                        allow_write=allow_write,
                        load_recordings=load_recordings,
                    )
                    future_map[future] = ds_folder

                # Collect results in original order for deterministic behavior
                for idx, ds_folder in enumerate(dataset_folders, start=1):
                    # Wait for the specific future to complete
                    for f in list(future_map.keys()):
                        if future_map[f] == ds_folder:
                            try:
                                ds_obj = f.result()
                                datasets.append(ds_obj)
                            except ValueError as e:
                                # Handle data validation errors (e.g., inconsistent channels) gracefully
                                if "Inconsistent" in str(e):
                                    # Note: Update regex in experiment_loader.py if this message changes
                                    logging.warning(
                                        f"Dataset '{ds_folder.name}' skipped due to validation error: {e}\n"
                                        f"This dataset has inconsistent data and cannot be loaded. "
                                        f"Please check the recording files in this dataset."
                                    )
                                else:
                                    logging.exception("Dataset load failed (ignored): %s", ds_folder)
                            except Exception:
                                logging.exception("Dataset load failed (ignored): %s", ds_folder)
                                # Skip/continue on failure; caller should handle missing datasets.
                            finally:
                                del future_map[f]
                            break
        else:
            for idx, ds_folder in enumerate(dataset_folders, start=1):
                if progress_callback is not None:
                    try:  # Never let a UI callback break loading UNLESS it's a cancellation
                        progress_callback(level="dataset", index=idx, total=total_datasets, name=ds_folder.name)
                    except InterruptedError:
                        # User canceled - propagate immediately
                        logging.info(f"Dataset loading interrupted at {idx}/{total_datasets}")
                        raise
                    except Exception:  # pragma: no cover - defensive
                        logging.debug("Progress callback errored (ignored)", exc_info=True)
                try:
                    ds_obj = DatasetRepository(ds_folder).load(
                        config=config,
                        strict_version=strict_version,
                        allow_write=allow_write,
                        load_recordings=load_recordings,
                    )
                    datasets.append(ds_obj)
                except ValueError as e:
                    # Handle data validation errors (e.g., inconsistent channels) gracefully
                    # Note: Update regex in experiment_loader.py if this message changes
                    if "Inconsistent" in str(e):
                        logging.warning(
                            f"Dataset '{ds_folder.name}' skipped due to validation error: {e}\n"
                            f"This dataset has inconsistent data and cannot be loaded. "
                            f"Please check the recording files in this dataset."
                        )
                    else:
                        logging.exception("Dataset load failed (ignored): %s", ds_folder)
                except Exception:
                    logging.exception("Dataset load failed (ignored): %s", ds_folder)
                    # Skip/continue on failure; caller should handle missing datasets.

        # Log summary of loaded datasets
        if datasets:
            loaded_count = len(datasets)
            total_count = len(dataset_folders)
            if loaded_count < total_count:
                skipped = total_count - loaded_count
                logging.warning(
                    f"Experiment '{self.folder.name}' loaded with {loaded_count}/{total_count} datasets. "
                    f"{skipped} dataset(s) were skipped due to errors (see warnings above)."
                )
            else:
                logging.info(f"Experiment '{self.folder.name}' loaded successfully with {loaded_count} dataset(s).")
        else:
            logging.error(f"No datasets could be loaded from experiment '{self.folder.name}'.")

        expt_id = self.folder.name  # e.g. "ExperimentRoot"
        if self.expt_js.exists():
            try:
                text = self.expt_js.read_text()
                if not text.strip():
                    logging.warning(f"Experiment annotation file '{self.expt_js}' is empty. Creating a new one.")
                    annot_dict = asdict(ExperimentAnnot.create_empty())
                    if allow_write:
                        self.expt_js.write_text(json.dumps(annot_dict, indent=2))
                else:
                    try:
                        annot_dict = json.loads(text)
                    except json.JSONDecodeError as e:
                        size = None
                        try:
                            size = self.expt_js.stat().st_size
                        except Exception:
                            logging.exception(f"Failed to get size of experiment annotation file '{self.expt_js}'")
                            pass
                        logging.error(f"Failed to decode JSON in experiment annotation '{self.expt_js}' (size={size}): {e}")
                        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
                        corrupt_path = self.expt_js.with_name(f"{self.expt_js.name}.corrupt-{ts}")
                        try:
                            if allow_write:
                                self.expt_js.rename(corrupt_path)
                                logging.warning(f"Moved corrupt experiment annotation to {corrupt_path}. Creating new one.")
                        except Exception:
                            logging.exception(f"Failed to move corrupt experiment annotation file {self.expt_js}")
                        annot_dict = asdict(ExperimentAnnot.create_empty())
                        if allow_write:
                            self.expt_js.write_text(json.dumps(annot_dict, indent=2))

                try:
                    report = migrate_annotation_dict(annot_dict, strict_version=strict_version)
                    if report.changed:
                        logging.debug(
                            f"Experiment annotation migrated {report.original_version}->{report.final_version} for {self.expt_js.name}"
                        )
                        # Persist migrations regardless of allow_write; schema updates must be saved
                        self.expt_js.write_text(json.dumps(annot_dict, indent=2))
                except FutureVersionError as e:
                    logging.error(str(e))
                    raise
                except UnknownVersionError as e:
                    logging.warning(f"Unknown version for {self.expt_js}: {e}. Proceeding without migration.")
                annot = ExperimentAnnot.from_dict(annot_dict)
            except Exception:
                logging.exception(f"Unexpected error while reading experiment annotation {self.expt_js}")
                raise
        else:
            logging.info(f"Experiment annotation file '{self.expt_js}' not found. Using a new empty annotation in-memory.")
            annot = ExperimentAnnot.create_empty()
            if allow_write:
                self.expt_js.write_text(json.dumps(asdict(annot), indent=2))
        expt = Experiment(expt_id, datasets=datasets, annot=annot, repo=self, config=config)
        return expt

    def get_metadata(self) -> dict:
        """
        Get lightweight metadata about the experiment without loading heavy data.
        Returns basic info like dataset count, names, etc.
        """
        try:
            # Get experiment annotation
            if self.expt_js.exists():
                annot_dict = json.loads(self.expt_js.read_text())
                annot = ExperimentAnnot.from_dict(annot_dict)
            else:
                annot = ExperimentAnnot.create_empty()

            # Get dataset folders without loading them
            dataset_folders = [p for p in self.folder.iterdir() if p.is_dir()]
            dataset_metadata = []

            for ds_folder in dataset_folders:
                # Get basic dataset info without loading sessions/recordings
                ds_repo = DatasetRepository(ds_folder)
                ds_meta = ds_repo.get_metadata()
                dataset_metadata.append(ds_meta)

            return {
                "id": self.folder.name,
                "path": str(self.folder),
                "dataset_count": len(dataset_folders),
                "datasets": dataset_metadata,
                "is_completed": annot.is_completed,
                "excluded_datasets": annot.excluded_datasets,
                "data_version": annot.data_version,
                "date_added": annot.date_added,
                "date_modified": annot.date_modified,
            }

        except Exception as e:
            logging.error(f"Failed to get metadata for experiment {self.folder.name}: {e}")
            return {
                "id": self.folder.name,
                "path": str(self.folder),
                "dataset_count": 0,
                "datasets": [],
                "is_completed": False,
                "excluded_datasets": [],
                "data_version": "unknown",
                "error": str(e),
            }

    def save(self, expt: Experiment) -> None:
        """
        Save all datasets in this experiment.
        If I want experiment‐level annotations in the future, write them here.
        This is called when the user edits any dataset's sessions.
        """
        try:
            expt.annot.date_modified = datetime.datetime.now().isoformat(timespec="seconds")
        except Exception:
            logging.debug("Failed to set date_modified on ExperimentAnnot", exc_info=True)
        self.expt_js.write_text(json.dumps(asdict(expt.annot), indent=2))
        # Save ALL datasets including excluded ones to persist their state
        for ds in expt._all_datasets:
            ds.repo.save(ds)
