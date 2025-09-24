# monstim_signals/io/repositories.py
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

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

    def load(self, config=None) -> "Recording":
        # 1) Load meta JSON (immutable, record‐time facts)
        meta_dict = json.loads(self.meta_js.read_text())
        meta = RecordingMeta.from_dict(meta_dict)

        # 2) Load or create annot JSON (user edits)
        if self.annot_js.exists():
            annot_dict = json.loads(self.annot_js.read_text())
            annot = RecordingAnnot.from_dict(annot_dict)
        else:
            logging.warning(f"Annotation file '{self.annot_js}' not found. Creating a new empty one.")
            annot = RecordingAnnot.create_empty()
            self.annot_js.write_text(json.dumps(asdict(annot), indent=2))

        # 3) Open HDF5 file in read‐only mode; pass the dataset itself (lazy)
        h5file = h5py.File(self.raw_h5, "r")
        raw_dataset: h5py.Dataset = h5file["raw"]  # type: ignore

        # 4) Patch in num_samples from the raw array shape
        meta.num_samples = raw_dataset.shape[0]  # (#samples × #channels)

        # 5) Build the domain object, passing the `h5py.Dataset` directly
        recording = Recording(meta=meta, annot=annot, raw=raw_dataset, repo=self, config=config)
        return recording

    def save(self, recording: Recording) -> None:
        """
        Only rewrite the annot JSON (we assume meta/raw never change).
        This is called when the user edits the recording's annotation.
        """
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

    def load(self, config=None) -> "Session":
        # 1) Discover all recordings in this folder
        recording_repos = list(RecordingRepository.discover_in_folder(self.folder))
        recordings = [repo.load(config=config) for repo in recording_repos]

        # 2) Sort by the primary StimCluster’s stim_v
        recordings.sort(key=lambda r: r.meta.primary_stim.stim_v)

        # 3) Load or create session annotation JSON
        if self.session_js.exists():
            session_annot_dict = json.loads(self.session_js.read_text())
            session_annot = SessionAnnot.from_dict(session_annot_dict)
        else:  # If no session.annot.json, initialize a brand‐new one
            if recordings:
                logging.info(
                    f"Session annotation file '{self.session_js}' not found. Using first recording's meta to create a new one."
                )
                session_annot = SessionAnnot.from_meta(recordings[0].meta)
            else:
                logging.warning(f"Session annotation file '{self.session_js}' not found. Creating a new empty one.")
                session_annot = SessionAnnot.create_empty()
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

    def save(self, session: Session) -> None:
        self.session_js.write_text(json.dumps(asdict(session.annot), indent=2))
        for rec in session.recordings:
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

    def load(self, config=None) -> "Dataset":
        # 1) Discover valid session folders (those with annot or any *.raw.h5)
        session_repos = list(SessionRepository.discover_in_folder(self.folder))

        # 2) Load each Session from discovered repos
        sessions = [repo.load(config=config) for repo in session_repos]

        # 3) Load or create dataset annotation JSON
        if self.dataset_js.exists():
            session_annot_dict = json.loads(self.dataset_js.read_text())
            dataset_annot = DatasetAnnot.from_dict(session_annot_dict)
        else:  # If no session.annot.json, initialize a brand‐new one
            logging.info(f"Session annotation file '{self.dataset_js}' not found. Using the dataset name to create a new one.")
            dataset_annot = DatasetAnnot.from_ds_name(self.dataset_id)
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
        self.dataset_js.write_text(json.dumps(asdict(dataset.annot), indent=2))
        for session in dataset.sessions:
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

    def get_metadata(self) -> dict:
        """
        Get lightweight metadata about the dataset without loading heavy session/recording data.
        Returns basic info like session count, names, completion status, etc.
        """
        try:
            # Get dataset annotation
            if self.dataset_js.exists():
                annot_dict = json.loads(self.dataset_js.read_text())
                dataset_annot = DatasetAnnot.from_dict(annot_dict)
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

    def load(self, config=None) -> "Experiment":
        dataset_folders = [p for p in self.folder.iterdir() if p.is_dir()]
        datasets = [DatasetRepository(ds_folder).load(config=config) for ds_folder in dataset_folders]
        expt_id = self.folder.name  # e.g. "ExperimentRoot"
        if self.expt_js.exists():
            annot_dict = json.loads(self.expt_js.read_text())
            annot = ExperimentAnnot.from_dict(annot_dict)
        else:
            logging.info(f"Experiment annotation file '{self.expt_js}' not found. Creating a new one.")
            annot = ExperimentAnnot.create_empty()
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
        self.expt_js.write_text(json.dumps(asdict(expt.annot), indent=2))
        for ds in expt.datasets:
            ds.repo.save(ds)
