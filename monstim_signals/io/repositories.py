# monstim_signals/io/repositories.py
import json
import h5py
import logging
from pathlib import Path
from typing import Iterator
from dataclasses import asdict

from monstim_signals.core.data_models  import RecordingMeta, RecordingAnnot
from monstim_signals.domain.recording  import Recording
from monstim_signals.domain.session    import Session
from monstim_signals.domain.dataset    import Dataset
from monstim_signals.domain.experiment import Experiment

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
        self.stem    = stem
        self.raw_h5  = stem.with_suffix(".raw.h5")
        self.meta_js = stem.with_suffix(".meta.json")
        self.annot_js= stem.with_suffix(".annot.json")

    def load(self) -> Recording:
        # 1) Load meta JSON (immutable, record‐time facts)
        meta_dict = json.loads(self.meta_js.read_text())
        meta = RecordingMeta.from_dict(meta_dict)

        # 2) Load or create annot JSON (user edits)
        if self.annot_js.exists():
            annot_dict = json.loads(self.annot_js.read_text())
            annot = RecordingAnnot.from_dict(annot_dict)
        else:
            # If no annot.json, initialize a brand‐new one:
            logging.warning(f"Annotation file '{self.annot_js}' not found. Creating a new empty one.")
            annot = RecordingAnnot.from_meta(meta)
            self.annot_js.write_text(json.dumps(asdict(annot_dict), indent=2))

        # 3) Open HDF5 file in read‐only mode; pass the dataset itself (lazy)
        h5file = h5py.File(self.raw_h5, "r")
        raw_dataset = h5file["raw"]

        # 4) Patch in num_samples from the raw array shape
        meta.num_samples = raw_dataset.shape[0]  # (#samples × #channels)

        # 5) Build the domain object, passing the `h5py.Dataset` directly
        recording = Recording(
            meta=meta,
            annot=annot,
            raw=raw_dataset,
            repo=self
        )
        return recording

    def save(self, recording: Recording) -> None:
        """
        Only rewrite the annot JSON (we assume meta/raw never change).
        This is called when the user edits the recording's annotation.
        """
        updated = {
            "excluded": recording.annot.excluded,
            "channels": [ch.__dict__ for ch in recording.annot.channels],
            "cache": recording.annot.cache,
            "version": recording.annot.version
        }
        self.annot_js.write_text(json.dumps(updated, indent=2))

    @staticmethod
    def discover_in_folder(folder: Path) -> Iterator['RecordingRepository']:
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
            logging.info(f"Discovered recording: {stem}")
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

    def load(self) -> Session:
        """
        1) Discover all recordings in this folder
        2) Load each via RecordingRepository
        3) Sort them by stimulus amplitude
        4) Wrap them in a domain `Session` object
        """
        recording_repos = list(RecordingRepository.discover_in_folder(self.folder))

        # 1) Load each Recording object
        recordings = [repo.load() for repo in recording_repos]

        # 2) Sort by the primary StimCluster’s stim_v
        recordings.sort(key=lambda r: r.meta.primary_stim.stim_v)

        # 3) Build a Session domain object
        session_id = self.folder.name  # e.g. "AA00"
        session = Session(
            session_id=session_id,
            recordings=recordings,
            repo=self
        )
        return session

    def save(self, session: Session) -> None:
        for rec in session.recordings:
            # Save each recording's annotation
            rec.repo.save(rec)

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

    def load(self) -> Dataset:
        # 1) Each subfolder of `folder` is a session
        session_folders = [p for p in self.folder.iterdir() if p.is_dir()]

        # 2) Load each Session
        sessions = [SessionRepository(sess_folder).load() for sess_folder in session_folders]

        # 3) Build a Dataset domain object
        dataset_id = self.folder.name  # e.g. "Dataset_01"
        dataset = Dataset(
            dataset_id=dataset_id,
            sessions=sessions,
            repo=self
        )
        return dataset

    def save(self, dataset: Dataset) -> None:
        """
        Save all sessions in this dataset.
        (If I want dataset‐level annotations in the future, write them here.)
        This is called when the user edits any session's recordings.
        """
        for session in dataset.sessions:
            session.repo.save(session)

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

    def load(self) -> Experiment:
        dataset_folders = [p for p in self.folder.iterdir() if p.is_dir()]
        datasets = [DatasetRepository(ds_folder).load() for ds_folder in dataset_folders]
        expt_id = self.folder.name  # e.g. "ExperimentRoot"
        expt = Experiment(expt_id, datasets=datasets, repo=self)
        return expt
    
    def save(self, expt: Experiment) -> None:
        """
        Save all datasets in this experiment.
        If I want experiment‐level annotations in the future, write them here.
        This is called when the user edits any dataset's sessions.
        """
        for ds in expt.datasets:
            ds.repo.save(ds)