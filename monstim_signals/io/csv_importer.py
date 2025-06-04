import json
import h5py
from pathlib import Path
from dataclasses import asdict
from typing import Any
import logging
import numpy as np

from monstim_signals.io.csv_parser import parse
from monstim_signals.core.data_models import RecordingAnnot


def discover_by_ext(base: Path, pattern='*.csv') -> list[Path]:
    """
    Discover all files in the given base directory and its subdirectories that match the given pattern.
    The pattern is typically '*.csv' to find all CSV files: returns a list of Paths to CSV files that are non-empty.
    """
    all_csv = list(base.rglob(pattern))
    return [csv for csv in all_csv if csv.is_file() and csv.stat().st_size > 0]

def parse_session_rec(csv_path: Path):
    """
    Expects name like 'AB12-0034.csv':
      - first part (4 letters/numbers) = session ID
      - second part (4 digits)     = recording ID
    """
    stem = csv_path.stem  # e.g. "AB12-0034"
    if "-" in stem:
        session_id, recording_id = stem.split("-", 1)
        if len(session_id) == 4 and recording_id.isdigit() and len(recording_id) == 4:
            return session_id, recording_id
    return None, None

def infer_ds_ex(csv_path: Path, base_dir: Path):
    """
    dataset_name = immediate parent folder (if not base_dir)
    experiment_name = grandparent folder (if not base_dir)
    """
    parent = csv_path.parent
    dataset_name    = parent.name    if parent != base_dir else None
    grandparent = parent.parent
    experiment_name = grandparent.name if grandparent != base_dir else None
    return dataset_name, experiment_name

def detect_format(path: Path) -> str:
    """
    Inspect the first handful of lines to pick a format tag.
    Returns 'v3h' if it sees the new [Parameters]/[DATA] markers, else 'v3d'.
    """
    with path.open() as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            if line.strip().startswith('[Parameters]'):
                return 'v3h'
            if line.lower().startswith('file version'):
                return 'v3d'
    raise ValueError(f"Could not detect MonStim version for {path}. "
                     "Please ensure it is a valid MonStim CSV file.")

def csv_to_store(csv_path : Path, output_fp : Path, overwrite_h5: bool = False, overwrite_meta: bool = False, overwrite_annot: bool = False):
    '''Convert a CSV file to an HDF5 file with metadata and data.
    This verion is compatible for MonStim V3D and later.'''
    meta_dict : dict[str, Any]
    arr : np.ndarray
    meta_dict, arr = parse(csv_path)
    meta_dict['session_id'] = output_fp.stem.split('-')[0]  # Use the first part of the filename as session ID
    meta_dict['recording_id'] = output_fp.stem.split('-')[1] if '-' in output_fp.stem else None

    h5_path = output_fp.with_suffix('.raw.h5')
    if h5_path.exists() and not overwrite_h5:
        logging.warning(f"HDF5 file {h5_path} already exists. Use 'overwrite=True' to replace it.")
    else:
        if h5_path.exists() and overwrite_h5:
            logging.warning(f"HDF5 file {h5_path} already exists. Overwriting it.")
        elif not h5_path.parent.exists():
            logging.info(f"Creating directory {h5_path.parent} for HDF5 file.")
            h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path,'w') as h5:
            h5.create_dataset('raw', data=arr,
                            chunks=(min(30000,arr.shape[0]), arr.shape[1]),
                            compression='gzip')
            h5.attrs['scan_rate'] = meta_dict.get('scan_rate')
            h5.attrs['num_channels'] = meta_dict.get('num_channels')
            h5.attrs['channel_types'] = meta_dict.get('channel_types')
            h5.attrs['num_samples'] = arr.shape[0]  # (#samples × #channels)

    # Write meta JSON
    meta_path = output_fp.with_suffix('.meta.json')
    if meta_path.exists() and not overwrite_meta:
        logging.warning(f"Meta file {meta_path} already exists. Use 'overwrite_meta=True' to replace it.")
    else:
        if meta_path.exists() and overwrite_meta:
            logging.warning(f"Meta file {meta_path} already exists. Overwriting it.")  
        with meta_path.open('w') as f:
            json.dump(meta_dict, f, indent=4)

    # Write annotation JSON
    annot_path = output_fp.with_suffix('.annot.json')
    if annot_path.exists() and not overwrite_annot:
        logging.warning(f"Annotation file {annot_path} already exists. Use 'overwrite_annot=True' to replace it.")
    else:
        if annot_path.exists() and overwrite_annot:
            logging.warning(f"Annotation file {annot_path} already exists. Overwriting it.")
        with annot_path.open('w') as f:
            annot = RecordingAnnot.from_meta_dict(meta_dict)
            json.dump(asdict(annot), annot_path.open('w'), indent=2)
