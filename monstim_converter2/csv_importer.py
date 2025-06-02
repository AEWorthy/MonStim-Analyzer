import h5py
from pathlib import Path
import os
import json

import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from monstim_converter2.csv_parser import parse
from monstim_signals.core.version import __version__ as VERSION


def discover_files_by_ext(base: Path, pattern='*.csv') -> list[Path]:
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

def csv_to_store(csv_path : Path, output_fp : Path):
    '''Convert a CSV file to an HDF5 file with metadata and data.
    This verion is compatible for MonStim V3D and later.'''

    meta, arr = parse(csv_path)
    meta['session_id'] = output_fp.stem.split('-')[0]  # Use the first part of the filename as session ID
    meta['recording_id'] = output_fp.stem.split('-')[1] if '-' in output_fp.stem else None


    h5_path = output_fp.with_suffix('.raw.h5')
    with h5py.File(h5_path,'w') as h5:
        h5.create_dataset('raw', data=arr,
                          chunks=(min(30000,arr.shape[0]), arr.shape[1]),
                          compression='gzip')
        h5.attrs['scan_rate'] = meta.get('scan_rate')
        h5.attrs['num_channels'] = meta.get('num_channels')
        h5.attrs['channel_types'] = meta.get('channel_types')

    # Write annotations
    meta_path = output_fp.with_suffix('.meta.json')
    if not meta_path.exists():
        with meta_path.open('w') as f:
            json.dump(meta, f, indent=4)

    annot_path = output_fp.with_suffix('.annot.json')
    if not annot_path.exists():
        default = {
          "version":VERSION,
          "excluded":False,
          "channels":[{"invert":False} for _ in range(meta['num_channels'])],
          "cache":{}
        }
        json.dump(default, annot_path.open('w'), indent=2)

if __name__ == '__main__':
    current_path = __file__
    base_path = Path.resolve(Path(current_path).parent.parent)
    data_path = os.path.join(base_path, 'EXAMPLE DATA')
    store_root = Path(os.path.join(base_path, 'data_store'))

    all_csv = discover_files_by_ext(Path(data_path), pattern='*.csv')

    experiments = {}

    for csv_path in all_csv:
        # Parse name/id information from the CSV path/filename
        sess, rec = parse_session_rec(csv_path)
        ds, ex = infer_ds_ex(csv_path, data_path)
        ds = ds or sess # If no dataset name, use session name
        ex = ex or ds # If no experiment name, use dataset name
        
        # Store the CSV path in a nested dictionary structure
        experiments.setdefault(ex, {})\
               .setdefault(ds, {})\
               .setdefault(sess, [])\
               .append(csv_path)
    
    # Print the discovered structure for debugging
    print(f"Found {len(all_csv)} CSV files in {data_path}:")
    for expt_name, ds_dict in experiments.items():
        for ds_name, ses_dict in ds_dict.items():
            for sess_name, csv_list in ses_dict.items():
                print(f"  > Experiment: '{expt_name}', Dataset: '{ds_name}', Session: '{sess_name}', Recordings: {len(csv_list)}")

    # Create the output directory structure
    experiments: dict[str, dict[str, dict[str, list[Path]]]] = experiments
    for expt, ds_dict in experiments.items():
        for ds, ses_dict in ds_dict.items():
            for sess, csv_list in ses_dict.items():
                for csv_path in csv_list:
                    out_dir = store_root / expt / ds / sess# / csv_path.stem
                    out_dir.mkdir(parents=True, exist_ok=True)

                    filename = csv_path.stem
                    output_fp    = out_dir / f"{filename}.ext"
                    annot_fp = output_fp.with_suffix('.annot.json')

                    # create the HDF5 file and meta JSON
                    csv_to_store(csv_path, output_fp)

                    # Create an annotation file if it doesn't exist
                    if not annot_fp.exists():
                        with annot_fp.open('w') as f:
                            meta, _ = parse(csv_path)
                            json.dump(meta, f, indent=4)