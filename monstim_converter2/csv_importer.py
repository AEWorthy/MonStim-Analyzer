import h5py
from pathlib import Path
import os
import json

from parser import parse
from monstim_signals.core.version import __version__ as VERSION



def discover_csv(base: Path, pattern='*.csv', max_depth = 2):
    all_csv = list(base.rglob("*.csv"))
    return [csv for csv in all_csv if csv.is_file() and csv.stat().st_size > 0]

def parse_session_rec(csv_path: Path):
    """
    Expects name like 'AB12-0034.csv':
      - first part (4 letters/numbers) = session ID
      - second part (4 digits)     = recording ID
    """
    stem = csv_path.stem  # e.g. "AB12-0034"
    if "-" in stem:
        sess, rec = stem.split("-", 1)
        if len(sess) == 4 and rec.isdigit() and len(rec) == 4:
            return sess, rec
    return None, None

def infer_ds_ex(csv_path: Path, base_dir: Path):
    """
    dataset = immediate parent folder (if not base_dir)
    experiment = grandparent folder (if not base_dir)
    """
    parent = csv_path.parent
    dataset    = parent.name    if parent != base_dir else None
    grandparent = parent.parent
    experiment = grandparent.name if grandparent != base_dir else None
    return dataset, experiment

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

def csv_to_store(csv_path, h5_path : Path):
    '''Convert a CSV file to an HDF5 file with metadata and data.
    This verion is compatible for MonStim V3H and later.'''

    meta, arr = parse(csv_path)

    with h5py.File(h5_path,'w') as h5:
        h5.create_dataset('raw', data=arr,
                          chunks=(min(30000,arr.shape[0]), arr.shape[1]),
                          compression='gzip')
        h5.attrs['scan_rate'] = meta.get('scan_rate')
        h5.attrs['num_channels'] = meta.get('num_channels')
        h5.attrs['channel_types'] = meta.get('channel_types')


    # Write annotations
    meta_path = h5_path.with_suffix('.meta.json')
    if not meta_path.exists():
        with meta_path.open('w') as f:
            json.dump(meta, f, indent=4)

    annot_path = h5_path.with_suffix('.annot.json')
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

    all_csv = discover_csv(Path(data_path))

    experiments = {}

    for csv_path in all_csv:
        sess, rec = parse_session_rec(csv_path)
        ds, ex = infer_ds_ex(csv_path, data_path)

        # Failsafe naming
        ds = ds or sess
        ex = ex or ds

        experiments.setdefault(ex, {})\
               .setdefault(ds, {})\
               .setdefault(sess, [])\
               .append(csv_path)
    
    for expt_name, ds_dict in experiments.items():
        for ds_name, ses_dict in ds_dict.items():
            for sess_name, csv_list in ses_dict.items():
                print(f"Experiment: '{expt_name}', Dataset: '{ds_name}', Session: '{sess_name}', Recordings: {len(csv_list)}")

    experiments: dict[str, dict[str, dict[str, list[Path]]]] = experiments
    for expt, ds_dict in experiments.items():
        for ds, ses_dict in ds_dict.items():
            for sess, csv_list in ses_dict.items():
                for csv_path in csv_list:
                    out_dir = store_root / expt / ds / sess / csv_path.stem
                    out_dir.mkdir(parents=True, exist_ok=True)


                    h5_fp    = out_dir / "rec.h5"
                    annot_fp = out_dir / "rec.annot.json"

                    # 1) CSV → HDF5
                    csv_to_store(csv_path, h5_fp)
                    if not annot_fp.exists():
                        with annot_fp.open('w') as f:
                            meta, _ = parse(csv_path)
                            json.dump(meta, f, indent=4)

    
    # csv_to_store(os.path.join(os.path.dirname(current_path), 'rec.csv'),
    #              os.path.join(os.path.dirname(current_path), 'rec.h5'))

    # with h5py.File(os.path.join(os.path.dirname(current_path), 'rec.h5')) as h5:
    #     sr = float(h5.attrs['A/D Monitor Rate (Hz)'])
    #     thirty_ms = int(sr*0.03)
    #     channel_idx = 1
    #     snippet = h5['raw'][:thirty_ms,channel_idx]   # 30-ms of ch0
    #     # list all attributes
    #     attrs = {k: h5.attrs[k] for k in h5.attrs}    
    # print(snippet)
    # print(f"Snippet length: {len(snippet)}")
    # print("Attributes:", attrs)

