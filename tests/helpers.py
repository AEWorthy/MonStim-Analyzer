from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def _write_bytes(path: Path, seed: str, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    # repeat digest to desired size
    data = (h * (size // len(h) + 1))[:size]
    path.write_bytes(data)


def make_experiment_and_dataset_with_nested_content(
    root: Path,
    exp_name: str,
    ds_name: str,
    sessions: List[str] | None = None,
) -> Tuple[Path, Path, Dict]:
    """Create an experiment/dataset with nested files for integrity tests.

    Returns: (experiment_path, dataset_path, dataset_metadata_dict)
    """
    sessions = sessions or ["S01", "S02", "S03"]
    exp_path = root / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)

    # Minimal, but present, experiment annot
    (exp_path / "experiment.annot.json").write_text(json.dumps({"data_version": "1.0", "excluded_datasets": []}, indent=2))

    ds_path = exp_path / ds_name
    ds_path.mkdir(parents=True, exist_ok=True)

    # Dataset metadata with specific fields to verify preservation
    ds_meta = {
        "is_completed": True,
        "data_version": "2025.09-test",
        "animal_id": "A1",
        "date": "250101",
        "condition": "cond-X",
        "excluded_sessions": [sessions[1]] if len(sessions) > 1 else [],
    }
    (ds_path / "dataset.annot.json").write_text(json.dumps(ds_meta, indent=2))

    # Create nested session content
    for s in sessions:
        sdir = ds_path / s
        sdir.mkdir(parents=True, exist_ok=True)
        # session annot
        (sdir / "session.annot.json").write_text(json.dumps({"notes": f"session {s}"}, indent=2))
        # raw data placeholder(s)
        _write_bytes(sdir / f"{s}.raw.h5", seed=f"raw-{s}", size=2048)
        # nested auxiliary dirs/files
        _write_bytes(sdir / "aux" / "log.txt", seed=f"log-{s}", size=256)
        _write_bytes(sdir / "deep" / "n1" / "n2" / "blob.bin", seed=f"blob-{s}", size=1536)

    return exp_path, ds_path, ds_meta


def list_files_with_hashes(root: Path) -> Dict[str, Tuple[int, str]]:
    """Return mapping of relative file path -> (size, md5) for all files under root."""
    result: Dict[str, Tuple[int, str]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            rel = str(fpath.relative_to(root)).replace("\\", "/")
            b = fpath.read_bytes()
            md5 = hashlib.md5(b).hexdigest()
            result[rel] = (len(b), md5)
    return result
