"""
Golden data import channel-count validation

Purpose: Ensure every imported session from tests/fixtures/golden has >= 2 channels
and the resulting H5/JSON reflect that consistently.
Markers: integration, slow (imports all golden CSVs)
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from monstim_signals.io.csv_importer import import_experiment

from .helpers import get_golden_root

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_imported_sessions_have_at_least_two_channels(tmp_path: Path):
    golden_root = get_golden_root()

    out_expt = tmp_path / "GoldenExp"
    import_experiment(golden_root, out_expt, overwrite=True, max_workers=1)

    # If there are no datasets (e.g., all invalid), that's a test data problem
    ds_dirs = [p for p in out_expt.iterdir() if p.is_dir()]
    assert ds_dirs, "No datasets were imported from golden fixtures"

    issues = []
    for ds in ds_dirs:
        for sess in [p for p in ds.iterdir() if p.is_dir()]:
            metas = list(sess.glob("*.meta.json"))
            h5s = list(sess.glob("*.raw.h5"))

            # Require at least one recording per session
            if not metas or not h5s:
                issues.append(("missing_files", ds.name, sess.name))
                continue

            for mp in metas:
                meta = json.loads(mp.read_text())
                nc = meta.get("num_channels")
                if nc is None or int(nc) < 2:
                    issues.append(("meta_channels", ds.name, sess.name, mp.name, nc))

            for hp in h5s:
                with h5py.File(hp, "r") as f:
                    shape = f["raw"].shape
                    if shape[1] < 2:
                        issues.append(("h5_channels", ds.name, sess.name, hp.name, shape))

    assert not issues, f"Found channel-count issues: {issues}"
