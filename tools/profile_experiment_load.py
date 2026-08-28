"""Report cold/warm catalog and experiment-load timings without touching source data.

Usage:
    conda run -n monstim python tools/profile_experiment_load.py PATH-TO-EXPERIMENT

Pass --build-catalog only when intentionally preparing a missing catalog; the
default mode is read-only and refuses to create cache data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

# Running a tool by path puts ``tools/`` first on sys.path rather than the
# repository root. Add the root explicitly so the installed package is not a
# prerequisite for profiling a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monstim_signals.core import load_config
from monstim_signals.io.experiment_catalog import ExperimentCatalog, build_catalog
from monstim_signals.io.repositories import ExperimentRepository


def _seconds(start: float) -> float:
    return round(time.perf_counter() - start, 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile MonStim experiment startup.")
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--build-catalog", action="store_true", help="Build a missing catalog before profiling.")
    parser.add_argument("--workers", type=int, default=1, help="Dataset loader workers (default: 1).")
    arguments = parser.parse_args()
    experiment_path = arguments.experiment.resolve()
    catalog = ExperimentCatalog(experiment_path)
    if not catalog.is_usable():
        if not arguments.build_catalog:
            parser.error(f"No usable catalog at {catalog.path}. Re-run with --build-catalog to prepare it.")
        started = time.perf_counter()
        catalog = build_catalog(experiment_path)
        print(json.dumps({"catalog_build_seconds": _seconds(started), "catalog_path": str(catalog.path)}))

    started = time.perf_counter()
    datasets = catalog.dataset_paths()
    catalog_seconds = _seconds(started)

    config = load_config()
    config.update({"lazy_open_h5": True, "load_workers": max(1, arguments.workers)})
    tracemalloc.start()
    started = time.perf_counter()
    experiment = ExperimentRepository(experiment_path).load(config=config, allow_write=False)
    load_seconds = _seconds(started)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    try:
        print(
            json.dumps(
                {
                    "catalog_query_seconds": catalog_seconds,
                    "experiment_load_seconds": load_seconds,
                    "datasets": len(datasets),
                    "loaded_datasets": len(experiment.datasets),
                    "peak_tracemalloc_mib": round(peak_bytes / 1024 / 1024, 2),
                },
                indent=2,
            )
        )
    finally:
        experiment.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
