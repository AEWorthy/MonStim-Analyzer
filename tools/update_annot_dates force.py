"""
Update annotation JSON date fields recursively.
Forcibly updates all files regardless of version or existing dates.

Usage:
    python tools/update_annot_dates.py <root_dir>

Walks the directory recursively and for each *.annot.json file:
 - Sets date_added and date_modified to current datetime in ISO format.
 - Updates data_version to TARGET_VERSION if is_older.

Respects the repository guidance: run within the activated `monstim` env.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path

from packaging.version import parse as parse_version

TARGET_VERSION = parse_version("2.1.0")  # 2.1.0 is first version with date fields
TARGET_DIR = Path(__file__).parent.parent / "data"  # Set to data folder of root


def _is_version_older(v: str | None) -> bool:
    """Return True if v is older than TARGET_VERSION. Missing or malformed
    versions are treated as older.
    """
    if not v or not isinstance(v, str):
        return True
    try:
        return parse_version(v) < TARGET_VERSION
    except Exception:
        # Malformed versions are considered older so we can update them
        return True


def process_file(path: Path, now_iso: str) -> bool:
    # Load existing JSON. If empty, initialize an empty dict.
    try:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            logging.warning(f"Empty annot file: {path}. Initializing basic dates.")
            data = {}
        else:
            data = json.loads(text)
    except Exception as e:
        logging.error(f"Failed reading JSON {path}: {e}")
        return False

    changed = False

    is_older = _is_version_older(data.get("data_version"))
    if is_older:
        # Update version
        data["data_version"] = str(TARGET_VERSION)

    # Set date_added and date_modified
    data["date_added"] = now_iso
    data["date_modified"] = now_iso
    changed = True

    if changed:  # Write updated JSON back to file
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            logging.info(f"Updated dates in {path}")
            return True
        except Exception as e:
            logging.error(f"Failed writing JSON {path}: {e}")
            return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Update annot JSON date fields recursively")
    parser.add_argument("root", type=Path, nargs="?", default=TARGET_DIR, help="Root directory to scan")
    args = parser.parse_args()

    root: Path = args.root

    if not root.exists():
        logging.error(f"Root not found: {root}")
        return 2

    now_iso = datetime.datetime.now().isoformat(timespec="seconds")
    updated = 0
    files_to_process = list(root.rglob("*.annot.json"))
    logging.info(f"Found {len(files_to_process)} annot.json files to process in {root}")
    for path in files_to_process:
        if process_file(path, now_iso):
            updated += 1

    logging.info(f"Done. Files updated: {updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
