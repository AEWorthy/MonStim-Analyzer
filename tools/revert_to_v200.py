import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple

"""
Revert annotations under a folder to data_version '2.0.0' and remove index files.

- Deletes files named '.index.json' or ending with '.index.json'
- Sets/overwrites 'data_version' to '2.0.0' in any '*.annot.json' (experiment/dataset/session/recording)

Usage (PowerShell):
  conda activate alv_lab
  python tools/revert_to_v200.py --root PATH-TO\\TEST_MIGR [--dry-run]
"""

log = logging.getLogger("revert_v200")


def configure_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def is_index_file(p: Path) -> bool:
    name = p.name.lower()
    return name == ".index.json" or name.endswith(".index.json")


def is_annotation_file(p: Path) -> bool:
    return p.suffix.lower() == ".json" and p.name.lower().endswith(".annot.json")


def rewrite_annotation_version(path: Path, dry_run: bool) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log.warning(f"Skipping unreadable JSON: {path} ({e})")
        return False

    changed = False

    # Ensure target data_version
    current = data.get("data_version")
    if current != "2.0.0":
        data["data_version"] = "2.0.0"
        changed = True

    # Remove date_added / date_modified if present
    for k in ("date_added", "date_modified"):
        if k in data:
            del data[k]
            changed = True

    if changed and not dry_run:
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")
            tmp.replace(path)
        except Exception as e:
            log.error(f"Failed to write updated annotation: {path} ({e})")
            return False
    return True


def remove_index_file(path: Path, dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        path.unlink(missing_ok=True)  # type: ignore[arg-type]
        return True
    except Exception as e:
        log.error(f"Failed to delete index file: {path} ({e})")
        return False


def process_root(root: Path, dry_run: bool) -> Tuple[int, int, int, int]:
    ann_total = ann_ok = idx_total = idx_ok = 0
    for p in root.rglob("*.json"):
        if is_index_file(p):
            idx_total += 1
            if remove_index_file(p, dry_run):
                idx_ok += 1
        elif is_annotation_file(p):
            ann_total += 1
            if rewrite_annotation_version(p, dry_run):
                ann_ok += 1
    return ann_total, ann_ok, idx_total, idx_ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Revert annotations to data_version 1.0.0 and remove index files")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "TEST_MIGR",
        help="Root folder to process (defaults to repo/TEST_MIGR)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without modifying files")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    configure_logging(args.verbose)

    root: Path = args.root
    if not root.exists() or not root.is_dir():
        log.error(f"Root path does not exist or is not a directory: {root}")
        return 2

    log.info(f"Processing root: {root}")
    if args.dry_run:
        log.info("Dry-run enabled; no files will be changed.")

    ann_total, ann_ok, idx_total, idx_ok = process_root(root, args.dry_run)

    log.info(f"Annotations processed: {ann_ok}/{ann_total}; Index files removed: {idx_ok}/{idx_total}")
    if args.dry_run and (ann_total or idx_total):
        log.info("Run again without --dry-run to apply changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
