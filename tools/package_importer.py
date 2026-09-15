"""Create a deterministic ZIP from an importer pack directory."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Package a MonStim importer add-on.")
    parser.add_argument("source", type=Path, help="Pack directory containing manifest.json")
    parser.add_argument("output", type=Path, help="Destination ZIP")
    args = parser.parse_args()
    if not (args.source / "manifest.json").is_file():
        parser.error("source must contain manifest.json")
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(args.source).as_posix())
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"{digest}  {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
