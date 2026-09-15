"""Sign the Pages application-update catalog using an external Ed25519 key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.catalog_signing import write_signed_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "1.0" or not isinstance(catalog.get("releases"), list):
        parser.error("catalog must have schema_version '1.0' and a releases list")
    try:
        write_signed_catalog(catalog, args.private_key, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Signed update catalog written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
