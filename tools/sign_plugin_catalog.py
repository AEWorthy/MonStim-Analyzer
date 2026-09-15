"""Sign the official importer catalog with the maintainer's Ed25519 private key.

The private key is intentionally read from an explicit path outside this
repository. The resulting wrapper is safe to publish at the Pages catalog URL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.catalog_signing import write_signed_catalog


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True, help="Unsigned catalog JSON")
    parser.add_argument("--private-key", type=Path, required=True, help="Raw Ed25519 private key outside the repository")
    parser.add_argument("--output", type=Path, required=True, help="Signed Pages catalog destination")
    args = parser.parse_args()
    payload_object = json.loads(args.catalog.read_text(encoding="utf-8"))
    if not isinstance(payload_object.get("plugins", []), list):
        parser.error("catalog must contain a plugins list")
    try:
        write_signed_catalog(payload_object, args.private_key, args.output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Signed catalog written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
