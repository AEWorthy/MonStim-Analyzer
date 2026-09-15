"""Generate a dedicated Ed25519 key for the signed application-update catalog."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-key", type=Path, required=True, help="Private key destination outside the repository")
    args = parser.parse_args()
    if args.private_key.exists():
        parser.error("refusing to overwrite an existing private key")
    key = Ed25519PrivateKey.generate()
    args.private_key.parent.mkdir(parents=True, exist_ok=True)
    args.private_key.write_bytes(key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
    print("Public application-update key (embed in monstim_gui.updates):")
    print(base64.b64encode(key.public_key().public_bytes_raw()).decode())
    print("Private key saved outside the repository. Do not commit or paste it into issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
