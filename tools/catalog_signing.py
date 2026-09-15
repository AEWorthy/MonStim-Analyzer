"""Shared safe primitives for MonStim's signed release catalogs.

Private keys are deliberately supplied as paths outside the repository.  This
module contains no key-generation or network functionality: it only produces
deterministic signed catalog wrappers from reviewed local inputs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def canonical_payload(catalog: dict[str, Any]) -> bytes:
    """Serialize a catalog exactly as the application verifies it."""
    return json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    """Return the lower-case SHA-256 digest of *path* without loading it all."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    """Replace a text file atomically, avoiding a partially-written catalog."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def signed_wrapper(catalog: dict[str, Any], private_key_path: Path) -> dict[str, str]:
    """Sign *catalog* with a raw 32-byte Ed25519 private key."""
    private_key = private_key_path.read_bytes()
    if len(private_key) != 32:
        raise ValueError("private key must be a 32-byte raw Ed25519 key")
    payload = canonical_payload(catalog)
    signature = Ed25519PrivateKey.from_private_bytes(private_key).sign(payload)
    return {
        "payload": base64.b64encode(payload).decode("ascii"),
        "signature": base64.b64encode(signature).decode("ascii"),
    }


def write_signed_catalog(catalog: dict[str, Any], private_key_path: Path, output: Path) -> None:
    """Write the safe-to-publish signed catalog wrapper."""
    atomic_write_text(output, json.dumps(signed_wrapper(catalog, private_key_path), indent=2) + "\n")
