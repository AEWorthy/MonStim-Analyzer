#!/usr/bin/env python3
"""Sync `environment.yml` from `requirements.txt`.

Strategy:
- For conda dependencies (top-level string entries), update versions when the package name matches a requirements entry.
- For pip sublist (if present), replace with pip-only requirements (those not present as conda entries).

This script is intended to be run in CI and committed back to the repo when changes occur.
"""

import re
from pathlib import Path

import yaml

DEV_EXCLUDE = {"pytest", "setuptools", "flake8", "black", "isort", "bandit", "safety", "pytest-qt"}

# Packages that should always be managed by conda (do not promote to pip:)
# This prevents Qt bindings/runtimes (PySide6, PySide6, shiboken6, etc.) from
# being accidentally placed in the `pip:` subsection of `environment.yml`.
PIP_EXCLUDE = {
    "PySide6",
    "pyqt",
    "pyside6",
    "PySide6-qt6",
    "PySide6_sip",
    "PySide6-sip",
    "shiboken6",
}


def parse_requirements(path="requirements.txt"):
    reqs = {}
    p = Path(path)
    if not p.exists():
        return reqs
    for line in p.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln or ln.startswith("#"):
            continue
        name = re.split(r"[=<>!~\[]", ln, maxsplit=1)[0].strip()
        if name.lower() in DEV_EXCLUDE:
            continue
        # Prefer == versions if available
        m = re.search(r"==(.+)$", ln)
        ver = m.group(1).strip() if m else None
        reqs[name] = {"raw": ln, "ver": ver}
    return reqs


def load_env(path="environment.yml"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data


def sync(env_path="environment.yml", req_path="requirements.txt"):
    env = load_env(env_path)
    reqs = parse_requirements(req_path)

    deps = env.get("dependencies", [])
    new_deps = []
    conda_updated = set()

    for item in deps:
        if isinstance(item, str):
            name = item.split("=")[0].strip()
            lower = name.lower()
            # If requirements contain this package, update the conda style to name=version
            match = None
            for rname in reqs.keys():
                if rname.lower() == lower:
                    match = rname
                    break
            if match:
                ver = reqs[match]["ver"]
                if ver:
                    new_deps.append(f"{name}={ver}")
                else:
                    # keep original if no exact version available
                    new_deps.append(item)
                conda_updated.add(match.lower())
            else:
                new_deps.append(item)
        else:
            # preserve non-string entries (like dicts) for now
            new_deps.append(item)

    # Build pip list from requirements that were not applied to conda entries
    pip_items = []
    for rname, data in reqs.items():
        lname = rname.lower()
        if lname in conda_updated:
            continue
        # Skip items that should be managed by conda (avoid pip/conda mixing)
        if lname in PIP_EXCLUDE:
            continue
        pip_items.append(data["raw"])

    # Replace or add pip dict
    replaced = False
    for i, item in enumerate(new_deps):
        if isinstance(item, dict) and "pip" in item:
            new_deps[i] = {"pip": pip_items}
            replaced = True
            break
    if not replaced and pip_items:
        new_deps.append({"pip": pip_items})

    env["dependencies"] = new_deps

    Path(env_path).write_text(yaml.safe_dump(env, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    import sys

    try:
        sync()
    except Exception as e:
        print("Error syncing environment.yml:", e)
        sys.exit(2)
    print("environment.yml synced from requirements.txt")
    sys.exit(0)
