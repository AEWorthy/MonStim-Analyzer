#!/usr/bin/env python3
"""Check consistency between requirements.txt and environment.yml.

This script tolerates common pip<->conda name differences via a small mapping.
Exit code 0 if consistent, non-zero otherwise.
"""
import re
import sys
from pathlib import Path

import yaml

DEV_EXCLUDE = {"pytest", "setuptools", "flake8", "black", "isort", "bandit", "safety", "pytest-qt"}

# Known name equivalences between pip and conda naming
NAME_MAP = {
    "PySide6": "pyqt",
    "pyqt": "PySide6",
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
        name = re.split(r"[=<>!~\[]", ln, maxsplit=1)[0].strip().lower()
        if name in DEV_EXCLUDE:
            continue
        reqs[name] = ln
    return reqs


def parse_environment(path="environment.yml"):
    env = Path(path)
    if not env.exists():
        return {}, {}
    data = yaml.safe_load(env.read_text(encoding="utf-8")) or {}
    deps = data.get("dependencies", [])
    conda_map = {}
    pip_list = []
    for item in deps:
        if isinstance(item, str):
            parts = item.split("=")
            name = parts[0].strip().lower()
            ver = parts[1].strip() if len(parts) > 1 else None
            conda_map[name] = (item, ver)
        elif isinstance(item, dict) and "pip" in item:
            for p in item["pip"]:
                pip_list.append(p.strip())
    pip_map = {}
    for p in pip_list:
        name = re.split(r"[=<>!~\[]", p, maxsplit=1)[0].strip().lower()
        pip_map[name] = p
    return conda_map, pip_map


def equivalents(name):
    """Yield name and any mapped equivalents for matching."""
    yield name
    if name in NAME_MAP:
        yield NAME_MAP[name]


def main():
    reqs = parse_requirements()
    conda_map, pip_map = parse_environment()

    mismatches = []
    for name, spec in reqs.items():
        found = False
        for candidate in equivalents(name):
            if candidate in conda_map:
                found = True
                _, env_ver = conda_map[candidate]
                m = re.search(r"==(.+)$", spec)
                req_ver = m.group(1).strip() if m else None
                if req_ver and env_ver and req_ver != env_ver:
                    mismatches.append(f"{name}: requirements.txt -> {spec} but environment.yml -> {conda_map[candidate][0]}")
                break
            if candidate in pip_map:
                found = True
                env_spec = pip_map[candidate]
                if env_spec != spec:
                    mismatches.append(f"{name}: requirements.txt -> {spec} but environment.yml(pip) -> {env_spec}")
                break
        if not found:
            mismatches.append(f"{name}: present in requirements.txt ({spec}) but missing from environment.yml")

    if mismatches:
        print("Dependency consistency check FAILED. Differences:")
        for m in mismatches:
            print(" - ", m)
        sys.exit(1)
    print("Dependencies are consistent between requirements.txt and environment.yml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
