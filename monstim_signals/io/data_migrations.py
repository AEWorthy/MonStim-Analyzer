"""Data annotation migration framework.

This module provides a lightweight, explicit migration system for annotation JSON
files (recording/session/dataset/experiment). Each annotation JSON includes a
`data_version` key. When the structure or semantics of these annotation blobs
change, add a forward, *idempotent* migration function here and register it in
`MIGRATIONS`.

Core Design Goals:
* Explicit - each step is a pure(ish) function old_dict -> new_dict
* Ordered - applied strictly in ascending version order with no skipping
* Safe - migrations must be idempotent (running twice produces same result)
* Atomic - original dict left unchanged unless migration fully succeeds (unless
  caller explicitly opts into in_place=True)
* Observable - returns a rich MigrationReport detailing actions
* Extensible - dry-run mode and schema validation hooks

Version Format:
Semantic-style strings (e.g. "2.0.0"). Comparisons are performed by
`parse_version_tuple`. Pre-release or build metadata is ignored; keep versions simple.

Adding a Migration:
1. Bump DATA_VERSION in `monstim_signals/version.py` to the *target* version.
2. Implement a function `def migrate_A_B_C_to_X_Y_Z(data: dict) -> dict:` that:
    * Accepts the loaded JSON dict (NOT yet converted to dataclass)
    * Mutates and returns the provided dict (in-place is fine inside step)
    * Sets `data['data_version']` to the new version string
    * Handles missing legacy keys gracefully
    * Is idempotent (a second call with already-migrated shape is a no-op)
3. Append a `MigrationStep(from_version, to_version, func)` to MIGRATIONS.
4. Add/extend tests (idempotence, forward path).

API Enhancements:
`migrate_annotation_dict(raw, *, dry_run=False, in_place=False, validate=True)`
* dry_run - returns a report without mutating `raw`. Steps planned, not executed.
* in_place - if True, the original dict is mutated; otherwise a copy is migrated.
* validate - if True, run `validate_annotation_schema` post-migration.

Unsupported Cases:
* Downgrades (future version > CURRENT) raise FutureVersionError
* Skipping versions (migrations must chain through every intermediate version)

Developer Checklist for Each Migration:
* Sets final data_version
* Leaves unrelated keys intact
* Handles missing / legacy fields gracefully
* Idempotent
* Tested
"""

# TODO: Implement migrations for meta JSONs as well. Add a schema validation step.

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from monstim_signals.version import DATA_VERSION as CURRENT_DATA_VERSION

# Type alias
MigrationFunc = Callable[[dict], dict]


@dataclass
class MigrationStep:
    from_version: str
    to_version: str
    func: MigrationFunc


@dataclass
class MigrationReport:
    original_version: str
    final_version: str
    steps_applied: List[str]
    changed: bool
    dry_run: bool = False
    field_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class InvalidVersionStringError(RuntimeError):
    """Raised when a version string cannot be parsed."""


class FutureVersionError(RuntimeError):
    """Raised when stored data_version is newer than this code understands."""


class UnknownVersionError(RuntimeError):
    """Raised when a migration path cannot be found from a known version."""


# Registry: ordered list of steps.
# You will append new steps at the *end* as versions advance.
MIGRATIONS: List[MigrationStep] = []


def parse_version_tuple(v: str) -> Tuple[int, int, int]:
    """Parse a semantic version string into a (major, minor, patch) tuple.

    Supported forms (most to least strict):
    * MAJOR.MINOR.PATCH  (e.g. 2.0.1) – canonical
    * MAJOR.MINOR        (e.g. 2.1) → interpreted as (MAJOR, MINOR, 0) with warning
    * Date-like YYYY.MM-suffix (e.g. 2025.09-test) – treated as legacy (1,0,0) with warning

    The relaxed MAJOR.MINOR form exists to tolerate hand‑edited files or early
    prototype exports. A structured warning is emitted so calling code / users
    can normalize their annotation files over time.
    """
    if re.fullmatch(r"\d+\.\d+\.\d+", v):
        try:
            major, minor, patch = v.split(".")
            return int(major), int(minor), int(patch)
        except Exception as e:  # pragma: no cover (regex already constrains)
            raise InvalidVersionStringError(f"Invalid version string: {v}") from e
    # Relaxed MAJOR.MINOR form
    if re.fullmatch(r"\d+\.\d+", v):
        major, minor = v.split(".")
        logging.warning("Annotation data_version '%s' missing patch component; assuming '%s.0'.", v, v)
        return int(major), int(minor), 0
    # Date-like pattern: YYYY.MM-anything
    if re.fullmatch(r"\d{4}\.\d{2}[-_A-Za-z0-9]+", v):
        logging.warning(
            "Non-semver annotation data_version '%s' encountered; treating as legacy '1.0.0' for migration.",
            v,
        )
        return (1, 0, 0)
    raise InvalidVersionStringError(f"Invalid version string: {v}")


# Migration Functions. Add new ones as needed.


def migrate_1_0_0_to_2_0_0(data: dict) -> dict:
    """Full rewrite migration for any annotation prior to 2.0.0.

    Rationale: Pre-2.0.0 annotation structures were inconsistent; instead of
    attempting piecemeal field normalization, we *reconstruct* a canonical
    minimal annotation object based on heuristics that infer its level.

    Heuristics (order matters):
    - Experiment annot if it contains keys suggesting dataset aggregation
      (e.g. 'excluded_datasets')
    - Dataset annot if it has 'excluded_sessions' or fields like 'animal_id'
    - Session annot if it has 'excluded_recordings' or 'channels'
    - Recording annot fallback otherwise

    All legacy / unknown keys are discarded. This guarantees a clean schema.
    Future: If additional mandatory fields are added, update canonical dicts.
    Idempotent: Calling again on an already rewritten 2.0.0 object keeps it stable.
    """
    level = None
    if any(k in data for k in ("excluded_datasets",)):
        level = "experiment"
    elif any(k in data for k in ("excluded_sessions", "animal_id", "condition", "date")):
        level = "dataset"
    elif any(k in data for k in ("excluded_recordings", "channels", "latency_windows")):
        level = "session"
    else:
        level = "recording"

    if level == "experiment":
        rebuilt = {
            "excluded_datasets": data.get("excluded_datasets", []),
            "is_completed": bool(data.get("is_completed", False)),
            "data_version": "2.0.0",
        }
    elif level == "dataset":
        rebuilt = {
            "date": data.get("date"),
            "animal_id": data.get("animal_id"),
            "condition": data.get("condition"),
            "excluded_sessions": data.get("excluded_sessions") or data.get("excluded_session_ids", []) or [],
            "is_completed": bool(data.get("is_completed", False)),
            "data_version": "2.0.0",
        }
    elif level == "session":
        # Preserve existing channels if present and valid; otherwise leave empty for reconstruction.
        preserved_channels = []
        raw_channels = data.get("channels")
        if isinstance(raw_channels, list):
            # Expect list[dict]; filter to known keys and keep as-is
            for ch in raw_channels:
                if isinstance(ch, dict):
                    name = ch.get("name")
                    unit = ch.get("unit")
                    type_override = ch.get("type_override")
                    invert = ch.get("invert")
                    preserved_channels.append(
                        {
                            k: v
                            for k, v in {"name": name, "unit": unit, "type_override": type_override, "invert": invert}.items()
                            if v is not None
                        }
                    )
        elif isinstance(raw_channels, str):
            # Legacy comma-separated or whitespace string; split into names
            parts = [p.strip() for p in raw_channels.replace(";", ",").split(",") if p.strip()]
            for nm in parts:
                preserved_channels.append({"name": nm, "unit": "V"})

        rebuilt = {
            "excluded_recordings": data.get("excluded_recordings", []),
            "latency_windows": [],  # Legacy windows discarded; user can recreate
            "channels": preserved_channels,  # Preserve if any; else empty for later reconstruction
            "m_max_values": [],
            "is_completed": bool(data.get("is_completed", False)),
            "data_version": "2.0.0",
        }
    else:  # recording
        rebuilt = {
            "cache": {},  # Drop legacy cache (potentially incompatible structure)
            "data_version": "2.0.0",
        }

    return rebuilt


def migrate_2_0_0_to_2_0_1(data: dict) -> dict:
    """Only need to update the version number.
    2.0.0-->2.0.1 only adds the migration system itself; no data changes."""
    data["data_version"] = "2.0.1"
    return data


def migrate_2_0_1_to_2_1_0(data: dict) -> dict:
    """Add date_added/date_modified fields and bump version to 2.1.0.

    Applies to all annotation levels (recording/session/dataset/experiment).
    - Preserve existing date_added if present; otherwise set to now (ISO, seconds).
    - Always set/refresh date_modified to now (ISO, seconds).
    - Set data_version to 2.1.0.

    Idempotence: This step only runs when starting at 2.0.1; once at 2.1.0 it
    will not run again. Within a single run, repeated application would simply
    refresh date_modified.
    """
    import datetime as _dt

    now = _dt.datetime.now().isoformat(timespec="seconds")
    if not data.get("date_added"):
        data["date_added"] = now
    # Always update modified to reflect migration time
    data["date_modified"] = now
    data["data_version"] = "2.1.0"
    return data


# Register initial step only if current version is >= target (guard for future refactors)
if parse_version_tuple(CURRENT_DATA_VERSION) >= (2, 0, 0):
    MIGRATIONS.append(MigrationStep("1.0.0", "2.0.0", migrate_1_0_0_to_2_0_0))
if parse_version_tuple(CURRENT_DATA_VERSION) >= (2, 0, 1):
    MIGRATIONS.append(MigrationStep("2.0.0", "2.0.1", migrate_2_0_0_to_2_0_1))
if parse_version_tuple(CURRENT_DATA_VERSION) >= (2, 1, 0):
    MIGRATIONS.append(MigrationStep("2.0.1", "2.1.0", migrate_2_0_1_to_2_1_0))


def _build_step_map(migrations: List[MigrationStep]) -> Dict[str, MigrationStep]:
    step_map: Dict[str, MigrationStep] = {}
    for m in migrations:
        if m.from_version in step_map:
            raise RuntimeError(f"Duplicate migration registration for from_version={m.from_version}")
        step_map[m.from_version] = m
    return step_map


# Build fast lookup from from_version to step (guard duplicates)
_STEP_MAP: Dict[str, MigrationStep] = _build_step_map(MIGRATIONS)


def needs_migration(stored_version: str) -> bool:
    return parse_version_tuple(stored_version) < parse_version_tuple(CURRENT_DATA_VERSION)


def validate_annotation_schema(data: dict) -> None:
    """Lightweight schema validation hook.

    Currently enforces that `data_version` exists and is a non-empty string.
    Can be extended to include required top-level keys for future versions.
    Raise a RuntimeError (or custom future ValidationError) on violation.
    """
    if "data_version" not in data or not isinstance(data["data_version"], str) or not data["data_version"]:
        raise RuntimeError("Annotation missing valid 'data_version' after migration.")


def _compute_field_changes(before: dict, after: dict) -> Dict[str, Tuple[Any, Any]]:
    changes: Dict[str, Tuple[Any, Any]] = {}
    for k in set(before.keys()).union(after.keys()):
        b = before.get(k, object())
        a = after.get(k, object())
        if b is object() and a is not object():
            changes[k] = (None, a)
        elif a is object() and b is not object():
            changes[k] = (b, None)
        elif b != a:
            changes[k] = (b, a)
    return changes


def migrate_annotation_dict(
    raw: dict,
    *,
    dry_run: bool = False,
    in_place: bool = False,
    validate: bool = True,
    strict_version: bool = False,
) -> MigrationReport:
    """Migrate an annotation dictionary to CURRENT_DATA_VERSION.

    Parameters
    ----------
    raw : dict
        Source annotation dict.
    dry_run : bool, default False
        If True, compute the sequence of steps but do NOT modify the input dict.
    in_place : bool, default False
        If True (and not dry_run), mutate `raw`. Otherwise operate on a shallow copy.
    validate : bool, default True
        Run schema validation after successful migration.
    strict_version : bool, default False
        If True, raise FutureVersionError if stored version is newer than supported.
    """
    # Determine initial version (bootstrap if missing)
    if "data_version" not in raw or not raw["data_version"]:
        logging.warning("Annotation missing data_version; assuming legacy '1.0.0'.")
        bootstrap_version = "1.0.0"
    else:
        bootstrap_version = raw["data_version"]

    # Canonicalize version string (e.g., '1.0' -> '1.0.0') so that step lookup works.
    try:
        m, n, p = parse_version_tuple(bootstrap_version)
        canonical = f"{m}.{n}.{p}"
        if canonical != bootstrap_version:
            logging.debug("Canonicalizing annotation data_version '%s' -> '%s'", bootstrap_version, canonical)
            bootstrap_version = canonical
    except InvalidVersionStringError:
        # Will be raised again below consistently; let existing tests capture.
        raise

    # Future version guard. In strict mode raise immediately. In non-strict
    # mode we downgrade to a warning so that users with newer annotations can
    # still *attempt* to open them (best‑effort forward compatibility) but are
    # encouraged to upgrade.
    if parse_version_tuple(bootstrap_version) > parse_version_tuple(CURRENT_DATA_VERSION):
        msg = (
            f"Stored data_version {bootstrap_version} is newer than supported {CURRENT_DATA_VERSION}. "
            "Please upgrade package."
        )
        if strict_version:
            raise FutureVersionError(msg)
        logging.warning("%s (non-strict mode - proceeding without migration)", msg)
        # Nothing else to do; we cannot migrate forward. Return a dry style report.
        return MigrationReport(bootstrap_version, bootstrap_version, [], changed=False, dry_run=dry_run)

    if not needs_migration(bootstrap_version):
        # Ensure version key exists when missing but no migration needed (e.g., already current but absent)
        if "data_version" not in raw or not raw["data_version"]:
            if not dry_run:
                raw["data_version"] = CURRENT_DATA_VERSION
        return MigrationReport(bootstrap_version, bootstrap_version, [], changed=False, dry_run=dry_run)

    # Plan steps
    planned_steps: List[MigrationStep] = []
    current = bootstrap_version
    safety_counter = 0
    while current != CURRENT_DATA_VERSION:
        safety_counter += 1
        if safety_counter > 50:
            raise RuntimeError("Excessive migration steps; possible cycle.")
        step = _STEP_MAP.get(current)
        if not step:
            raise UnknownVersionError(
                f"No migration path from {current} to {CURRENT_DATA_VERSION}. Missing step in MIGRATIONS."
            )
        planned_steps.append(step)
        current = step.to_version

    if dry_run:
        return MigrationReport(
            bootstrap_version,
            CURRENT_DATA_VERSION,
            [f"{s.from_version}->{s.to_version}" for s in planned_steps],
            changed=bool(planned_steps),
            dry_run=True,
        )

    # Perform migration on copy unless in_place requested
    working = raw if in_place else dict(raw)
    before_snapshot = dict(working)

    executed_steps: List[str] = []
    for step in planned_steps:
        logging.debug(f"Migrating annotation {step.from_version} -> {step.to_version}")
        result = step.func(working)
        # If the migration function returned a *different* dict object while in_place=True,
        # we must sync changes back to the original object so callers observing `raw` see updates.
        if in_place and result is not working:
            working.clear()
            working.update(result)
        else:
            working = result
        executed_steps.append(f"{step.from_version}->{step.to_version}")
        # Ensure version advanced
        if working.get("data_version") != step.to_version:
            raise RuntimeError(f"Migration function {step.func.__name__} did not set data_version to {step.to_version}")

    if validate:
        validate_annotation_schema(working)

    # Commit back if not in_place
    if not in_place:
        raw.update(working)

    field_changes = _compute_field_changes(before_snapshot, working)
    return MigrationReport(
        bootstrap_version,
        working.get("data_version", bootstrap_version),
        executed_steps,
        changed=bool(executed_steps),
        dry_run=False,
        field_changes=field_changes,
    )


__all__ = [
    "MigrationStep",
    "MigrationReport",
    "FutureVersionError",
    "InvalidVersionStringError",
    "UnknownVersionError",
    "migrate_annotation_dict",
    "needs_migration",
    "MIGRATIONS",
    "validate_annotation_schema",
]


# ---- High-level scan utility -------------------------------------------------
def scan_annotation_versions(
    root: "Path | str", *, include_levels: Iterable[str] = ("experiment", "dataset", "session", "recording")
) -> List[dict]:
    """Scan an experiment root for annotation files and summarize migration needs.

    Parameters
    ----------
    root : Path | str
        Experiment root directory containing dataset folders (and optionally
        an experiment annot file).
    include_levels : iterable[str]
        Subset of annotation levels to include. Defaults to all.

    Returns
    -------
    list of dict with keys:
        path: str – file path
        level: str – annotation level
        version: str – discovered version (or 'missing')
        needs_migration: bool
        planned_steps: list[str] – empty if none or unknown path
    """
    from pathlib import Path  # local import to avoid circulars in some contexts

    root_path = Path(root)
    results: List[dict] = []

    def _plan(v: str) -> List[str]:
        try:
            # Canonicalize for planning
            m, n, p = parse_version_tuple(v)
            v_canon = f"{m}.{n}.{p}"
            if not needs_migration(v_canon):
                return []
            # replicate planning logic quickly
            seq = []
            current = v_canon
            safety = 0
            while current != CURRENT_DATA_VERSION:
                safety += 1
                if safety > 50:
                    break
                step = _STEP_MAP.get(current)
                if not step:
                    break
                seq.append(f"{step.from_version}->{step.to_version}")
                current = step.to_version
            return seq
        except InvalidVersionStringError:
            return []

    # Experiment level
    if "experiment" in include_levels:
        expt_file = root_path / "experiment.annot.json"
        if expt_file.exists():
            try:
                import json

                data = json.loads(expt_file.read_text())
                version_raw = data.get("data_version") or "missing"
                if version_raw not in ("missing",):
                    try:
                        m, n, p = parse_version_tuple(version_raw)
                        version = f"{m}.{n}.{p}"
                    except InvalidVersionStringError:
                        version = version_raw
                else:
                    version = version_raw
            except Exception:
                version = "unreadable"
            results.append(
                {
                    "path": str(expt_file),
                    "level": "experiment",
                    "version": version,
                    "needs_migration": version not in ("missing", "unreadable") and needs_migration(version),
                    "planned_steps": _plan(version) if version not in ("missing", "unreadable") else [],
                }
            )
    # Dataset/session/recording walk
    for ds_folder in root_path.iterdir():
        if not ds_folder.is_dir():
            continue
        # dataset annot
        if "dataset" in include_levels:
            ds_annot = ds_folder / "dataset.annot.json"
            if ds_annot.exists():
                try:
                    import json

                    data = json.loads(ds_annot.read_text())
                    version_raw = data.get("data_version") or "missing"
                    if version_raw not in ("missing",):
                        try:
                            m, n, p = parse_version_tuple(version_raw)
                            version = f"{m}.{n}.{p}"
                        except InvalidVersionStringError:
                            version = version_raw
                    else:
                        version = version_raw
                except Exception:
                    version = "unreadable"
                results.append(
                    {
                        "path": str(ds_annot),
                        "level": "dataset",
                        "version": version,
                        "needs_migration": version not in ("missing", "unreadable") and needs_migration(version),
                        "planned_steps": _plan(version) if version not in ("missing", "unreadable") else [],
                    }
                )
        # sessions
        for session_folder in ds_folder.iterdir():
            if not session_folder.is_dir():
                continue
            if "session" in include_levels:
                sess_annot = session_folder / "session.annot.json"
                if sess_annot.exists():
                    try:
                        import json

                        data = json.loads(sess_annot.read_text())
                        version_raw = data.get("data_version") or "missing"
                        if version_raw not in ("missing",):
                            try:
                                m, n, p = parse_version_tuple(version_raw)
                                version = f"{m}.{n}.{p}"
                            except InvalidVersionStringError:
                                version = version_raw
                        else:
                            version = version_raw
                    except Exception:
                        version = "unreadable"
                    results.append(
                        {
                            "path": str(sess_annot),
                            "level": "session",
                            "version": version,
                            "needs_migration": version not in ("missing", "unreadable") and needs_migration(version),
                            "planned_steps": _plan(version) if version not in ("missing", "unreadable") else [],
                        }
                    )
            # recordings
            if "recording" in include_levels:
                for raw_h5 in session_folder.glob("*.raw.h5"):
                    stem = raw_h5.with_suffix("")
                    rec_annot = stem.with_suffix(".annot.json")
                    if rec_annot.exists():
                        try:
                            import json

                            data = json.loads(rec_annot.read_text())
                            version_raw = data.get("data_version") or "missing"
                            if version_raw not in ("missing",):
                                try:
                                    m, n, p = parse_version_tuple(version_raw)
                                    version = f"{m}.{n}.{p}"
                                except InvalidVersionStringError:
                                    version = version_raw
                            else:
                                version = version_raw
                        except Exception:
                            version = "unreadable"
                        results.append(
                            {
                                "path": str(rec_annot),
                                "level": "recording",
                                "version": version,
                                "needs_migration": version not in ("missing", "unreadable") and needs_migration(version),
                                "planned_steps": _plan(version) if version not in ("missing", "unreadable") else [],
                            }
                        )
    return results


__all__.append("scan_annotation_versions")
