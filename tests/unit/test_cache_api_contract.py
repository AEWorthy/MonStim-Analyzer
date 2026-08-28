import re
from pathlib import Path


def test_removed_broad_cache_reset_has_no_repository_invocations():
    root = Path(__file__).parents[2]
    removed_name = "reset_" + "all_caches"
    invocation = re.compile(rf"\.{removed_name}\s*\(")
    offenders = []
    for source_root in (root / "monstim_gui", root / "monstim_signals", root / "tests"):
        for path in source_root.rglob("*.py"):
            if path == Path(__file__):
                continue
            if invocation.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(root)))
    assert offenders == []
