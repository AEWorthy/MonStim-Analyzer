from pathlib import Path
import json
import os
import re

root = Path("data")
experiments = sorted(p for p in root.iterdir() if p.is_dir() and p.name.casefold().startswith("vibes"))
files = []
for experiment in experiments:
    for dataset in sorted(p for p in experiment.iterdir() if p.is_dir()):
        for session in sorted(p for p in dataset.iterdir() if p.is_dir()):
            annotation = session / "session.annot.json"
            if annotation.exists():
                files.append(annotation)

changed = 0
for annotation in files:
    text = annotation.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if parsed.get("is_completed") is True:
        continue
    if "is_completed" not in parsed:
        parsed["is_completed"] = True
        updated = json.dumps(parsed, indent=2) + "\n"
    else:
        updated, replacements = re.subn(r'("is_completed"\s*:\s*)false\b', r"\1true", text, count=1, flags=re.IGNORECASE)
        if replacements != 1:
            raise RuntimeError(f"Could not safely update completion field in {annotation}")
    temporary = annotation.with_name(annotation.name + ".vibes-completion.tmp")
    temporary.write_text(updated, encoding="utf-8", newline="")
    os.replace(temporary, annotation)
    changed += 1

print(f"experiments={len(experiments)} annotation_files={len(files)} changed={changed}")
