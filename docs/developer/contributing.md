# Contributing to MonStim

## Purpose

This guide is for external contributors preparing a pull request or a
documentation change. It introduces the project conventions that protect
scientific behavior and the workflows researchers use.

## Local setup

Create the supported environment from the repository root, then run project
commands through it:

```powershell
conda env create -f environment.yml
conda run -n monstim python -m pytest <target>
conda run -n monstim ruff check <paths>
```

For pytest runs on Windows, provide a new temporary directory outside the
checkout and disable the cache provider. For example:

```powershell
conda run -n monstim python -m pytest <target> -q --basetemp C:\tmp\monstim-pytest-<run-id> -p no:cacheprovider
```

## Code boundaries

| Area | Responsibility | Keep out of it |
| --- | --- | --- |
| `monstim_signals` | Domain objects, signal transforms, repositories, and data models | Qt widgets and GUI-only state |
| `monstim_gui` | Dialogs, widgets, menus, commands, and application orchestration | Scientific calculations duplicated from the domain |
| `docs/resources` | Shipped default configuration and profiles | User-edited data or local settings |
| `tests` | Isolated regression coverage and curated fixtures | Writes to repository data or fixtures |

Latency windows are stored with session annotations. Dataset and experiment edits are bulk operations over their child sessions; preserve that ownership model in new code.

## Change checklist

1. Locate the domain owner before changing a calculation or persistence rule.
2. Preserve undo/redo behavior for user-visible edits; see [Commands and undo](command_testing_strategy.md).
3. Add or update focused tests with temporary output paths.
4. Update the relevant user or developer document when behavior, defaults, or troubleshooting changes.
5. Run the smallest relevant test set, lint touched Python files, and run `git diff --check`.

## Related topics

- [Architecture](architecture.md)
- [Testing](testing.md)
- [Documentation maintenance](documentation.md)
