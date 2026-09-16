# Documentation contribution guide

## Purpose

The in-app help library is a packaged set of Markdown files. This guide
explains how contributors can update it while keeping it accurate, navigable,
and distinct from developer reference material.

## Layout

| Directory | Audience |
| --- | --- |
| `docs/user` | Everyday workflows, troubleshooting, and interface guidance |
| `docs/science` | Calculations, assumptions, units, and scientific review limits |
| `docs/developer` | Architecture, testing, persistence, and maintenance rules |
| `docs/resources` | Shipped configuration and analysis-profile files |

`docs/user/index.md` is the in-app help entry point. Use relative Markdown
links between topics. The help repository accepts links only to Markdown files
inside `docs`, including nested folders; application source files are not help
topics.

## Writing or updating a topic

1. Start with a clear purpose and the user or developer task it supports.
2. Describe the visible behavior or maintenance contract, not the history of how a feature was implemented.
3. State units, defaults, assumptions, and failure behavior where they affect decisions.
4. End with related topics or a link back to the appropriate index.
5. Add a topic when a workflow cannot be explained clearly within an existing page.

## Packaging and validation

`win-main.spec` collects the complete `docs` tree. Configuration and bundled
profiles belong beneath `docs/resources`; the user override `config-user.yml`
is deliberately excluded from release data.

After moving or linking topics, run `tests/gui/test_help_navigation.py`. It
verifies nested links and math-table rendering. Parse or build the PyInstaller
spec in a release environment when packaging dependencies are available.

## Related topics

- [Contributing](contributing.md)
- [Testing](testing.md)
