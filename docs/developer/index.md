# Developer documentation

This section is for contributors, add-on authors, and anyone evaluating or
changing MonStim's source code. It explains the current architecture and the
contracts that changes must preserve. It is separate from the normal
[application help](../user/index.md).

## Start here

- [Contributing](contributing.md) — local setup, code boundaries, and a change checklist.
- [Architecture](architecture.md) — packages, ownership, persistence, and UI flow.
- [Testing](testing.md) — focused and full validation commands.

## Stateful systems

- [Commands and undo](command_testing_strategy.md) — command contracts and tests.
- [Annotation data versions](data_versioning.md) — safe schema migrations.
- [Application settings](qsettings_management.md) — QSettings keys, migrations, and recovery.

## Documentation, settings, and releases

- [Documentation maintenance](documentation.md) — topic layout, link rules, and packaged help.
- [Publishing releases](releasing.md) — procedure for authorized project maintainers.

For normal application use, return to the [Help Library](../user/index.md).
