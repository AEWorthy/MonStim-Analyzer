# Commands and undo

## Purpose

Use a command for a user-visible mutation that must support application-level undo and redo. Commands live in `monstim_gui/commands.py`; the command invoker owns history and updates the Edit menu.

## Command contract

A command subclass must provide a stable `command_name`, implement `execute()` and `undo()`, and retain exactly the state needed to reverse a successful execution. `execute()` should either complete or leave state unchanged; batch work must roll back already-completed child actions when a later one fails.

`BatchCommand` is a helper, not a `Command` subclass. Do not make it inherit from `Command`: the coverage test intentionally discovers concrete command subclasses.

## Adding a command

1. Identify the domain owner and persistence boundary.
2. Capture pre-change state before mutating it.
3. Implement execute and undo without using the widget as storage.
4. Add the concrete class to `EXPECTED_COMMANDS` in `tests/commands_tests/test_all_commands_coverage.py`.
5. Add a focused execute/undo test. Use a temporary directory for filesystem work.
6. Run the command coverage test and the affected workflow tests.

## Testing checklist

- Execute changes the intended objects and only those objects.
- Undo restores the original values and persisted annotations.
- Redo reapplies the same result.
- A failure does not leave a partial bulk change.
- Menu labels and selection/plot refreshes match the resulting state when applicable.

Some filesystem-heavy commands have structural coverage plus dedicated workflow tests rather than a fully mocked unit test. Keep that exception narrow and document the reason beside the test registration.

## Related topics

- [Architecture](architecture.md)
- [Testing](testing.md)
