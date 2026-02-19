# Command Testing Strategy

## Overview

The MonStim Analyzer uses the Command Pattern for all user actions that modify data. This document outlines the testing strategy for commands and provides guidance for developers adding new commands.

## Command Pattern Requirements

Every command class must:

1. **Inherit from `Command` base class**
2. **Implement required methods:**
   - `execute()` - Applies the command's changes
   - `undo()` - Reverses the command's changes
   - `get_description()` - Returns human-readable description (inherited)

3. **Set `command_name` attribute** - A user-friendly string shown in the undo/redo menu

4. **Maintain state for undo** - Store necessary information to reverse changes

## Testing Levels

### Level 1: Structural Testing (All Commands)
**File:** `tests/commands_tests/test_all_commands_coverage.py`

Every command must pass structural tests that verify:
- Class exists and inherits from `Command`
- Required methods (`execute`, `undo`, `get_description`) are implemented
- `command_name` is set appropriately

**CI Integration:** The test suite automatically detects new commands and fails if they're not registered in `EXPECTED_COMMANDS`.

### Level 2: Unit Testing (Most Commands)
**Files:** `tests/commands_tests/test_command_pattern_system.py`, others

Commands should have unit tests that verify:
- `execute()` correctly applies changes
- `undo()` correctly reverses changes
- State is properly maintained
- Edge cases are handled

**Approach:**
- Use mocks for GUI components (`Mock()` for simple commands)
- Use real domain objects where meaningful (Session, Dataset, Recording)
- Test execute/undo cycles to verify reversibility

### Level 3: Integration Testing (Complex Commands)
**Files:** `tests/commands_tests/test_move_dataset_commands.py`, `test_restore_dataset_command.py`

Complex commands involving filesystem operations, multiple objects, or GUI state should have dedicated integration tests with:
- Real or realistic test data
- Temporary directories (`temp_output_dir` fixture)
- Full workflow validation

## Untestable Commands

Some commands cannot be reasonably unit tested due to their complexity and dependencies. These are documented in `UNTESTABLE_COMMANDS`:

### DeleteExperimentCommand
**Why untestable:**
- Requires complex GUI state management (experiment dictionary, combo boxes)
- Involves filesystem operations across multiple directories
- Has many side effects on GUI state
- Requires QMessageBox user confirmation

**Alternative validation:**
- Manual testing checklist
- Structure verification (has required methods)

### DeleteDatasetCommand
**Why untestable:**
- Similar to DeleteExperimentCommand
- Requires parent experiment state
- Complex GUI updates
- Filesystem operations with rollback potential

**Alternative validation:**
- Manual testing checklist
- Structure verification

### CopyDatasetCommand
**Why untestable:**
- Deep copy of entire dataset filesystem structure
- Requires generating unique IDs
- Complex state updates across experiment hierarchy
- Difficult to mock without losing test meaning

**Alternative validation:**
- Manual testing with real data
- Structure verification

## Adding a New Command

### 1. Implement the Command Class

```python
class MyNewCommand(Command):
    def __init__(self, gui, ...):
        self.command_name = "My Action Name"
        self.gui = gui
        # Store state needed for undo
        self.old_state = ...
        self.new_state = ...
    
    def execute(self):
        # Apply changes
        ...
    
    def undo(self):
        # Reverse changes using stored state
        ...
```

### 2. Register in Test Suite

Add command name to `EXPECTED_COMMANDS` in `test_all_commands_coverage.py`:

```python
EXPECTED_COMMANDS = {
    ...
    "MyNewCommand",
}
```

### 3. Add Tests

**For simple commands (mock-based):**
```python
class TestMyNewCommand:
    def test_execute_and_undo(self):
        mock_gui = Mock()
        mock_gui.menu_bar = Mock()
        mock_gui.data_selection_widget = Mock()
        
        cmd = MyNewCommand(mock_gui, ...)
        
        # Test execute
        cmd.execute()
        # Assert expected calls/changes
        
        # Test undo
        cmd.undo()
        # Assert reversal
```

**For complex commands (domain object-based):**
```python
class TestMyNewCommand:
    def test_with_real_data(self, fake_gui, temp_output_dir):
        # Create or load test data
        session = ...
        
        cmd = MyNewCommand(fake_gui, session, ...)
        
        # Capture initial state
        initial_value = session.some_property
        
        # Execute
        cmd.execute()
        assert session.some_property == new_value
        
        # Undo
        cmd.undo()
        assert session.some_property == initial_value
```

**If command is untestable:**
1. Add to `UNTESTABLE_COMMANDS` set with explanatory comment
2. Document why in this file
3. Ensure structural tests pass
4. Create manual testing checklist

### 4. Run Tests

```bash
conda activate monstim
pytest tests/commands_tests/test_all_commands_coverage.py -v
```

The CI will fail if:
- New command is detected but not in `EXPECTED_COMMANDS`
- Command doesn't have required methods
- Command tests don't pass

## Continuous Integration

### Automated Checks

The test suite includes automated detection:

```python
def test_no_unexpected_commands():
    """Alert if new commands are added without updating tests."""
    actual_commands = set(get_all_command_classes().keys())
    unexpected = actual_commands - EXPECTED_COMMANDS
    
    if unexpected:
        pytest.fail(
            f"New command classes detected: {unexpected}\n"
            f"Please add them to EXPECTED_COMMANDS and create tests."
        )
```

This ensures:
1. No command is added without being acknowledged
2. Developers are prompted to add tests or justify untestability
3. Command testing coverage is maintained

### Manual Review Checklist

For untestable commands, reviewers should verify:
- [ ] Reason for untestability is documented
- [ ] Command is added to `UNTESTABLE_COMMANDS`
- [ ] Structural tests pass
- [ ] Manual testing procedure is documented
- [ ] Command follows existing patterns for similar operations

## Test Fixtures Available

### From conftest.py
- `temp_output_dir` - Temporary directory for test data
- `fake_gui` - Minimal GUI mock for filesystem operations
- `fake_gui.data_manager` - Pre-configured DataManager

### Creating Test Data
- Use `monstim_signals.testing` helpers where available
- Create minimal test data (don't load full real datasets)
- Clean up in teardown or use temp directories

## Best Practices

1. **Test execute/undo cycles** - Every command should be reversible
2. **Use appropriate mocking** - Mock GUI, use real domain objects
3. **Test edge cases** - Empty lists, None values, boundary conditions
4. **Keep tests fast** - Avoid real file I/O where possible
5. **Document untestability** - Don't just skip tests, explain why
6. **Update this doc** - Keep strategy current as patterns evolve

## Running All Command Tests

```bash
# All command tests
pytest tests/commands_tests/ -v

# Just coverage validation
pytest tests/commands_tests/test_all_commands_coverage.py -v

# Specific command test file
pytest tests/commands_tests/test_move_dataset_commands.py -v
```

## References

- Command Pattern: `monstim_gui/commands.py`
- Test Suite: `tests/commands_tests/`
- CI Configuration: `.github/workflows/` (when implemented)
- Developer Guide: `CONTRIBUTING.md`
