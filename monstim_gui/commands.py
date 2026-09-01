from __future__ import annotations

import abc
import copy
import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QMessageBox

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI

logger = logging.getLogger(__name__)


def _refresh_data_views(gui, *experiment_ids):
    """Refresh real GUI views while remaining compatible with command unit mocks."""
    data_manager = getattr(gui, "data_manager", None)
    refresh = getattr(data_manager, "refresh_data_views", None)
    experiments = getattr(gui, "expts_dict", None)
    if not callable(refresh) or not isinstance(experiments, dict):
        return
    paths = [Path(experiments[experiment_id]) for experiment_id in experiment_ids if experiment_id in experiments]
    refresh(*paths)


def _cancel_cache_warmup(gui) -> None:
    coordinator = getattr(gui, "cache_warmup", None)
    if coordinator is not None:
        coordinator.cancel_and_wait()


class Command(abc.ABC):
    command_name: str = None

    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def undo(self):
        pass

    def get_description(self) -> str:
        """Return a human-readable description of this command."""
        return getattr(self, "command_name", type(self).__name__)


class BatchCommand:
    """Group already-compatible commands into one atomic undo-history entry."""

    def __init__(self, command_name: str, commands: list[Command]):
        self.command_name = command_name
        self.commands = list(commands)
        self._executed: list[Command] = []

    def execute(self):
        self._executed.clear()
        try:
            for command in self.commands:
                command.execute()
                self._executed.append(command)
        except Exception:
            for command in reversed(self._executed):
                command.undo()
            self._executed.clear()
            raise

    def undo(self):
        for command in reversed(self._executed or self.commands):
            command.undo()

    def get_description(self) -> str:
        return self.command_name


class CommandInvoker:
    def __init__(self, parent: MonstimGUI):
        self.parent: MonstimGUI = parent  # type: MonstimGUI
        # Limit history to avoid unbounded memory growth in long-running sessions
        # Default max history retains the most recent 100 commands (configurable)
        self.max_history = 100
        self.history = deque()  # type: deque[Command]
        self.redo_stack = deque()  # type: deque[Command]

    def execute(self, command: Command):
        command.execute()
        self.history.append(command)
        # Trim oldest history entries if we exceed max_history
        try:
            while self.max_history is not None and len(self.history) > self.max_history:
                self.history.popleft()
        except Exception:
            logger.warning("Non-fatal: Command history trimming failed.", exc_info=True)
        self.redo_stack.clear()
        self.parent.menu_bar.update_undo_redo_labels()
        # Always refresh notice icons after a command executes so diagnostics stay in sync with domain state.
        try:
            self.parent.data_selection_widget.refresh_notice_icons()
        except Exception as e:
            logger.warning("Non-fatal: refresh_notice_icons failed after execute: %s", e, exc_info=True)

    def undo(self):
        if self.history:
            # Do not remove history until persistence has succeeded.  This is
            # particularly important for disk-backed commands: a failed undo
            # remains available for the user to retry.
            command = self.history[-1]
            command.undo()
            self.history.pop()
            self.redo_stack.append(command)
            self.parent.menu_bar.update_undo_redo_labels()
            try:
                self.parent.data_selection_widget.refresh_notice_icons()
            except Exception as e:
                logger.warning("Non-fatal: refresh_notice_icons failed after undo: %s", e, exc_info=True)

    def redo(self):
        if self.redo_stack:
            # Likewise retain a failed redo in its original stack.
            command = self.redo_stack[-1]
            command.execute()
            self.redo_stack.pop()
            self.history.append(command)
            self.parent.menu_bar.update_undo_redo_labels()
            try:
                self.parent.data_selection_widget.refresh_notice_icons()
            except Exception as e:
                logger.warning("Non-fatal: refresh_notice_icons failed after redo: %s", e, exc_info=True)

    def get_undo_command_name(self):
        if self.history:
            return self.history[-1].command_name
        return None

    def get_redo_command_name(self):
        if self.redo_stack:
            return self.redo_stack[-1].command_name
        return None

    def remove_command_by_name(self, command_name: str):
        # Remove all occurrences from history
        self.history = deque(command for command in self.history if command.command_name != command_name)

        # Remove all occurrences from redo_stack
        self.redo_stack = deque(command for command in self.redo_stack if command.command_name != command_name)


# GUI command classes
class ExcludeRecordingCommand(Command):
    def __init__(self, gui, recording_id: str):
        self.command_name: str = "Exclude Recording"
        self.gui: MonstimGUI = gui
        self.recording_id: str = recording_id

    def execute(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_id)
            # Recording exclusion doesn't affect dataset/session selections, so use sync instead
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

    def undo(self):
        try:
            self.gui.current_session.restore_recording(self.recording_id)
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

    def get_description(self) -> str:
        return f"Excluded recording '{self.recording_id}'"


class RestoreRecordingCommand(Command):
    def __init__(self, gui, recording_id: str):
        self.command_name: str = "Restore Recording"
        self.gui: MonstimGUI = gui
        self.recording_id = recording_id

    def execute(self):
        try:
            self.gui.current_session.restore_recording(self.recording_id)
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

    def undo(self):
        try:
            self.gui.current_session.exclude_recording(self.recording_id)
            self.gui.data_selection_widget.sync_combo_selections()
        except ValueError as e:
            QMessageBox.critical(self.gui, "Error", str(e))

    def get_description(self) -> str:
        return f"Restored recording '{self.recording_id}'"


class ExcludeSessionCommand(Command):
    """Exclude the currently selected session."""

    def __init__(self, gui):
        self.command_name = "Exclude Session"
        self.gui: MonstimGUI = gui
        self.removed_session = None
        self.session_id = None
        self.idx = None
        self.previous_dataset = None

    def execute(self):
        # Verify we have valid session and dataset
        if not self.gui.current_session or not self.gui.current_dataset:
            logger.warning("Cannot exclude session: No session or dataset is currently selected.")
            return  # Exit gracefully

        self.removed_session = self.gui.current_session
        self.session_id = self.gui.current_session.id
        self.previous_dataset = self.gui.current_dataset  # Preserve dataset selection

        # Verify the session is in the dataset's sessions list before excluding
        try:
            self.idx = self.gui.current_dataset.sessions.index(self.gui.current_session)
        except ValueError:
            # Session is not in the list - it may have already been excluded
            # (e.g., when all its recordings were excluded)
            logger.warning(
                f"Cannot exclude session '{self.session_id}': Session is not in the dataset's sessions list. "
                f"It may have already been excluded (e.g., by excluding all its recordings)."
            )
            return  # Exit gracefully without making changes

        self.gui.current_dataset.exclude_session(self.session_id)
        # Determine new selection: try next session at same index, else previous.
        new_current = None
        remaining_sessions = self.gui.current_dataset.sessions
        if remaining_sessions:
            new_current = remaining_sessions[self.idx] if self.idx < len(remaining_sessions) else remaining_sessions[-1]
        self.gui.current_session = new_current
        # Update session list; keep dataset selection
        self.gui.data_selection_widget.update(levels=("session",))
        # Reflect new selection in combo (block signals to avoid recursive loads)
        if new_current:
            try:
                session_index = self.gui.current_dataset.sessions.index(new_current)
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(session_index)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except ValueError:
                pass  # Session not found in the list (may have been removed); safe to ignore.
        else:
            # No sessions left; clear plots
            if hasattr(self.gui, "plot_widget"):
                try:
                    self.gui.plot_widget.on_data_selection_changed()
                except Exception:
                    logger.warning("Plot refresh after session exclusion (no sessions left) failed (non-fatal).", exc_info=True)

        # Always refresh plots after exclusion to reflect new session
        if self.gui.current_session and hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception:
                logger.warning("Plot refresh after session exclusion failed (non-fatal).", exc_info=True)

    def undo(self):
        self.gui.current_dataset.restore_session(self.session_id)
        self.gui.current_session = self.removed_session
        # Ensure we maintain the correct dataset selection
        if self.previous_dataset and self.gui.current_dataset != self.previous_dataset:
            self.gui.current_dataset = self.previous_dataset
        # Update session list and set the correct selection
        self.gui.data_selection_widget.update(levels=("session",))
        if self.removed_session:
            try:
                session_index = self.gui.current_dataset.sessions.index(self.removed_session)
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(session_index)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except ValueError:
                pass  # Session not found in list
        # Refresh plots to reflect restored session
        if self.gui.current_session and hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception:
                logger.warning("Plot refresh after session exclusion undo failed (non-fatal).", exc_info=True)


class ExcludeDatasetCommand(Command):
    """Exclude the currently selected dataset."""

    def __init__(self, gui):
        self.command_name = "Exclude Dataset"
        self.gui: MonstimGUI = gui
        self.removed_dataset = None
        self.dataset_id = None
        self.idx = None
        self.previous_experiment = None

    def execute(self):
        # Verify we have valid dataset and experiment
        if not self.gui.current_dataset or not self.gui.current_experiment:
            logger.warning("Cannot exclude dataset: No dataset or experiment is currently selected.")
            return  # Exit gracefully

        # Capture state prior to exclusion
        self.removed_dataset = self.gui.current_dataset
        self.dataset_id = self.gui.current_dataset.id

        # Verify the dataset is in the experiment's datasets list before excluding
        if self.gui.current_dataset not in self.gui.current_experiment.datasets:
            logger.warning(
                f"Cannot exclude dataset '{self.dataset_id}': Dataset is not in the experiment's datasets list. It may have already been excluded."
            )
            return  # Exit gracefully without making changes

        self.idx = self.gui.current_experiment.datasets.index(self.gui.current_dataset)
        self.previous_experiment = self.gui.current_experiment  # Preserve experiment selection

        # Perform exclusion in domain
        self.gui.current_experiment.exclude_dataset(self.dataset_id)

        # Determine next dataset selection (next at same index if available, else previous, else none)
        remaining = self.gui.current_experiment.datasets
        new_dataset = None
        if remaining:
            new_dataset = remaining[self.idx] if self.idx is not None and self.idx < len(remaining) else remaining[-1]

        self.gui.current_dataset = new_dataset
        # Reset session selection relative to new dataset
        if new_dataset:
            sessions_attr = getattr(new_dataset, "sessions", None)
            if isinstance(sessions_attr, (list, tuple)) and sessions_attr:
                try:
                    self.gui.current_session = sessions_attr[0]
                except Exception:
                    self.gui.current_session = None
            else:
                self.gui.current_session = None
        else:
            self.gui.current_session = None

        # Update combos: dataset then session
        self.gui.data_selection_widget.update(levels=("dataset",))
        if new_dataset:
            try:
                ds_index = self.gui.current_experiment.datasets.index(new_dataset)
                self.gui.data_selection_widget.dataset_combo.blockSignals(True)
                self.gui.data_selection_widget.dataset_combo.setCurrentIndex(ds_index)
                self.gui.data_selection_widget.dataset_combo.blockSignals(False)
            except ValueError as e:
                logger.warning(f"Index error during dataset exclusion execute: {e}")
        self.gui.data_selection_widget.update(levels=("session",))
        if self.gui.current_session:
            try:
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(0)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except Exception as e:
                logger.warning(f"Non-fatal: session combo update failed after dataset exclusion: {e}", exc_info=True)

        # Trigger downstream updates (plots etc.)
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception:
                logger.debug("Plot refresh after dataset exclusion failed (non-fatal).", exc_info=True)

    def undo(self):
        # Restore dataset in domain
        if self.dataset_id is not None:
            self.gui.current_experiment.restore_dataset(self.dataset_id)

        # Re-acquire restored dataset reference safely
        try:
            restored = next(ds for ds in self.gui.current_experiment.datasets if ds.id == self.dataset_id)
        except StopIteration:
            restored = self.removed_dataset  # fallback to prior object reference

        # Maintain experiment selection
        if self.previous_experiment and self.gui.current_experiment != self.previous_experiment:
            self.gui.current_experiment = self.previous_experiment

        self.gui.current_dataset = restored

        # Update dataset combo first
        self.gui.data_selection_widget.update(levels=("dataset",))
        if restored:
            try:
                ds_index = self.gui.current_experiment.datasets.index(restored)
                self.gui.data_selection_widget.dataset_combo.blockSignals(True)
                self.gui.data_selection_widget.dataset_combo.setCurrentIndex(ds_index)
                self.gui.data_selection_widget.dataset_combo.blockSignals(False)
            except ValueError as e:
                logger.warning(f"Index error during dataset exclusion undo: {e}")

        # Update sessions and select first session (consistent with RestoreDatasetCommand)
        self.gui.data_selection_widget.update(levels=("session",))
        if restored:
            sessions_attr = getattr(restored, "sessions", None)
            if isinstance(sessions_attr, (list, tuple)) and sessions_attr:
                try:
                    self.gui.current_session = sessions_attr[0]
                except Exception as e:
                    self.gui.current_session = None
                    logger.warning(f"Non-fatal: session selection failed after dataset exclusion undo: {e}", exc_info=True)
            else:
                self.gui.current_session = None
        else:
            self.gui.current_session = None
        if self.gui.current_session:
            try:
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(0)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except Exception as e:
                logger.warning(f"Non-fatal: session combo update failed after dataset exclusion undo: {e}", exc_info=True)

        # Trigger downstream updates
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception:
                logger.debug("Plot refresh after dataset exclusion undo failed (non-fatal).", exc_info=True)


class RestoreSessionCommand(Command):
    """Restore an excluded session by ID."""

    def __init__(self, gui, session_id: str):
        self.command_name = "Restore Session"
        self.gui: MonstimGUI = gui
        self.session_id = session_id
        self.session_obj = None

    def execute(self):
        self.session_obj = next(
            (s for s in self.gui.current_dataset.get_all_sessions(include_excluded=True) if s.id == self.session_id),
            None,
        )
        self.gui.current_dataset.restore_session(self.session_id)
        self.gui.current_session = self.session_obj
        # Only update session list and sync its selection
        self.gui.data_selection_widget.update(levels=("session",))
        if self.session_obj:
            # Find the index of the restored session
            try:
                session_index = self.gui.current_dataset.sessions.index(self.session_obj)
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(session_index)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except ValueError as e:
                logger.warning(f"Session index error during session restore: {e}")

        # Refresh plots since restored session becomes active
        if self.gui.current_session and hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception as e:
                logger.warning(f"Plot refresh after session restore failed (non-fatal): {e}", exc_info=True)

    def undo(self):
        self.gui.current_dataset.exclude_session(self.session_id)
        self.gui.current_session = None
        self.gui.data_selection_widget.update(levels=("session",))
        # Refresh plots to clear session-dependent displays
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception as e:
                logger.warning(f"Plot refresh after session restore undo failed (non-fatal): {e}", exc_info=True)


class RestoreDatasetCommand(Command):
    """Restore an excluded dataset by ID."""

    def __init__(self, gui, dataset_id: str):
        self.command_name = "Restore Dataset"
        self.gui: MonstimGUI = gui
        self.dataset_id = dataset_id
        self.dataset_obj = None

    def execute(self):
        # Restore the dataset in the domain model first
        self.gui.current_experiment.restore_dataset(self.dataset_id)

        # Re-acquire the (now restored) dataset object from the active experiment's current datasets list
        try:
            self.dataset_obj = next(ds for ds in self.gui.current_experiment.datasets if ds.id == self.dataset_id)
        except StopIteration:
            self.dataset_obj = None

        # Set current_dataset explicitly to the restored object
        self.gui.current_dataset = self.dataset_obj

        # Update dataset list (do not touch sessions yet) so combo has restored entry
        self.gui.data_selection_widget.update(levels=("dataset",))

        if self.dataset_obj:
            try:
                dataset_index = self.gui.current_experiment.datasets.index(self.dataset_obj)
                # Block signals so we avoid triggering a redundant load (we'll do it manually below)
                self.gui.data_selection_widget.dataset_combo.blockSignals(True)
                self.gui.data_selection_widget.dataset_combo.setCurrentIndex(dataset_index)
                self.gui.data_selection_widget.dataset_combo.blockSignals(False)
            except ValueError as e:
                logger.warning(f"Dataset index error during dataset restore: {e}")

        # Now refresh the session list for this dataset
        self.gui.data_selection_widget.update(levels=("session",))

        # Choose a current session (first available) to keep internal state consistent with UI
        if self.gui.current_dataset and self.gui.current_dataset.sessions:
            self.gui.current_session = self.gui.current_dataset.sessions[0]
            # Reflect selection in session combo
            try:
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(0)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except Exception as e:
                logger.warning(f"Non-fatal: session combo update failed after dataset restore: {e}", exc_info=True)
        else:
            self.gui.current_session = None

        # Trigger downstream updates that normally occur via combo change handlers
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception as e:
                logger.warning(f"Plot widget refresh after dataset restore failed (non-fatal): {e}", exc_info=True)

    def undo(self):
        self.gui.current_experiment.exclude_dataset(self.dataset_id)
        self.gui.current_dataset = None
        self.gui.current_session = None
        self.gui.data_selection_widget.update(levels=("dataset", "session"))
        # Clear plots / dependent UI since selection is now empty
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception as e:
                logger.warning(f"Plot widget refresh after dataset undo failed (non-fatal): {e}", exc_info=True)


class InvertChannelPolarityCommand(Command):
    def __init__(self, gui, level: str, channel_indexes_to_invert: list[int]):
        self.command_name = "Invert Channel Polarity"
        self.gui: MonstimGUI = gui  # type: EMGAnalysisGUI
        self.channel_indexes_to_invert = channel_indexes_to_invert

        match level:
            case "experiment":
                self.level = self.gui.current_experiment
            case "dataset":
                self.level = self.gui.current_dataset
            case "session":
                self.level = self.gui.current_session
            case _:
                raise ValueError(f"Invalid level: {level}")

    def execute(self):
        for channel_index in self.channel_indexes_to_invert:
            self.level.invert_channel_polarity(channel_index)

    def undo(self):
        for channel_index in self.channel_indexes_to_invert:
            self.level.invert_channel_polarity(channel_index)


class SetLatencyWindowsCommand(Command):
    def __init__(self, gui, level: str, new_windows: list):
        self.command_name: str = "Set Latency Windows"
        self.gui: MonstimGUI = gui
        match level:
            case "experiment":
                self.level = self.gui.current_experiment
                self.sessions = [s for ds in self.level.datasets for s in ds.sessions]
            case "dataset":
                self.level = self.gui.current_dataset
                self.sessions = list(self.level.sessions)
            case "session":
                self.level = self.gui.current_session
                self.sessions = [self.level]
            case _:
                raise ValueError(f"Invalid level: {level}")
        self.new_windows = [copy.deepcopy(w) for w in new_windows]
        self.old_windows = {s.id: copy.deepcopy(s.annot.latency_windows) for s in self.sessions}

    def _apply(self, windows):
        _cancel_cache_warmup(self.gui)
        import copy

        from monstim_signals.io.repositories import SessionRepository

        calculation_changed = False
        for s in self.sessions:
            old_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            s.annot.latency_windows = [copy.deepcopy(w) for w in windows]
            new_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            changed = old_fingerprint != new_fingerprint
            calculation_changed |= changed
            if changed:
                s.invalidate_window_results()
            else:
                s.update_latency_window_parameters()
        SessionRepository.save_many(self.sessions)
        if calculation_changed and hasattr(self.level, "invalidate_aggregate_results"):
            self.level.invalidate_aggregate_results()

    def execute(self):
        self._apply(self.new_windows)

    def undo(self):
        _cancel_cache_warmup(self.gui)
        from monstim_signals.io.repositories import SessionRepository

        calculation_changed = False
        for s in self.sessions:
            windows = self.old_windows[s.id]
            old_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            s.annot.latency_windows = windows
            new_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            changed = old_fingerprint != new_fingerprint
            calculation_changed |= changed
            if changed:
                s.invalidate_window_results()
            else:
                s.update_latency_window_parameters()
        SessionRepository.save_many(self.sessions)
        if calculation_changed and hasattr(self.level, "invalidate_aggregate_results"):
            self.level.invalidate_aggregate_results()


class InsertSingleLatencyWindowCommand(Command):
    """Insert or replace a single latency window by name across hierarchy.

    This command merges a single window into existing configurations without
    replacing all windows. If a window with the same name exists, it's replaced;
    otherwise the window is appended.
    """

    def __init__(self, gui, level: str, window, replace_mode: bool = True):
        """
        Args:
            gui: The main GUI instance
            level: "experiment", "dataset", or "session"
            window: The LatencyWindow to insert/replace
            replace_mode: If True and window name exists, replace it. If False, append with unique name.
        """
        self.command_name: str = f"Insert Window '{window.name}'"
        self.gui: MonstimGUI = gui
        self.replace_mode = replace_mode

        match level:
            case "experiment":
                self.level = self.gui.current_experiment
                self.sessions = [s for ds in self.level.datasets for s in ds.sessions]
            case "dataset":
                self.level = self.gui.current_dataset
                self.sessions = list(self.level.sessions)
            case "session":
                self.level = self.gui.current_session
                self.sessions = [self.level]
            case _:
                raise ValueError(f"Invalid level: {level}")

        self.new_window = copy.deepcopy(window)
        # Store old windows for each session for undo
        self.old_windows = {s.id: copy.deepcopy(s.annot.latency_windows) for s in self.sessions}

    def _merge_window(self, existing_windows, new_window):
        """Merge a single window into existing windows, replacing by name if it exists."""
        result = []
        replaced = False

        for w in existing_windows:
            if w.name == new_window.name and self.replace_mode:
                result.append(copy.deepcopy(new_window))
                replaced = True
            else:
                result.append(copy.deepcopy(w))

        if not replaced:
            result.append(copy.deepcopy(new_window))

        return result

    def execute(self):
        _cancel_cache_warmup(self.gui)
        from monstim_signals.io.repositories import SessionRepository

        calculation_changed = False
        for s in self.sessions:
            old_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            s.annot.latency_windows = self._merge_window(s.annot.latency_windows, self.new_window)
            new_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            changed = old_fingerprint != new_fingerprint
            calculation_changed |= changed
            if changed:
                s.invalidate_window_results()
            else:
                s.update_latency_window_parameters()
        SessionRepository.save_many(self.sessions)
        if calculation_changed and hasattr(self.level, "invalidate_aggregate_results"):
            self.level.invalidate_aggregate_results()

    def undo(self):
        _cancel_cache_warmup(self.gui)
        from monstim_signals.io.repositories import SessionRepository

        calculation_changed = False
        for s in self.sessions:
            windows = self.old_windows[s.id]
            old_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            s.annot.latency_windows = windows
            new_fingerprint = tuple((w.name, tuple(w.start_times), tuple(w.durations)) for w in s.annot.latency_windows)
            changed = old_fingerprint != new_fingerprint
            calculation_changed |= changed
            if changed:
                s.invalidate_window_results()
            else:
                s.update_latency_window_parameters()
        SessionRepository.save_many(self.sessions)
        if calculation_changed and hasattr(self.level, "invalidate_aggregate_results"):
            self.level.invalidate_aggregate_results()


class ChangeChannelNamesCommand(Command):
    def __init__(self, gui, level: str, new_names: dict):
        self.command_name: str = "Change Channel Names"
        self.gui: MonstimGUI = gui
        self.new_names = copy.deepcopy(new_names)

        match level:
            case "experiment":
                self.level = self.gui.current_experiment
            case "dataset":
                self.level = self.gui.current_dataset
            case "session":
                self.level = self.gui.current_session
            case _:
                raise ValueError(f"Invalid level: {level}")

        # Store old channel names for undo - create reverse mapping
        self.old_names = {new_name: old_name for old_name, new_name in new_names.items()}

    def execute(self):
        self.level.rename_channels(self.new_names)

    def undo(self):
        self.level.rename_channels(self.old_names)


class BulkRecordingExclusionCommand(Command):
    """Apply bulk recording exclusions/inclusions across multiple sessions."""

    def __init__(self, gui, changes: list):
        """
        Initialize bulk recording exclusion command.

        Args:
            gui: The main GUI instance
            changes: list of dicts with format:
                [
                    {
                        'session': session_object,
                        'changes': [
                            {'recording_id': str, 'exclude': bool},
                            ...
                        ]
                    },
                    ...
                ]
        """
        self.command_name = "Bulk Recording Exclusion"
        self.gui: MonstimGUI = gui
        self.changes = changes
        self._previous_curation: dict[tuple[int, str], dict | None] = {}
        self._state_before_execute: dict[int, tuple[object, list[str], dict]] = {}

    def execute(self):
        """Apply all changes and persist each affected session only once."""
        if not self._supports_batched_persistence():
            self._execute_legacy()
            return
        changed_sessions = []
        self._state_before_execute = {}
        try:
            for session_change in self.changes:
                session = session_change["session"]
                session_key = id(session)
                if session_key not in self._state_before_execute:
                    self._state_before_execute[session_key] = (
                        session,
                        list(session.annot.excluded_recordings),
                        copy.deepcopy(session.annot.recording_curation),
                    )
                excluded = set(session.annot.excluded_recordings)
                for change in session_change["changes"]:
                    recording_id = change["recording_id"]
                    should_exclude = change["exclude"]
                    if should_exclude:
                        excluded.add(recording_id)
                    else:
                        excluded.discard(recording_id)
                    curation = change.get("curation")
                    if curation is not None:
                        key = (session_key, recording_id)
                        if key not in self._previous_curation:
                            previous = session.annot.recording_curation.get(recording_id)
                            self._previous_curation[key] = dict(previous) if previous is not None else None
                        session.annot.recording_curation[recording_id] = curation
                session.annot.excluded_recordings = sorted(excluded)
                session.invalidate_selection_results()
                changed_sessions.append(session)

            # Let persistence errors reach CommandInvoker and the editor.  Catching
            # them here previously made a failed Apply look successful: the invoker
            # added the command to undo history and the dialog closed regardless.
            self._save_sessions(changed_sessions)
        except Exception:
            # Keep the live model aligned with the dialog if the save failed, so
            # the user can safely retry instead of needing a second Apply to
            # reconcile an unpersisted in-memory mutation.
            for session, excluded, curation in self._state_before_execute.values():
                session.annot.excluded_recordings = excluded
                session.annot.recording_curation = curation
                session.invalidate_selection_results()
            raise
        self.gui.data_selection_widget.sync_combo_selections()

    def _supports_batched_persistence(self) -> bool:
        return all(isinstance(change["session"].annot.excluded_recordings, list) for change in self.changes)

    def _execute_legacy(self) -> None:
        """Keep this command usable for lightweight domain doubles and old sessions."""
        for session_change in self.changes:
            session = session_change["session"]
            for change in session_change["changes"]:
                if change["exclude"]:
                    session.exclude_recording(change["recording_id"])
                else:
                    session.restore_recording(change["recording_id"])
        self.gui.data_selection_widget.sync_combo_selections()

    @staticmethod
    def _save_sessions(sessions):
        """Batch JSON writes and their matching catalog updates."""
        if not sessions:
            return
        from monstim_signals.io.repositories import SessionRepository

        SessionRepository.save_many(sessions)

    def undo(self):
        """Reverse all changes with the same batched persistence path."""
        if not self._supports_batched_persistence():
            self._undo_legacy()
            return
        changed_sessions = []
        for session_change in reversed(self.changes):
            session = session_change["session"]
            excluded = set(session.annot.excluded_recordings)
            for change in reversed(session_change["changes"]):
                recording_id = change["recording_id"]
                should_exclude = change["exclude"]
                if should_exclude:
                    excluded.discard(recording_id)
                else:
                    excluded.add(recording_id)
                if change.get("curation") is not None:
                    previous = self._previous_curation.get((id(session), recording_id))
                    if previous is None:
                        session.annot.recording_curation.pop(recording_id, None)
                    else:
                        session.annot.recording_curation[recording_id] = previous
            session.annot.excluded_recordings = sorted(excluded)
            session.invalidate_selection_results()
            changed_sessions.append(session)

        self._save_sessions(changed_sessions)
        self.gui.data_selection_widget.sync_combo_selections()

    def _undo_legacy(self) -> None:
        for session_change in reversed(self.changes):
            session = session_change["session"]
            for change in reversed(session_change["changes"]):
                if change["exclude"]:
                    session.restore_recording(change["recording_id"])
                else:
                    session.exclude_recording(change["recording_id"])
        self.gui.data_selection_widget.sync_combo_selections()


# Data Curation Commands
class CreateExperimentCommand(Command):
    def __init__(self, gui, exp_name: str):
        self.command_name = f"Create Experiment '{exp_name}'"
        self.gui: MonstimGUI = gui
        self.exp_name = exp_name

    def execute(self):
        """Create the experiment immediately."""
        try:
            self.gui.data_manager.create_experiment(self.exp_name)
            _refresh_data_views(self.gui, self.exp_name)
        except Exception as e:
            logger.exception(f"Failed to create experiment: {e!s}")
            raise Exception(f"Failed to create experiment: {e!s}") from e

    def undo(self):
        """Delete the created experiment."""
        try:
            self.gui.data_manager.delete_experiment_by_id(self.exp_name)
            _refresh_data_views(self.gui)
        except Exception as e:
            logger.exception(f"Failed to undo experiment creation: {e!s}")
            raise Exception(f"Failed to undo experiment creation: {e!s}") from e

    def get_description(self) -> str:
        return f"Created experiment '{self.exp_name}'"


class MoveDatasetCommand(Command):
    def __init__(self, gui, dataset_id: str, dataset_name: str, from_exp: str, to_exp: str):
        self.command_name = f"Move '{dataset_name}' from '{from_exp}' to '{to_exp}'"
        self.gui: MonstimGUI = gui
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.from_exp = from_exp
        self.to_exp = to_exp

    def execute(self):
        """Move the dataset immediately."""
        try:
            self.gui.data_manager.move_dataset(self.dataset_id, self.dataset_name, self.from_exp, self.to_exp)
            _refresh_data_views(self.gui, self.from_exp, self.to_exp)
        except Exception as e:
            logger.exception(f"Failed to move dataset: {e!s}")
            raise Exception(f"Failed to move dataset: {e!s}") from e

    def undo(self):
        """Move the dataset back to original location."""
        try:
            self.gui.data_manager.move_dataset(self.dataset_id, self.dataset_name, self.to_exp, self.from_exp)
            _refresh_data_views(self.gui, self.from_exp, self.to_exp)
        except Exception as e:
            logger.exception(f"Failed to undo dataset move: {e!s}")
            raise Exception(f"Failed to undo dataset move: {e!s}") from e

    def get_description(self) -> str:
        return f"Moved dataset '{self.dataset_name}' from '{self.from_exp}' to '{self.to_exp}'"


class MoveDatasetsCommand(Command):
    """Batched move of multiple datasets executed as a single undoable command."""

    def __init__(self, gui, moves: list[tuple]):
        """
        moves: list of tuples (dataset_id, dataset_name, from_exp, to_exp)
        """
        self.gui: MonstimGUI = gui
        self.moves = list(moves)
        self.command_name = f"Move {len(self.moves)} datasets"
        # Will record only the moves that actually succeeded during execute()
        self._succeeded = []

    def execute(self):
        """Execute all moves sequentially. Record successes for undo."""
        try:
            self._succeeded.clear()
            self.gui.data_manager.close_all_data()

            for ds_id, ds_name, from_exp, to_exp in self.moves:
                try:
                    self.gui.data_manager.move_dataset(
                        ds_id,
                        ds_name,
                        from_exp,
                        to_exp,
                        close_open_data=False,  # Already closed all data at start
                    )
                    self._succeeded.append((ds_id, ds_name, from_exp, to_exp))
                except Exception as e:
                    logger.error(f"Failed to move dataset '{ds_name}' from '{from_exp}' to '{to_exp}': {e}")

                if len(self._succeeded) % 10 == 0:
                    QApplication.processEvents()

                logger.debug(f"Processed {len(self._succeeded)} dataset moves.")

            affected = {exp_id for _, _, from_exp, to_exp in self._succeeded for exp_id in (from_exp, to_exp)}
            _refresh_data_views(self.gui, *affected)

        except Exception as e:
            logger.exception(f"Failed to execute batched dataset moves: {e!s}")
            raise Exception(f"Failed to execute batched dataset moves: {e!s}") from e

    def undo(self):
        """Undo by moving succeeded items back in reverse order."""
        try:
            self.gui.data_manager.close_all_data()

            for ds_id, ds_name, from_exp, to_exp in reversed(self._succeeded):
                try:
                    # Move back from to_exp -> from_exp
                    self.gui.data_manager.move_dataset(
                        ds_id,
                        ds_name,
                        to_exp,
                        from_exp,
                        close_open_data=False,
                    )
                except Exception as e:
                    logger.error(f"Failed to undo move of dataset '{ds_name}' from '{to_exp}' back to '{from_exp}': {e}")

                if len(self._succeeded) % 10 == 0:
                    QApplication.processEvents()

            affected = {exp_id for _, _, from_exp, to_exp in self._succeeded for exp_id in (from_exp, to_exp)}
            _refresh_data_views(self.gui, *affected)

        except Exception as e:
            logger.exception(f"Failed to undo batched dataset moves: {e!s}")
            raise Exception(f"Failed to undo batched dataset moves: {e!s}") from e

    def get_description(self) -> str:
        return f"Moved {len(self._succeeded)} dataset(s) in batch"


class CopyDatasetCommand(Command):
    def __init__(self, gui, dataset_id: str, dataset_name: str, from_exp: str, to_exp: str, new_name: str | None = None):
        self.command_name = f"Copy '{dataset_name}' from '{from_exp}' to '{to_exp}'"
        self.gui: MonstimGUI = gui
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.from_exp = from_exp
        self.to_exp = to_exp
        self.new_name = new_name  # Optional new name for the copied dataset
        self.copied_folder_name = None  # Will be set after execution

    def execute(self):
        """Copy the dataset immediately."""
        try:
            # Store the original target experiment datasets before copy
            from pathlib import Path

            to_exp_path = Path(self.gui.expts_dict[self.to_exp])
            original_datasets = {f.name for f in to_exp_path.iterdir() if f.is_dir()}

            self.gui.data_manager.copy_dataset(self.dataset_id, self.dataset_name, self.from_exp, self.to_exp, self.new_name)

            self.finalize_copy(original_datasets)
        except Exception as e:
            logger.exception(f"Failed to copy dataset: {e!s}")
            raise Exception(f"Failed to copy dataset: {e!s}") from e

    def finalize_copy(self, original_datasets=None):
        """Finish command bookkeeping and refresh UI after an async copy."""
        from pathlib import Path

        to_exp_path = Path(self.gui.expts_dict[self.to_exp])
        if original_datasets is None:
            original_datasets = set()

        # Find the new dataset folder name (might have _copy suffix)
        new_datasets = {f.name for f in to_exp_path.iterdir() if f.is_dir()}
        added_datasets = new_datasets - original_datasets
        self.copied_folder_name = next(iter(added_datasets), self.new_name or self.dataset_id)

        _refresh_data_views(self.gui, self.to_exp)

    def undo(self):
        """Delete the copied dataset."""
        try:
            if self.copied_folder_name:
                self.gui.data_manager.delete_dataset(self.copied_folder_name, self.copied_folder_name, self.to_exp)
                _refresh_data_views(self.gui, self.to_exp)
        except Exception as e:
            logger.exception(f"Failed to undo dataset copy: {e!s}")
            raise Exception(f"Failed to undo dataset copy: {e!s}") from e

    def get_description(self) -> str:
        if self.from_exp == self.to_exp:
            action = "Duplicated"
            location = f"within '{self.from_exp}'"
            if self.new_name:
                location += f" as '{self.new_name}'"
        else:
            action = "Copied"
            location = f"from '{self.from_exp}' to '{self.to_exp}'"
        return f"{action} dataset '{self.dataset_name}' {location}"


class DeleteExperimentCommand(Command):
    def __init__(self, gui, exp_name: str):
        self.command_name = f"Delete Experiment '{exp_name}'"
        self.gui = gui
        self.exp_name = exp_name
        self.backup_path = None  # Will store backup information if needed

    def execute(self):
        """Delete the experiment immediately (with user confirmation already handled)."""
        try:
            # For now, we'll use the existing delete method from data manager
            # Note: This is irreversible, so undo will show a warning
            self.gui.data_manager.delete_experiment_by_id(self.exp_name)
            _refresh_data_views(self.gui)
        except Exception as e:
            logger.exception(f"Failed to delete experiment: {e!s}")
            raise Exception(f"Failed to delete experiment: {e!s}") from e

    def undo(self):
        """Cannot undo experiment deletion - show warning."""
        QMessageBox.warning(
            self.gui,
            "Cannot Undo Deletion",
            f"Experiment '{self.exp_name}' was permanently deleted and cannot be restored.\n\n"
            "Deletion operations are irreversible for safety reasons.",
        )

    def get_description(self) -> str:
        return f"Deleted experiment '{self.exp_name}' (irreversible)"


class RenameExperimentCommand(Command):
    def __init__(self, gui, old_name: str, new_name: str):
        self.command_name = f"Rename Experiment '{old_name}' to '{new_name}'"
        self.gui = gui
        self.old_name = old_name
        self.new_name = new_name

    def execute(self):
        """Rename the experiment immediately."""
        # Let exceptions from data_manager propagate with their original messages
        self.gui.data_manager.rename_experiment_by_id(self.old_name, self.new_name)
        _refresh_data_views(self.gui, self.new_name)

    def undo(self):
        """Rename back to original name."""
        try:
            self.gui.data_manager.rename_experiment_by_id(self.new_name, self.old_name)
            _refresh_data_views(self.gui, self.old_name)
        except Exception as e:
            logger.exception(f"Failed to undo experiment rename: {e!s}")
            raise Exception(f"Failed to undo experiment rename: {e!s}") from e

    def get_description(self) -> str:
        return f"Renamed experiment '{self.old_name}' to '{self.new_name}'"


class DeleteDatasetCommand(Command):
    """Delete a dataset from an experiment. This operation is irreversible; undo will show a warning."""

    def __init__(self, gui, dataset_id: str, dataset_name: str, exp_id: str):
        self.command_name = f"Delete Dataset '{dataset_name}' in '{exp_id}'"
        self.gui = gui
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.exp_id = exp_id

    def execute(self):
        try:
            self.gui.data_manager.delete_dataset(self.dataset_id, self.dataset_name, self.exp_id)
            _refresh_data_views(self.gui, self.exp_id)
        except Exception as e:
            logger.exception(f"Failed to delete dataset: {e!s}")
            raise Exception(f"Failed to delete dataset: {e!s}") from e

    def undo(self):
        QMessageBox.warning(
            self.gui,
            "Cannot Undo Deletion",
            f"Dataset '{self.dataset_name}' in experiment '{self.exp_id}' was permanently deleted and cannot be restored.\n\n"
            "Deletion operations are irreversible for safety reasons.",
        )

    def get_description(self) -> str:
        return f"Deleted dataset '{self.dataset_name}' in '{self.exp_id}' (irreversible)"


class ToggleDatasetInclusionCommand(Command):
    """Include or exclude a dataset at the experiment level by updating ExperimentAnnot.excluded_datasets."""

    def __init__(self, gui, exp_id: str, dataset_id: str, exclude: bool):
        action = "Exclude" if exclude else "Include"
        self.command_name = f"{action} Dataset '{dataset_id}' in '{exp_id}'"
        self.gui = gui
        self.exp_id = exp_id
        self.dataset_id = dataset_id
        self.exclude = exclude
        self._prev_was_excluded = None

    def _apply(self, set_excluded: bool):
        from pathlib import Path

        from monstim_signals.io.repositories import ExperimentRepository

        exp_path = Path(self.gui.expts_dict[self.exp_id])
        repo = ExperimentRepository(exp_path)
        # Load annot minimally through repo.load or by reading file
        # To keep it lightweight, read annot JSON and write back
        import json
        from dataclasses import asdict

        from monstim_signals.core import ExperimentAnnot

        try:
            if repo.expt_js.exists():
                annot_dict = json.loads(repo.expt_js.read_text())
                annot = ExperimentAnnot.from_dict(annot_dict)
            else:
                annot = ExperimentAnnot.create_empty()

            if self._prev_was_excluded is None:
                self._prev_was_excluded = self.dataset_id in annot.excluded_datasets

            if set_excluded:
                if self.dataset_id not in annot.excluded_datasets:
                    annot.excluded_datasets.append(self.dataset_id)
            else:
                if self.dataset_id in annot.excluded_datasets:
                    annot.excluded_datasets = [d for d in annot.excluded_datasets if d != self.dataset_id]

            repo.expt_js.write_text(json.dumps(asdict(annot), indent=2))
        except Exception as e:
            logger.exception(f"Failed to update dataset inclusion: {e!s}")
            raise Exception(f"Failed to update dataset inclusion: {e!s}") from e

        _refresh_data_views(self.gui, self.exp_id)

    def execute(self):
        self._apply(self.exclude)

    def undo(self):
        # Revert to original exclusion state
        if self._prev_was_excluded is None:
            # If somehow unknown, just toggle opposite
            self._apply(not self.exclude)
        else:
            self._apply(self._prev_was_excluded)

    def get_description(self) -> str:
        action = "Excluded" if self.exclude else "Included"
        return f"{action} dataset '{self.dataset_id}' in '{self.exp_id}'"


class ToggleCompletionStatusCommand(Command):
    """Toggle completion status for experiments, datasets, or sessions.

    Completion status is a user-facing organizational flag that allows
    marking data as complete for analysis. This command uses repository-based
    persistence to ensure undo/redo works reliably across selection changes.
    """

    def __init__(self, gui, level: str, target_object, *, experiment_id: str | None = None, new_status: bool | None = None, dataset_path=None):
        """
        Args:
            gui: The main GUI instance
            level: \"experiment\", \"dataset\", or \"session\"
            target_object: The domain object to toggle (Experiment, Dataset, or Session)
        """
        self.gui = gui
        self.level = level
        self.target_id = target_object.id
        self.old_status = getattr(target_object, "is_completed", False)
        self.new_status = not self.old_status if new_status is None else new_status
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None

        # Store hierarchy IDs for reliable lookup from disk
        if level == "experiment":
            self.experiment_id = target_object.id
            self.dataset_id = None
        elif level == "dataset":
            # Get parent experiment ID from current context
            self.experiment_id = experiment_id or (self.gui.current_experiment.id if self.gui.current_experiment else None)
            self.dataset_id = target_object.id
            if not self.experiment_id:
                raise ValueError("Cannot toggle dataset completion status: no parent experiment in context")
        elif level == "session":
            # Get parent experiment and dataset IDs from current context
            self.experiment_id = self.gui.current_experiment.id if self.gui.current_experiment else None
            self.dataset_id = self.gui.current_dataset.id if self.gui.current_dataset else None
            if not self.experiment_id:
                raise ValueError("Cannot toggle session completion status: no parent experiment in context")
            if not self.dataset_id:
                raise ValueError("Cannot toggle session completion status: no parent dataset in context")
        else:
            raise ValueError(f"Invalid level for completion status toggle: {level}")

        obj_name = getattr(target_object, "id", "Unknown")
        action = "Complete" if self.new_status else "Incomplete"
        self.command_name = f"Mark {level.title()} '{obj_name}' as {action}"

    def _apply_status(self, status: bool):
        """Apply completion status by directly modifying annotation JSON files."""
        from dataclasses import asdict
        from pathlib import Path

        from monstim_signals.core import DatasetAnnot, ExperimentAnnot, SessionAnnot

        try:
            match self.level:
                case "experiment":
                    if not self.experiment_id or self.experiment_id not in self.gui.expts_dict:
                        logger.error(f"Experiment '{self.experiment_id}' not found in expts_dict")
                        return
                    exp_path = Path(self.gui.expts_dict[self.experiment_id])
                    annot_file = exp_path / "experiment.annot.json"

                    if annot_file.exists():
                        annot_dict = json.loads(annot_file.read_text())
                        annot = ExperimentAnnot.from_dict(annot_dict)
                    else:
                        annot = ExperimentAnnot.create_empty()

                    annot.is_completed = status
                    annot_file.write_text(json.dumps(asdict(annot), indent=2))

                    # Also update the in-memory object if it's currently loaded so UI updates immediately
                    try:
                        if (
                            hasattr(self.gui, "current_experiment")
                            and self.gui.current_experiment
                            and self.gui.current_experiment.id == self.experiment_id
                        ):
                            self.gui.current_experiment.is_completed = status
                    except Exception:
                        logger.exception(
                            f"Failed to update in-memory experiment object for experiment '{self.experiment_id}'",
                            exc_info=True,
                        )

                case "dataset":
                    if not self.experiment_id or self.experiment_id not in self.gui.expts_dict:
                        logger.error(f"Parent experiment '{self.experiment_id}' not found")
                        return
                    if not self.dataset_id:
                        logger.error("Dataset ID is missing")
                        return
                    exp_path = Path(self.gui.expts_dict[self.experiment_id])
                    dataset_path = self.dataset_path or exp_path / self.dataset_id
                    if not dataset_path.exists():
                        logger.error(f"Dataset path '{dataset_path}' not found")
                        return
                    annot_file = dataset_path / "dataset.annot.json"

                    if annot_file.exists():
                        annot_dict = json.loads(annot_file.read_text())
                        annot = DatasetAnnot.from_dict(annot_dict)
                    else:
                        annot = DatasetAnnot.create_empty()

                    annot.is_completed = status
                    annot_file.write_text(json.dumps(asdict(annot), indent=2))

                    from monstim_signals.io.experiment_catalog import refresh_dataset_annotation

                    refresh_dataset_annotation(dataset_path)

                    # Update in-memory dataset object if present
                    try:
                        if (
                            hasattr(self.gui, "current_experiment")
                            and self.gui.current_experiment
                            and getattr(self.gui.current_experiment, "datasets", None)
                        ):
                            for ds in self.gui.current_experiment.datasets:
                                if getattr(ds, "id", None) == self.dataset_id:
                                    ds.is_completed = status
                                    break
                    except Exception:
                        logger.exception(f"Failed to update in-memory dataset object for dataset '{self.dataset_id}'", exc_info=True)

                case "session":
                    if not self.experiment_id or self.experiment_id not in self.gui.expts_dict:
                        logger.error(f"Parent experiment '{self.experiment_id}' not found")
                        return
                    if not self.dataset_id:
                        logger.error("Parent dataset ID is missing")
                        return
                    exp_path = Path(self.gui.expts_dict[self.experiment_id])
                    dataset_path = exp_path / self.dataset_id
                    if not dataset_path.exists():
                        logger.error(f"Parent dataset path '{dataset_path}' not found")
                        return
                    session_path = dataset_path / self.target_id
                    if not session_path.exists():
                        logger.error(f"Session path '{session_path}' not found")
                        return
                    annot_file = session_path / "session.annot.json"

                    if annot_file.exists():
                        annot_dict = json.loads(annot_file.read_text())
                        annot = SessionAnnot.from_dict(annot_dict)
                    else:
                        annot = SessionAnnot.create_empty()

                    annot.is_completed = status
                    annot_file.write_text(json.dumps(asdict(annot), indent=2))

                    from monstim_signals.io.experiment_catalog import refresh_session_annotation

                    refresh_session_annotation(session_path)

                    # Update in-memory session object if present
                    try:
                        if hasattr(self.gui, "current_dataset") and self.gui.current_dataset and getattr(self.gui.current_dataset, "sessions", None):
                            for s in self.gui.current_dataset.sessions:
                                if getattr(s, "id", None) == self.target_id:
                                    s.is_completed = status
                                    break
                    except Exception:
                        logger.exception(f"Failed to update in-memory session object for session '{self.target_id}'", exc_info=True)

                case _:
                    logger.error(f"Unknown level '{self.level}' for completion status toggle")
                    return

            # Refresh UI if the affected object is currently visible
            if hasattr(self.gui, "data_selection_widget"):
                self.gui.data_selection_widget.update_completion_status(self.level)
                self.gui.data_selection_widget.update_all_completion_statuses()

        except Exception as e:
            logger.exception(f"Failed to apply completion status: {e}", exc_info=True)

    def execute(self):
        """Toggle completion status to new value."""
        self._apply_status(self.new_status)

    def undo(self):
        """Restore previous completion status."""
        self._apply_status(self.old_status)

    def get_description(self) -> str:
        action = "completed" if self.new_status else "marked incomplete"
        return f"Marked {self.level} '{self.target_id}' as {action}"


class EditDatasetMetadataCommand(Command):
    """Command to edit dataset metadata (date, animal ID, condition) with optional folder rename."""

    def __init__(
        self,
        gui,
        dataset,
        old_date: str | None,
        new_date: str | None,
        old_animal_id: str | None,
        new_animal_id: str | None,
        old_condition: str | None,
        new_condition: str | None,
        old_folder_name: str | None = None,
        new_folder_name: str | None = None,
    ):
        """Initialize the edit metadata command.

        Args:
            gui: The main GUI instance
            dataset: The Dataset object to modify
            old_date: Previous date string (YYYY-MM-DD format)
            new_date: New date string (YYYY-MM-DD format)
            old_animal_id: Previous animal ID
            new_animal_id: New animal ID
            old_condition: Previous condition
            new_condition: New condition
            old_folder_name: Original folder name (if renaming)
            new_folder_name: New folder name (if renaming)
        """
        self.gui = gui
        self.dataset = dataset
        self.old_date = old_date
        self.new_date = new_date
        self.old_animal_id = old_animal_id
        self.new_animal_id = new_animal_id
        self.old_condition = old_condition
        self.new_condition = new_condition
        self.old_folder_name = old_folder_name
        self.new_folder_name = new_folder_name

        # Build command name
        parts = []
        if old_date != new_date:
            parts.append(f"date: {old_date} → {new_date}")
        if old_animal_id != new_animal_id:
            parts.append(f"ID: {old_animal_id} → {new_animal_id}")
        if old_condition != new_condition:
            parts.append(f"condition: {old_condition} → {new_condition}")

        if parts:
            self.command_name = f"Edit Dataset Metadata ({', '.join(parts)})"
        else:
            self.command_name = "Edit Dataset Metadata"

    def _apply_metadata(self, date: str | None, animal_id: str | None, condition: str | None, folder_name: str | None):
        """Apply metadata changes and optionally rename folder.

        Args:
            date: Date to set
            animal_id: Animal ID to set
            condition: Condition to set
            folder_name: Target folder name (None if no rename needed)
        """
        import errno

        # Rename folder if needed (before mutating metadata)
        # This ensures atomicity - if rename fails, metadata hasn't changed yet
        old_dataset_id = self.dataset.id
        if folder_name and self.dataset.repo:
            current_folder = self.dataset.repo.folder  # Already a Path object
            if current_folder.name != folder_name:
                new_folder_path = current_folder.parent / folder_name

                # Check if target exists
                if new_folder_path.exists():
                    raise FileExistsError(f"Target folder already exists: {new_folder_path}")

                # Explicitly close any open resources (e.g. HDF5 files) before renaming
                if hasattr(self.dataset, "close"):
                    try:
                        self.dataset.close()
                        logger.debug("Closed dataset before folder rename to release file handles.")
                    except Exception as close_err:
                        # Proceed with rename even if close fails; behavior is no worse than before
                        logger.warning(
                            "Failed to close dataset cleanly before rename: %s",
                            close_err,
                            exc_info=True,
                        )
                try:
                    # Use repository rename with retry logic
                    self.dataset.repo.rename(new_folder_path, dataset=self.dataset)
                    logger.info(f"Renamed dataset folder: {current_folder.name} → {folder_name}")
                    from monstim_gui.core.application_state import app_state

                    experiment_id = self.gui.current_experiment.id if self.gui.current_experiment else None
                    app_state.migrate_renamed_selection(
                        "dataset",
                        old_dataset_id,
                        self.dataset.id,
                        experiment_id=experiment_id,
                    )
                except OSError as e:
                    if getattr(e, "errno", None) == errno.EACCES:
                        raise OSError(f"Cannot rename folder - it is in use. Please close any programs accessing: {current_folder}") from e
                    raise

        # Update annotation after successful rename (or if no rename needed)
        self.dataset.annot.date = date
        self.dataset.annot.animal_id = animal_id
        self.dataset.annot.condition = condition

        # Save annotation changes
        if self.dataset.repo:
            self.dataset.repo.save(self.dataset)
            logger.info(f"Updated metadata for dataset '{self.dataset.id}': date={date}, animal_id={animal_id}, condition={condition}")

    def execute(self):
        """Apply the new metadata and folder name."""
        try:
            self._apply_metadata(self.new_date, self.new_animal_id, self.new_condition, self.new_folder_name)

            if hasattr(self.gui, "data_selection_widget"):
                self.gui.data_selection_widget.update(levels=("dataset", "session"))
            _refresh_data_views(self.gui)

        except Exception as e:
            logger.exception(f"Failed to apply dataset metadata changes: {e}", exc_info=True)
            raise

    def undo(self):
        """Revert to the old metadata and folder name."""
        try:
            self._apply_metadata(self.old_date, self.old_animal_id, self.old_condition, self.old_folder_name)

            if hasattr(self.gui, "data_selection_widget"):
                self.gui.data_selection_widget.update(levels=("dataset", "session"))
            _refresh_data_views(self.gui)

        except Exception as e:
            logger.exception(f"Failed to undo dataset metadata changes: {e!s}", exc_info=True)
            raise

    def get_description(self) -> str:
        """Return a human-readable description of this command."""
        parts = []
        if self.old_date != self.new_date:
            parts.append(f"date to {self.new_date}")
        if self.old_animal_id != self.new_animal_id:
            parts.append(f"ID to {self.new_animal_id}")
        if self.old_condition != self.new_condition:
            parts.append(f"condition to {self.new_condition}")

        if parts:
            return f"Changed dataset {', '.join(parts)}"
        return "Modified dataset metadata"
