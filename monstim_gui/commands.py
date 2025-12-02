import abc
import copy
import logging
from collections import deque
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI


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


class CommandInvoker:
    def __init__(self, parent: "MonstimGUI"):
        self.parent = parent  # type: MonstimGUI
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
            logging.warning("Non-fatal: Command history trimming failed.", exc_info=True)
            pass
        self.redo_stack.clear()
        self.parent.menu_bar.update_undo_redo_labels()
        # --> Set self.parent._has_unsaved_changes to True if needed <--
        # Always refresh notice icons after a command executes so diagnostics stay in sync with domain state.
        try:
            self.parent.data_selection_widget.refresh_notice_icons()
        except Exception as e:
            logging.warning("Non-fatal: refresh_notice_icons failed after execute: %s", e, exc_info=True)

    def undo(self):
        if self.history:
            command = self.history.pop()
            command.undo()
            self.redo_stack.append(command)
            self.parent.menu_bar.update_undo_redo_labels()
            # --> Set self.parent._has_unsaved_changes to True if needed <--
            try:
                self.parent.data_selection_widget.refresh_notice_icons()
            except Exception as e:
                logging.warning("Non-fatal: refresh_notice_icons failed after undo: %s", e, exc_info=True)

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.history.append(command)
            self.parent.menu_bar.update_undo_redo_labels()
            # --> Set self.parent._has_unsaved_changes to True if needed <--
            try:
                self.parent.data_selection_widget.refresh_notice_icons()
            except Exception as e:
                logging.warning("Non-fatal: refresh_notice_icons failed after redo: %s", e, exc_info=True)

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
        self.gui: "MonstimGUI" = gui
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
        self.gui: "MonstimGUI" = gui
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
        self.gui: "MonstimGUI" = gui
        self.removed_session = None
        self.session_id = None
        self.idx = None
        self.previous_dataset = None

    def execute(self):
        self.removed_session = self.gui.current_session
        self.session_id = self.gui.current_session.id
        self.idx = self.gui.current_dataset.sessions.index(self.gui.current_session)
        self.previous_dataset = self.gui.current_dataset  # Preserve dataset selection

        self.gui.current_dataset.exclude_session(self.session_id)
        # Determine new selection: try next session at same index, else previous.
        new_current = None
        remaining_sessions = self.gui.current_dataset.sessions
        if remaining_sessions:
            if self.idx < len(remaining_sessions):
                new_current = remaining_sessions[self.idx]
            else:
                new_current = remaining_sessions[-1]
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
                # Session not found in the list (may have been removed); safe to ignore.
                # Session not found in the list (may have been removed); safe to ignore.
                pass
        else:
            # No sessions left; clear plots
            if hasattr(self.gui, "plot_widget"):
                try:
                    self.gui.plot_widget.on_data_selection_changed()
                except Exception:
                    logging.warning(
                        "Plot refresh after session exclusion (no sessions left) failed (non-fatal).", exc_info=True
                    )

        # Always refresh plots after exclusion to reflect new session
        if self.gui.current_session and hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception:
                logging.warning("Plot refresh after session exclusion failed (non-fatal).", exc_info=True)

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
                logging.warning("Plot refresh after session exclusion undo failed (non-fatal).", exc_info=True)


class ExcludeDatasetCommand(Command):
    """Exclude the currently selected dataset."""

    def __init__(self, gui):
        self.command_name = "Exclude Dataset"
        self.gui: "MonstimGUI" = gui
        self.removed_dataset = None
        self.dataset_id = None
        self.idx = None
        self.previous_experiment = None

    def execute(self):
        # Capture state prior to exclusion
        self.removed_dataset = self.gui.current_dataset
        self.dataset_id = self.gui.current_dataset.id if self.gui.current_dataset else None
        self.idx = (
            self.gui.current_experiment.datasets.index(self.gui.current_dataset)
            if self.gui.current_dataset in self.gui.current_experiment.datasets
            else None
        )
        self.previous_experiment = self.gui.current_experiment  # Preserve experiment selection

        # Perform exclusion in domain
        if self.dataset_id is not None:
            self.gui.current_experiment.exclude_dataset(self.dataset_id)

        # Determine next dataset selection (next at same index if available, else previous, else none)
        remaining = self.gui.current_experiment.datasets
        new_dataset = None
        if remaining:
            if self.idx is not None and self.idx < len(remaining):
                new_dataset = remaining[self.idx]
            else:
                new_dataset = remaining[-1]

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
                logging.warning(f"Index error during dataset exclusion execute: {e}")
        self.gui.data_selection_widget.update(levels=("session",))
        if self.gui.current_session:
            try:
                self.gui.data_selection_widget.session_combo.blockSignals(True)
                self.gui.data_selection_widget.session_combo.setCurrentIndex(0)
                self.gui.data_selection_widget.session_combo.blockSignals(False)
            except Exception as e:
                logging.warning(f"Non-fatal: session combo update failed after dataset exclusion: {e}", exc_info=True)

        # Trigger downstream updates (plots etc.)
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception:
                logging.debug("Plot refresh after dataset exclusion failed (non-fatal).", exc_info=True)

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
                logging.warning(f"Index error during dataset exclusion undo: {e}")

        # Update sessions and select first session (consistent with RestoreDatasetCommand)
        self.gui.data_selection_widget.update(levels=("session",))
        if restored:
            sessions_attr = getattr(restored, "sessions", None)
            if isinstance(sessions_attr, (list, tuple)) and sessions_attr:
                try:
                    self.gui.current_session = sessions_attr[0]
                except Exception as e:
                    self.gui.current_session = None
                    logging.warning(f"Non-fatal: session selection failed after dataset exclusion undo: {e}", exc_info=True)
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
                logging.warning(f"Non-fatal: session combo update failed after dataset exclusion undo: {e}", exc_info=True)

        # Trigger downstream updates
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception:
                logging.debug("Plot refresh after dataset exclusion undo failed (non-fatal).", exc_info=True)


class RestoreSessionCommand(Command):
    """Restore an excluded session by ID."""

    def __init__(self, gui, session_id: str):
        self.command_name = "Restore Session"
        self.gui: "MonstimGUI" = gui
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
                logging.warning(f"Session index error during session restore: {e}")

        # Refresh plots since restored session becomes active
        if self.gui.current_session and hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception as e:
                logging.warning(f"Plot refresh after session restore failed (non-fatal): {e}", exc_info=True)

    def undo(self):
        self.gui.current_dataset.exclude_session(self.session_id)
        self.gui.current_session = None
        self.gui.data_selection_widget.update(levels=("session",))
        # Refresh plots to clear session-dependent displays
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception as e:
                logging.warning(f"Plot refresh after session restore undo failed (non-fatal): {e}", exc_info=True)


class RestoreDatasetCommand(Command):
    """Restore an excluded dataset by ID."""

    def __init__(self, gui, dataset_id: str):
        self.command_name = "Restore Dataset"
        self.gui: "MonstimGUI" = gui
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
                logging.warning(f"Dataset index error during dataset restore: {e}")

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
                logging.warning(f"Non-fatal: session combo update failed after dataset restore: {e}", exc_info=True)
        else:
            self.gui.current_session = None

        # Trigger downstream updates that normally occur via combo change handlers
        if hasattr(self.gui, "plot_widget"):
            try:
                self.gui.plot_widget.on_data_selection_changed()
            except Exception as e:
                logging.warning(f"Plot widget refresh after dataset restore failed (non-fatal): {e}", exc_info=True)

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
                logging.warning(f"Plot widget refresh after dataset undo failed (non-fatal): {e}", exc_info=True)


class InvertChannelPolarityCommand(Command):
    def __init__(self, gui, level: str, channel_indexes_to_invert: list[int]):
        self.command_name = "Invert Channel Polarity"
        self.gui: "MonstimGUI" = gui  # type: EMGAnalysisGUI
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
        self.gui: "MonstimGUI" = gui
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
        import copy

        for s in self.sessions:
            s.annot.latency_windows = [copy.deepcopy(w) for w in windows]
            s.update_latency_window_parameters()
            if s.repo is not None:
                s.repo.save(s)
        if hasattr(self.level, "update_latency_window_parameters"):
            if isinstance(self.level, list):
                for obj in self.level:
                    obj.update_latency_window_parameters()
            else:
                self.level.update_latency_window_parameters()

    def execute(self):
        self._apply(self.new_windows)

    def undo(self):
        for s in self.sessions:
            windows = self.old_windows[s.id]
            s.annot.latency_windows = windows
            s.update_latency_window_parameters()
            if s.repo is not None:
                s.repo.save(s)
        if hasattr(self.level, "update_latency_window_parameters"):
            if isinstance(self.level, list):
                for obj in self.level:
                    obj.update_latency_window_parameters()
            else:
                self.level.update_latency_window_parameters()


class ChangeChannelNamesCommand(Command):
    def __init__(self, gui, level: str, new_names: dict):
        self.command_name: str = "Change Channel Names"
        self.gui: "MonstimGUI" = gui
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
            changes: List of dicts with format:
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
        self.gui: "MonstimGUI" = gui
        self.changes = changes

    def execute(self):
        """Apply all recording exclusions/inclusions."""
        try:
            for session_change in self.changes:
                session = session_change["session"]
                for change in session_change["changes"]:
                    recording_id = change["recording_id"]
                    should_exclude = change["exclude"]

                    if should_exclude:
                        session.exclude_recording(recording_id)
                    else:
                        session.restore_recording(recording_id)

            # Update UI to reflect changes
            self.gui.data_selection_widget.sync_combo_selections()

        except Exception as e:
            QMessageBox.critical(self.gui, "Error", f"Failed to apply bulk exclusions: {str(e)}")

    def undo(self):
        """Reverse all recording exclusions/inclusions."""
        try:
            # Apply changes in reverse
            for session_change in reversed(self.changes):
                session = session_change["session"]
                for change in reversed(session_change["changes"]):
                    recording_id = change["recording_id"]
                    should_exclude = change["exclude"]

                    # Do the opposite of what was done
                    if should_exclude:
                        session.restore_recording(recording_id)
                    else:
                        session.exclude_recording(recording_id)

            # Update UI to reflect changes
            self.gui.data_selection_widget.sync_combo_selections()

        except Exception as e:
            QMessageBox.critical(self.gui, "Error", f"Failed to undo bulk exclusions: {str(e)}")


# Data Curation Commands
class CreateExperimentCommand(Command):
    def __init__(self, gui, exp_name: str):
        self.command_name = f"Create Experiment '{exp_name}'"
        self.gui = gui
        self.exp_name = exp_name

    def execute(self):
        """Create the experiment immediately."""
        try:
            self.gui.data_manager.create_experiment(self.exp_name)
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to create experiment: {str(e)}")

    def undo(self):
        """Delete the created experiment."""
        try:
            self.gui.data_manager.delete_experiment_by_id(self.exp_name)
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to undo experiment creation: {str(e)}")

    def get_description(self) -> str:
        return f"Created experiment '{self.exp_name}'"


class MoveDatasetCommand(Command):
    def __init__(self, gui, dataset_id: str, dataset_name: str, from_exp: str, to_exp: str):
        self.command_name = f"Move '{dataset_name}' from '{from_exp}' to '{to_exp}'"
        self.gui = gui
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.from_exp = from_exp
        self.to_exp = to_exp

    def execute(self):
        """Move the dataset immediately."""
        try:
            self.gui.data_manager.move_dataset(self.dataset_id, self.dataset_name, self.from_exp, self.to_exp)
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to move dataset: {str(e)}")

    def undo(self):
        """Move the dataset back to original location."""
        try:
            self.gui.data_manager.move_dataset(self.dataset_id, self.dataset_name, self.to_exp, self.from_exp)
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to undo dataset move: {str(e)}")

    def get_description(self) -> str:
        return f"Moved dataset '{self.dataset_name}' from '{self.from_exp}' to '{self.to_exp}'"


class MoveDatasetsCommand(Command):
    """Batched move of multiple datasets executed as a single undoable command."""

    def __init__(self, gui, moves: list[tuple]):
        """
        moves: list of tuples (dataset_id, dataset_name, from_exp, to_exp)
        """
        self.gui: "MonstimGUI" = gui
        self.moves = list(moves)
        self.command_name = f"Move {len(self.moves)} datasets"
        # Will record only the moves that actually succeeded during execute()
        self._succeeded = []

    def execute(self):
        """Execute all moves sequentially. Record successes for undo."""
        try:
            for ds_id, ds_name, from_exp, to_exp in self.moves:
                try:
                    self.gui.data_manager.move_dataset(ds_id, ds_name, from_exp, to_exp)
                    self._succeeded.append((ds_id, ds_name, from_exp, to_exp))
                except Exception as e:
                    logging.error(f"Failed to move dataset '{ds_name}' from '{from_exp}' to '{to_exp}': {e}")

            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                try:
                    self.gui._data_curation_manager.load_data()
                except Exception:
                    # Best-effort; do not fail the entire command if refresh errors
                    logging.exception("Failed to refresh Data Curation Manager after batched move")

        except Exception as e:
            raise Exception(f"Failed to execute batched dataset moves: {str(e)}")

    def undo(self):
        """Undo by moving succeeded items back in reverse order."""
        try:
            for ds_id, ds_name, from_exp, to_exp in reversed(self._succeeded):
                try:
                    # Move back from to_exp -> from_exp
                    self.gui.data_manager.move_dataset(ds_id, ds_name, to_exp, from_exp)
                except Exception as e:
                    logging.error(f"Failed to undo move of dataset '{ds_name}' from '{to_exp}' back to '{from_exp}': {e}")

            # Refresh once after undo
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                try:
                    self.gui._data_curation_manager.load_data()
                except Exception:
                    logging.exception("Failed to refresh Data Curation Manager after undoing batched move")

        except Exception as e:
            raise Exception(f"Failed to undo batched dataset moves: {str(e)}")

    def get_description(self) -> str:
        return f"Moved {len(self._succeeded)} dataset(s) in batch"


class CopyDatasetCommand(Command):
    def __init__(self, gui, dataset_id: str, dataset_name: str, from_exp: str, to_exp: str, new_name: str = None):
        self.command_name = f"Copy '{dataset_name}' from '{from_exp}' to '{to_exp}'"
        self.gui = gui
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
            original_datasets = set(f.name for f in to_exp_path.iterdir() if f.is_dir())

            self.gui.data_manager.copy_dataset(self.dataset_id, self.dataset_name, self.from_exp, self.to_exp, self.new_name)

            # Find the new dataset folder name (might have _copy suffix)
            new_datasets = set(f.name for f in to_exp_path.iterdir() if f.is_dir())
            added_datasets = new_datasets - original_datasets
            if added_datasets:
                self.copied_folder_name = list(added_datasets)[0]
            else:
                self.copied_folder_name = self.dataset_id  # fallback

            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to copy dataset: {str(e)}")

    def undo(self):
        """Delete the copied dataset."""
        try:
            if self.copied_folder_name:
                self.gui.data_manager.delete_dataset(self.copied_folder_name, self.copied_folder_name, self.to_exp)
                # Refresh the data curation manager if it's open
                if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                    self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to undo dataset copy: {str(e)}")

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
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to delete experiment: {str(e)}")

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
        try:
            self.gui.data_manager.rename_experiment_by_id(self.old_name, self.new_name)
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to rename experiment: {str(e)}")

    def undo(self):
        """Rename back to original name."""
        try:
            self.gui.data_manager.rename_experiment_by_id(self.new_name, self.old_name)
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to undo experiment rename: {str(e)}")

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
            # Refresh the data curation manager if it's open
            if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
                self.gui._data_curation_manager.load_data()
        except Exception as e:
            raise Exception(f"Failed to delete dataset: {str(e)}")

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
            raise Exception(f"Failed to update dataset inclusion: {e}")

        # Refresh open dialog/UI if present
        if hasattr(self.gui, "_data_curation_manager") and self.gui._data_curation_manager:
            self.gui._data_curation_manager.load_data()

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
