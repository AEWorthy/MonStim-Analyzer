import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMenu,
    QPushButton,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)

if TYPE_CHECKING:
    from gui_main import MonstimGUI


class CircleDelegate(QStyledItemDelegate):
    """
    A custom delegate for rendering a colored circle in a view item to indicate completion status.
    Attributes:
        completed_color (QColor): The color used to indicate a completed item (default is green).
        uncompleted_color (QColor): The color used to indicate an uncompleted item (default is red).
    Methods:
        paint(painter: QPainter, option, index):
            Renders the item with a colored circle indicating its completion status.
            The circle is drawn at the right side of the item.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.completed_color = QColor(0, 255, 0)  # Green
        self.uncompleted_color = QColor(255, 0, 0)  # Red
        self.CIRCLE_PADDING = 20  # Space reserved for the circle on the right side

    def paint(self, painter: QPainter, option, index):
        """Draw the item text and a completion circle."""
        # Check if painter is active before proceeding
        if not painter.isActive():
            return

        # Reserve space on the right for the status circle
        option_no_circle = QStyleOptionViewItem(option)
        option_no_circle.rect = option.rect.adjusted(0, 0, -self.CIRCLE_PADDING, 0)

        super().paint(painter, option_no_circle, index)

        # Get completion status from item data
        # Some combos store non-bool data in UserRole (e.g., experiment id).
        # If UserRole isn't a bool, fall back to UserRole+1 for the completion flag.
        data_user = index.data(Qt.ItemDataRole.UserRole)
        if isinstance(data_user, bool):
            is_completed = data_user
        else:
            is_completed = index.data(Qt.ItemDataRole.UserRole + 1)

        # For placeholder items or unknown status, don't draw a circle
        if is_completed is None:
            return

        # Draw circle indicating completion
        circle_rect = QRect(option.rect.right() - 18, option.rect.center().y() - 6, 12, 12)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.completed_color if is_completed else self.uncompleted_color
        painter.setBrush(color)
        painter.setPen(color)
        painter.drawEllipse(circle_rect)
        painter.restore()


class DataSelectionWidget(QGroupBox):
    def __init__(self, parent: "MonstimGUI"):
        super().__init__("Data Selection", parent)
        self.parent: "MonstimGUI" = parent
        self.circle_delegate = CircleDelegate(self)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(6)  # tighten the space between label & field
        form.setVerticalSpacing(4)  # vertical spacing between rows
        form.setContentsMargins(4, 4, 4, 4)  # outer margins of the groupbox

        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_combo_changed)
        self.experiment_combo.setToolTip("Select an experiment")
        self.experiment_combo.wheelEvent = lambda event: None  # Disable scroll wheel

        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_combo_changed)
        self.dataset_combo.setEnabled(False)  # Start disabled until experiment is loaded
        self.dataset_combo.setToolTip("Select a dataset")
        self.dataset_combo.wheelEvent = lambda event: None  # Disable scroll wheel

        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self._on_session_combo_changed)
        self.session_combo.setEnabled(False)  # Start disabled until dataset is loaded
        self.session_combo.setToolTip("Select a session")
        self.session_combo.wheelEvent = lambda event: None  # Disable scroll wheel

        # Create labels with tooltips and ensure they don't get elided/truncated
        experiment_label = QLabel("Experiment:")
        experiment_label.setToolTip("Experiment")
        experiment_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        experiment_label.setSizePolicy(
            experiment_label.sizePolicy().horizontalPolicy(), experiment_label.sizePolicy().verticalPolicy()
        )

        dataset_label = QLabel("Dataset:")
        dataset_label.setToolTip("Dataset")
        dataset_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        dataset_label.setSizePolicy(dataset_label.sizePolicy().horizontalPolicy(), dataset_label.sizePolicy().verticalPolicy())

        session_label = QLabel("Session:")
        session_label.setToolTip("Session")
        session_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        session_label.setSizePolicy(session_label.sizePolicy().horizontalPolicy(), session_label.sizePolicy().verticalPolicy())

        # Compute a safe minimum label width so short labels like "Experiment" are fully visible
        fm = experiment_label.fontMetrics()
        labels = (experiment_label, dataset_label, session_label)
        # Measure the widest label text and add padding
        widest = max(fm.horizontalAdvance(lbl.text()) for lbl in labels)
        min_label_width = int(widest)
        for lbl in labels:
            lbl.setMinimumWidth(min_label_width)
            lbl.setSizePolicy(lbl.sizePolicy().horizontalPolicy(), lbl.sizePolicy().verticalPolicy())

        form.addRow(experiment_label, self.experiment_combo)
        form.addRow(dataset_label, self.dataset_combo)
        form.addRow(session_label, self.session_combo)

        # Add management buttons in a horizontal layout
        from PyQt6.QtWidgets import QHBoxLayout, QWidget

        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)

        # Manage Recordings button
        self.manage_recordings_button = QPushButton("Manage Recordings")
        self.manage_recordings_button.setToolTip(
            "Open the recording exclusion editor to manage which recordings are included in analysis"
        )
        self.manage_recordings_button.clicked.connect(self._on_manage_recordings_clicked)
        self.manage_recordings_button.setEnabled(False)  # Start disabled until session is loaded
        button_layout.addWidget(self.manage_recordings_button)

        # Manage Data button
        self.manage_data_button = QPushButton("Manage Data")
        self.manage_data_button.setToolTip("Open the data curation manager to organize experiments and datasets")
        self.manage_data_button.clicked.connect(self._on_manage_data_clicked)
        button_layout.addWidget(self.manage_data_button)

        form.addRow("", button_widget)

        self.setLayout(form)
        self.setup_context_menus()
        self.update_all_completion_statuses()

        # Apply delegate to all combos
        for combo in (self.experiment_combo, self.dataset_combo, self.session_combo):
            combo.setItemDelegate(self.circle_delegate)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.view().setTextElideMode(Qt.TextElideMode.ElideRight)

    # -----------------------------
    # Unified update/refresh API
    # -----------------------------
    def update(self, levels: tuple[str, ...] | None = None, preserve_selection: bool = True) -> None:
        """Update the widget display to reflect current GUI/domain state.

        This is the primary method for refreshing the data selection combos. It rebuilds
        the combo box items and synchronizes visual selections with the current application state.

        - Rebuilds the requested combo boxes' items from current domain objects.
        - Preserves the user's current selection by default (using parent's current_*).
        - Blocks signals during updates to prevent recursive loading operations.
        - Updates completion status indicators and button states.

        Args:
            levels: Which levels to update. Defaults to ("experiment","dataset","session").
            preserve_selection: When True, selects parent's current objects; when False, resets to placeholders.

        Note:
            Use this method instead of the deprecated update_*_combo() methods.
            For startup restoration, ensure data loading happens through DataManager methods.
        """
        if levels is None:
            levels = ("experiment", "dataset", "session")

        # Block all combo signals during batched updates to avoid recursive loads
        combos = {
            "experiment": self.experiment_combo,
            "dataset": self.dataset_combo,
            "session": self.session_combo,
        }

        # Record intended selections from parent's current state (stable across rebuilds)
        parent = self.parent
        target_exp_id = getattr(parent.current_experiment, "id", None)
        target_dataset = parent.current_dataset
        target_session = parent.current_session

        # Helper lambdas
        def _block_all(val: bool):
            for c in combos.values():
                try:
                    c.blockSignals(val)
                except Exception:
                    pass

        def _select_by_index(combo: QComboBox, idx: int):
            try:
                combo.setCurrentIndex(idx)
            except Exception:
                pass

        _block_all(True)
        try:
            # ----- Experiment -----
            if "experiment" in levels:
                self.experiment_combo.clear()

                # Placeholder item
                self.experiment_combo.addItem("-- Select an Experiment --")
                self.experiment_combo.setItemData(0, "Please select an experiment to load", role=Qt.ItemDataRole.ToolTipRole)
                font = self.experiment_combo.font()
                font.setItalic(True)
                self.experiment_combo.setItemData(0, font, Qt.ItemDataRole.FontRole)

                if parent.expts_dict_keys:
                    for expt_id in parent.expts_dict_keys:
                        # Check for empty experiment via repository metadata (best-effort)
                        display_name = expt_id
                        tooltip = expt_id
                        exp_completed = None
                        try:
                            from pathlib import Path

                            from monstim_signals.io.repositories import ExperimentRepository

                            exp_path = Path(parent.expts_dict[expt_id])
                            repo = ExperimentRepository(exp_path)
                            metadata = repo.get_metadata()
                            dataset_count = metadata.get("dataset_count", 0)
                            exp_completed = metadata.get("is_completed", False)
                            tooltip = (
                                f"Experiment: {expt_id} - {dataset_count} dataset{'s' if dataset_count != 1 else ''}"
                                if dataset_count > 0
                                else f"Empty experiment: {expt_id} - No datasets found"
                            )
                        except Exception:
                            pass

                        self.experiment_combo.addItem(display_name)
                        idx = self.experiment_combo.count() - 1
                        # Store tooltip and actual id for lookup
                        self.experiment_combo.setItemData(idx, tooltip, role=Qt.ItemDataRole.ToolTipRole)
                        self.experiment_combo.setItemData(idx, expt_id, role=Qt.ItemDataRole.UserRole)
                        normal_font = self.experiment_combo.font()
                        normal_font.setItalic(False)
                        self.experiment_combo.setItemData(idx, normal_font, Qt.ItemDataRole.FontRole)
                        # Store completion status on a separate role to avoid clobbering the id
                        self.experiment_combo.setItemData(idx, exp_completed, Qt.ItemDataRole.UserRole + 1)

                # Selection
                if preserve_selection and target_exp_id and parent.expts_dict_keys:
                    try:
                        exp_index = parent.expts_dict_keys.index(target_exp_id) + 1  # +1 for placeholder
                    except ValueError:
                        exp_index = 0
                else:
                    exp_index = 0
                _select_by_index(self.experiment_combo, exp_index)

            # ----- Dataset -----
            if "dataset" in levels:
                self.dataset_combo.clear()

                # If we are in the middle of a threaded load, show loading state
                data_manager = getattr(parent, "data_manager", None)
                if (
                    data_manager
                    and hasattr(data_manager, "loading_thread")
                    and data_manager.loading_thread
                    and data_manager.loading_thread.isRunning()
                ):
                    self.dataset_combo.addItem("-- Loading Experiment --")
                    self.dataset_combo.setItemData(
                        0, "Please wait while experiment loads...", role=Qt.ItemDataRole.ToolTipRole
                    )
                    self.dataset_combo.setEnabled(False)
                elif parent.current_experiment:
                    if parent.current_experiment.datasets:
                        for ds in parent.current_experiment.datasets:
                            self.dataset_combo.addItem(ds.formatted_name)
                            idx = max(0, self.dataset_combo.count() - 1)
                            self.dataset_combo.setItemData(idx, ds.formatted_name, role=Qt.ItemDataRole.ToolTipRole)
                            self.dataset_combo.setItemData(idx, getattr(ds, "is_completed", False), Qt.ItemDataRole.UserRole)
                        self.dataset_combo.setEnabled(True)
                    else:
                        self.dataset_combo.addItem("-- Empty Experiment (No Datasets) --")
                        self.dataset_combo.setItemData(
                            0,
                            f"Experiment '{parent.current_experiment.id}' contains no datasets",
                            role=Qt.ItemDataRole.ToolTipRole,
                        )
                        self.dataset_combo.setEnabled(False)
                else:
                    self.dataset_combo.addItem("-- No Experiment Selected --")
                    self.dataset_combo.setItemData(0, "Please select an experiment first", role=Qt.ItemDataRole.ToolTipRole)
                    self.dataset_combo.setEnabled(False)

                # Selection - preserve existing selection or fall back to first available
                ds_idx = 0  # Default to first item (which might be placeholder)
                if preserve_selection and parent.current_experiment and target_dataset:
                    try:
                        ds_idx = parent.current_experiment.datasets.index(target_dataset)
                    except ValueError:
                        # Target dataset not found, fall back to first available dataset
                        ds_idx = 0 if parent.current_experiment.datasets else 0
                elif parent.current_experiment and parent.current_experiment.datasets:
                    # No target to preserve, select first available dataset
                    ds_idx = 0

                _select_by_index(self.dataset_combo, ds_idx)

            # ----- Session -----
            if "session" in levels:
                self.session_combo.clear()
                if parent.current_dataset:
                    for s in parent.current_dataset.sessions:
                        self.session_combo.addItem(s.formatted_name)
                        idx = max(0, self.session_combo.count() - 1)
                        self.session_combo.setItemData(idx, s.formatted_name, role=Qt.ItemDataRole.ToolTipRole)
                        self.session_combo.setItemData(idx, getattr(s, "is_completed", False), Qt.ItemDataRole.UserRole)
                    self.session_combo.setEnabled(True)
                else:
                    if parent.current_experiment and len(parent.current_experiment) == 0:
                        self.session_combo.addItem("-- Empty Experiment --")
                        self.session_combo.setItemData(
                            0, "This experiment contains no datasets", role=Qt.ItemDataRole.ToolTipRole
                        )
                    else:
                        self.session_combo.addItem("-- No Dataset Selected --")
                        self.session_combo.setItemData(0, "Please select a dataset first", role=Qt.ItemDataRole.ToolTipRole)
                    self.session_combo.setEnabled(False)

                # Selection - preserve existing selection or fall back to first available
                s_idx = 0  # Default to first item (which might be placeholder)
                if preserve_selection and parent.current_dataset and target_session:
                    try:
                        s_idx = parent.current_dataset.sessions.index(target_session)
                    except ValueError:
                        # Target session not found, fall back to first available session
                        s_idx = 0 if parent.current_dataset.sessions else 0
                elif parent.current_dataset and parent.current_dataset.sessions:
                    # No target to preserve, select first available session
                    s_idx = 0

                _select_by_index(self.session_combo, s_idx)

            # Update completion indicators for visible selections
            self.update_all_completion_statuses()
            self._update_manage_recordings_button()
        finally:
            _block_all(False)

    def refresh(self, levels: tuple[str, ...] | None = None) -> None:
        """Hard refresh of the widget display for specified levels.

        This method forces a complete rebuild and resets selections to placeholders.
        Use this for user-triggered refresh actions or after major data structure changes.

        - First rescans the filesystem to get ground-truth experiment data
        - Rebuilds combos and resets selection to placeholders/unselected for those levels.
        - Use sparingly: intended for explicit refresh operations, not routine updates.

        Args:
            levels: Which levels to refresh. Defaults to all levels.
        """
        # Set default levels if not specified
        if levels is None:
            levels = ("experiment", "dataset", "session")

        # Always rescan filesystem for ground-truth data when refreshing experiments
        if "experiment" in levels:
            self.parent.data_manager.unpack_existing_experiments()

        self.update(levels=levels, preserve_selection=False)

    def setup_context_menus(self):
        for combo in (self.experiment_combo, self.dataset_combo, self.session_combo):
            combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Connect signals
        self.experiment_combo.customContextMenuRequested.connect(lambda pos: self.show_context_menu(pos, "experiment"))
        self.dataset_combo.customContextMenuRequested.connect(lambda pos: self.show_context_menu(pos, "dataset"))
        self.session_combo.customContextMenuRequested.connect(lambda pos: self.show_context_menu(pos, "session"))

    def show_context_menu(self, pos, level):
        current_obj = {
            "experiment": self.parent.current_experiment,
            "dataset": self.parent.current_dataset,
            "session": self.parent.current_session,
        }.get(level)

        menu = QMenu(self)

        if current_obj:
            action_text = "Mark as Incomplete" if getattr(current_obj, "is_completed", False) else "Mark as Complete"
            toggle_action = menu.addAction(action_text)
            exclude_action = None if level == "experiment" else menu.addAction(f"Exclude {level.capitalize()}")
        else:
            toggle_action = exclude_action = None

        excluded_ids = []
        if level == "dataset" and self.parent.current_experiment:
            excluded_ids = list(self.parent.current_experiment.excluded_datasets)
        elif level == "session" and self.parent.current_dataset:
            excluded_ids = list(self.parent.current_dataset.excluded_sessions)

        restore_menu = None
        if excluded_ids:
            restore_menu = menu.addMenu(f"Restore {level.capitalize()}")
            for item_id in excluded_ids:
                restore_menu.addAction(item_id)

        selected = menu.exec(self.sender().mapToGlobal(pos))

        # If the user dismissed the menu without a selection, do nothing
        if selected is None:
            return

        if selected == toggle_action and current_obj:
            current_obj.is_completed = not getattr(current_obj, "is_completed", False)
            self.parent.has_unsaved_changes = True
            # Update only the toggled item, then refresh all statuses for safety
            self.update_completion_status(level)
            self.update_all_completion_statuses()
        elif exclude_action is not None and selected == exclude_action and current_obj:
            if level == "dataset":
                self.parent.exclude_dataset()
                # Rebuild combos to prevent stale item data/indices after removal
                self.update(levels=("dataset", "session"), preserve_selection=True)
                self._update_manage_recordings_button()
                # Notify downstream UI that selection/data changed
                if hasattr(self.parent, "plot_widget"):
                    try:
                        self.parent.plot_widget.on_data_selection_changed()
                    except Exception:
                        pass
            else:
                self.parent.exclude_session()
                # Rebuild sessions to keep statuses aligned
                self.update(levels=("session",), preserve_selection=True)
                self._update_manage_recordings_button()
                if hasattr(self.parent, "plot_widget"):
                    try:
                        self.parent.plot_widget.on_data_selection_changed()
                    except Exception:
                        pass
        elif restore_menu and selected in restore_menu.actions():
            if level == "dataset":
                self.parent.restore_dataset(selected.text())
                # Rebuild combos after restore to refresh items and statuses
                self.update(levels=("dataset", "session"), preserve_selection=True)
                self._update_manage_recordings_button()
                if hasattr(self.parent, "plot_widget"):
                    try:
                        self.parent.plot_widget.on_data_selection_changed()
                    except Exception:
                        pass
            else:
                self.parent.restore_session(selected.text())
                self.update(levels=("session",), preserve_selection=True)
                self._update_manage_recordings_button()
                if hasattr(self.parent, "plot_widget"):
                    try:
                        self.parent.plot_widget.on_data_selection_changed()
                    except Exception:
                        pass

    def update_completion_status(self, level):
        """Update visual completion status for specified level"""
        combo = {"experiment": self.experiment_combo, "dataset": self.dataset_combo, "session": self.session_combo}.get(level)

        if combo and combo.currentIndex() >= 0:
            current_obj = {
                "experiment": self.parent.current_experiment,
                "dataset": self.parent.current_dataset,
                "session": self.parent.current_session,
            }.get(level)

            # Set completion status in item data
            role = Qt.ItemDataRole.UserRole if level in ("dataset", "session") else Qt.ItemDataRole.UserRole + 1
            # Avoid writing to placeholder row in experiment combo
            if not (level == "experiment" and combo.currentIndex() == 0):
                combo.setItemData(
                    combo.currentIndex(),
                    getattr(current_obj, "is_completed", False),
                    role,
                )
            combo.update()

    def update_all_completion_statuses(self):
        """Update all visual completion statuses for dataset and session combos.

        This method rewrites the completion status (UserRole) for every item to
        ensure indices and statuses remain aligned after list mutations
        (exclude/restore) without relying on a full rebuild.
        """
        # Experiments
        try:
            if self.parent.expts_dict_keys and self.experiment_combo.count() > 0:
                from pathlib import Path

                from monstim_signals.io.repositories import ExperimentRepository

                # Skip index 0 (placeholder)
                for i, expt_id in enumerate(self.parent.expts_dict_keys, start=1):
                    if i < self.experiment_combo.count():
                        try:
                            exp_path = Path(self.parent.expts_dict[expt_id])
                            repo = ExperimentRepository(exp_path)
                            meta = repo.get_metadata()
                            self.experiment_combo.setItemData(
                                i, bool(meta.get("is_completed", False)), Qt.ItemDataRole.UserRole + 1
                            )
                        except Exception:
                            # If metadata not available, clear status (no circle)
                            self.experiment_combo.setItemData(i, None, Qt.ItemDataRole.UserRole + 1)
                self.experiment_combo.update()
        except Exception:
            pass

        # Datasets
        try:
            if (
                self.parent.current_experiment
                and getattr(self.parent.current_experiment, "datasets", None)
                and self.dataset_combo.isEnabled()
            ):
                datasets = self.parent.current_experiment.datasets
                # The dataset combo, when enabled, contains only dataset items (no placeholder)
                for i, ds in enumerate(datasets):
                    if i < self.dataset_combo.count():
                        self.dataset_combo.setItemData(i, getattr(ds, "is_completed", False), Qt.ItemDataRole.UserRole)
                self.dataset_combo.update()
        except Exception:
            pass

        # Sessions
        try:
            if (
                self.parent.current_dataset
                and getattr(self.parent.current_dataset, "sessions", None)
                and self.session_combo.isEnabled()
            ):
                sessions = self.parent.current_dataset.sessions
                for i, s in enumerate(sessions):
                    if i < self.session_combo.count():
                        self.session_combo.setItemData(i, getattr(s, "is_completed", False), Qt.ItemDataRole.UserRole)
                self.session_combo.update()
        except Exception:
            pass

    def _on_experiment_combo_changed(self, index):
        # Skip if selecting the placeholder item (index 0)
        if index <= 0:
            # Clear current experiment if placeholder is selected
            if self.parent.current_experiment:
                self.parent.set_current_experiment(None)
                self.parent.set_current_dataset(None)
                self.parent.set_current_session(None)

                # Block signals to prevent recursive calls
                self.experiment_combo.blockSignals(True)
                self.dataset_combo.blockSignals(True)
                self.session_combo.blockSignals(True)

                self.update()

                # Re-enable signals
                self.experiment_combo.blockSignals(False)
                self.dataset_combo.blockSignals(False)
                self.session_combo.blockSignals(False)

                self._update_manage_recordings_button()
                self.parent.plot_widget.on_data_selection_changed()
            return

        # Check if we have experiments available
        if not self.parent.expts_dict_keys or index > len(self.parent.expts_dict_keys):
            logging.warning(f"Invalid experiment index: {index}")
            return

        if self.parent.has_unsaved_changes:
            self.parent.data_manager.save_experiment()

        # Adjust index to account for placeholder item
        self.parent.data_manager.load_experiment(index - 1)

    def _on_dataset_combo_changed(self):
        index = self.dataset_combo.currentIndex()
        # Skip if invalid index or no experiment loaded
        if index < 0 or not self.parent.current_experiment:
            return

        self.parent.data_manager.load_dataset(index)
        self._update_manage_recordings_button()

    def _on_session_combo_changed(self):
        index = self.session_combo.currentIndex()
        # Skip if invalid index or no dataset loaded
        if index < 0 or not self.parent.current_dataset:
            return

        self.parent.data_manager.load_session(index)
        self._update_manage_recordings_button()

    def _on_manage_recordings_clicked(self):
        """Open the recording exclusion editor dialog."""
        if not self.parent.current_session:
            return

        from monstim_gui.dialogs.recording_exclusion_editor import RecordingExclusionEditor

        dialog = RecordingExclusionEditor(self.parent)
        dialog.exclusions_applied.connect(self._on_exclusions_applied)
        dialog.exec()

    def _on_exclusions_applied(self):
        """Handle when exclusions are applied from the recording exclusion editor."""
        # Refresh the current session data and update UI
        if self.parent.current_session:
            # Reset cached properties that might be affected by exclusions
            self.parent.current_session.reset_all_caches()

        # Notify other parts of the UI that data has changed
        self.parent.plot_widget.on_data_selection_changed()

    def _update_manage_recordings_button(self):
        """Update the enabled state of the manage recordings button."""
        self.manage_recordings_button.setEnabled(self.parent.current_session is not None)

    def _on_manage_data_clicked(self):
        """Open the data curation manager dialog."""
        try:
            from monstim_gui.dialogs.data_curation_manager import DataCurationManager

            dialog = DataCurationManager(self.parent)
            result = dialog.exec()

            # If changes were applied, refresh the GUI
            if result == dialog.DialogCode.Accepted:
                # Refresh experiments list and reset selections after structural changes
                self.refresh()

        except ImportError as e:
            logging.error(f"Failed to import Data Curation Manager: {e}")
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Error", "Data Curation Manager is not available.")
        except Exception as e:
            logging.error(f"Failed to open data curation manager: {e}")
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.warning(self, "Error", f"Failed to open data manager:\n{str(e)}")

    def sync_combo_selections(self):
        """Synchronize combo box selections with current objects without rebuilding them."""
        # This method is mainly for recording operations that don't affect higher-level selections
        # For exclude/restore operations, use more targeted updates

        # Sync experiment combo (should rarely be needed)
        if self.parent.current_experiment and self.parent.expts_dict_keys:
            try:
                exp_index = self.parent.expts_dict_keys.index(self.parent.current_experiment.id) + 1  # +1 for placeholder
                if self.experiment_combo.currentIndex() != exp_index:
                    self.experiment_combo.blockSignals(True)
                    self.experiment_combo.setCurrentIndex(exp_index)
                    self.experiment_combo.blockSignals(False)
            except ValueError:
                # Experiment not found in list - set to placeholder
                logging.warning("Current experiment not found in combo list. Setting to placeholder.")
                if self.experiment_combo.currentIndex() != 0:
                    self.experiment_combo.blockSignals(True)
                    self.experiment_combo.setCurrentIndex(0)  # Placeholder
                    self.experiment_combo.blockSignals(False)

        # Sync dataset combo (should rarely be needed for recording operations)
        if self.parent.current_dataset and self.parent.current_experiment:
            try:
                dataset_index = self.parent.current_experiment.datasets.index(self.parent.current_dataset)
                if self.dataset_combo.currentIndex() != dataset_index:
                    self.dataset_combo.blockSignals(True)
                    self.dataset_combo.setCurrentIndex(dataset_index)
                    self.dataset_combo.blockSignals(False)
            except ValueError:
                # Dataset not found in list (might be excluded)
                pass

        # Sync session combo (should rarely be needed for recording operations)
        if self.parent.current_session and self.parent.current_dataset:
            try:
                session_index = self.parent.current_dataset.sessions.index(self.parent.current_session)
                if self.session_combo.currentIndex() != session_index:
                    self.session_combo.blockSignals(True)
                    self.session_combo.setCurrentIndex(session_index)
                    self.session_combo.blockSignals(False)
            except ValueError:
                # Session not found in list (might be excluded)
                pass

    def update_combos_and_sync(self):
        """Deprecated: Update contents and sync selections. Use update()."""
        self.update()
