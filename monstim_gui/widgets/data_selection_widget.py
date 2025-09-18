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
        is_completed = index.data(Qt.ItemDataRole.UserRole)

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

        # Add Manage Recordings button
        self.manage_recordings_button = QPushButton("Manage Recordings")
        self.manage_recordings_button.setToolTip("Open the recording exclusion editor to manage which recordings are included in analysis")
        self.manage_recordings_button.clicked.connect(self._on_manage_recordings_clicked)
        self.manage_recordings_button.setEnabled(False)  # Start disabled until session is loaded
        form.addRow("", self.manage_recordings_button)

        self.setLayout(form)
        self.setup_context_menus()
        self.update_all_completion_statuses()

        # Apply delegate to all combos
        for combo in (self.dataset_combo, self.session_combo):
            combo.setItemDelegate(self.circle_delegate)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
            combo.view().setTextElideMode(Qt.TextElideMode.ElideRight)

    def setup_context_menus(self):
        for combo in (self.experiment_combo, self.dataset_combo, self.session_combo):
            combo.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        # Connect signals
        self.dataset_combo.customContextMenuRequested.connect(lambda pos: self.show_context_menu(pos, "dataset"))
        self.session_combo.customContextMenuRequested.connect(lambda pos: self.show_context_menu(pos, "session"))

    def show_context_menu(self, pos, level):
        current_obj = {
            "dataset": self.parent.current_dataset,
            "session": self.parent.current_session,
        }.get(level)

        menu = QMenu(self)

        if current_obj:
            action_text = "Mark as Incomplete" if getattr(current_obj, "is_completed", False) else "Mark as Complete"
            toggle_action = menu.addAction(action_text)
            exclude_action = menu.addAction(f"Exclude {level.capitalize()}")
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

        if selected == toggle_action and current_obj:
            current_obj.is_completed = not getattr(current_obj, "is_completed", False)
            self.parent.has_unsaved_changes = True
            self.update_completion_status(level)
        elif selected == exclude_action and current_obj:
            if level == "dataset":
                self.parent.exclude_dataset()
            else:
                self.parent.exclude_session()
        elif restore_menu and selected in restore_menu.actions():
            if level == "dataset":
                self.parent.restore_dataset(selected.text())
            else:
                self.parent.restore_session(selected.text())

    def update_completion_status(self, level):
        """Update visual completion status for specified level"""
        combo = {"dataset": self.dataset_combo, "session": self.session_combo}.get(level)

        if combo and combo.currentIndex() >= 0:
            current_obj = {
                "dataset": self.parent.current_dataset,
                "session": self.parent.current_session,
            }.get(level)

            # Set completion status in item data
            combo.setItemData(
                combo.currentIndex(),
                getattr(current_obj, "is_completed", False),
                Qt.ItemDataRole.UserRole,
            )
            combo.update()

    def update_all_completion_statuses(self):
        """Update all visual completion statuses"""
        for level in ("dataset", "session"):
            self.update_completion_status(level)

    def _on_experiment_combo_changed(self, index):
        # Skip if selecting the placeholder item (index 0)
        if index <= 0:
            # Clear current experiment if placeholder is selected
            if self.parent.current_experiment:
                self.parent.current_experiment = None
                self.parent.current_dataset = None
                self.parent.current_session = None

                # Block signals to prevent recursive calls
                self.experiment_combo.blockSignals(True)
                self.dataset_combo.blockSignals(True)
                self.session_combo.blockSignals(True)

                self.update_all_data_combos()

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

    def update_experiment_combo(self):
        self.experiment_combo.clear()

        # Add placeholder item
        self.experiment_combo.addItem("-- Select an Experiment --")
        self.experiment_combo.setItemData(0, "Please select an experiment to load", role=Qt.ItemDataRole.ToolTipRole)

        # Style the placeholder item to be grayed out/italic
        font = self.experiment_combo.font()
        font.setItalic(True)
        self.experiment_combo.setItemData(0, font, Qt.ItemDataRole.FontRole)

        if self.parent.expts_dict_keys:
            for expt_id in self.parent.expts_dict_keys:
                self.experiment_combo.addItem(expt_id)
                index = self.experiment_combo.count() - 1
                self.experiment_combo.setItemData(index, expt_id, role=Qt.ItemDataRole.ToolTipRole)
                # Ensure regular items have normal font
                normal_font = self.experiment_combo.font()
                normal_font.setItalic(False)
                self.experiment_combo.setItemData(index, normal_font, Qt.ItemDataRole.FontRole)
        else:
            logging.warning("Cannot update experiments combo. No experiments loaded.")

    def update_dataset_combo(self):
        self.dataset_combo.clear()
        if self.parent.current_experiment:
            for dataset in self.parent.current_experiment.datasets:
                self.dataset_combo.addItem(dataset.formatted_name)
                index = self.dataset_combo.count() - 1 if self.dataset_combo.count() > 0 else 0
                self.dataset_combo.setItemData(index, dataset.formatted_name, role=Qt.ItemDataRole.ToolTipRole)
                self.dataset_combo.setItemData(
                    index,
                    getattr(dataset, "is_completed", False),
                    Qt.ItemDataRole.UserRole,
                )
        else:
            # Add placeholder when no experiment is loaded
            self.dataset_combo.addItem("-- No Experiment Selected --")
            self.dataset_combo.setItemData(0, "Please select an experiment first", role=Qt.ItemDataRole.ToolTipRole)
            self.dataset_combo.setEnabled(False)
            logging.debug("Dataset combo cleared - no experiment loaded.")

    def update_session_combo(self):
        self.session_combo.clear()
        if self.parent.current_dataset:
            for session in self.parent.current_dataset.sessions:
                self.session_combo.addItem(session.formatted_name)
                index = self.session_combo.count() - 1 if self.session_combo.count() > 0 else 0
                self.session_combo.setItemData(index, session.formatted_name, role=Qt.ItemDataRole.ToolTipRole)
                self.session_combo.setItemData(
                    index,
                    getattr(session, "is_completed", False),
                    Qt.ItemDataRole.UserRole,
                )
            self.session_combo.setEnabled(True)
        else:
            # Add placeholder when no dataset is loaded
            self.session_combo.addItem("-- No Dataset Selected --")
            self.session_combo.setItemData(0, "Please select a dataset first", role=Qt.ItemDataRole.ToolTipRole)
            self.session_combo.setEnabled(False)
            logging.debug("Session combo cleared - no dataset loaded.")
        
        self._update_manage_recordings_button()

    def update_all_data_combos(self):
        self.update_experiment_combo()
        self.update_dataset_combo()
        self.update_session_combo()

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
        """Update combo contents and sync selections with current objects."""
        self.update_all_data_combos()
        self.sync_combo_selections()
