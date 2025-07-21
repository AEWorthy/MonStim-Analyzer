import logging
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (
    QGroupBox,
    QFormLayout,
    QComboBox,
    QMenu,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QColor, QPainter

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
        # Reserve space on the right for the status circle
        option_no_circle = QStyleOptionViewItem(option)
        option_no_circle.rect = option.rect.adjusted(0, 0, -self.CIRCLE_PADDING, 0)

        super().paint(painter, option_no_circle, index)

        # Get completion status from item data
        is_completed = index.data(Qt.ItemDataRole.UserRole)

        # Draw circle indicating completion
        circle_rect = QRect(option.rect.right() - 18,
                            option.rect.center().y() - 6,
                            12, 12)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = self.completed_color if is_completed else self.uncompleted_color
        painter.setBrush(color)
        painter.setPen(color)
        painter.drawEllipse(circle_rect)
        painter.restore()

class DataSelectionWidget(QGroupBox):
    def __init__(self, parent: 'MonstimGUI'):
        super().__init__("Data Selection", parent)
        self.parent : 'MonstimGUI' = parent
        self.circle_delegate = CircleDelegate(self)
        
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(6)      # tighten the space between label & field
        form.setVerticalSpacing(4)        # vertical spacing between rows
        form.setContentsMargins(8, 8, 8, 8)  # outer margins of the groupbox
        
        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(self._on_experiment_combo_changed)
        
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_combo_changed)

        self.session_combo = QComboBox()
        self.session_combo.currentIndexChanged.connect(self._on_session_combo_changed)

        form.addRow("Select Experiment:", self.experiment_combo)
        form.addRow("Select Dataset:",    self.dataset_combo)
        form.addRow("Select Session:",    self.session_combo)

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
        self.dataset_combo.customContextMenuRequested.connect(
            lambda pos: self.show_context_menu(pos, 'dataset'))
        self.session_combo.customContextMenuRequested.connect(
            lambda pos: self.show_context_menu(pos, 'session'))

    def show_context_menu(self, pos, level):
        current_obj = {
            'dataset': self.parent.current_dataset,
            'session': self.parent.current_session,
        }.get(level)

        menu = QMenu(self)

        if current_obj:
            action_text = (
                "Mark as Incomplete" if getattr(current_obj, 'is_completed', False)
                else "Mark as Complete"
            )
            toggle_action = menu.addAction(action_text)
            exclude_action = menu.addAction(f"Exclude {level.capitalize()}")
        else:
            toggle_action = exclude_action = None

        excluded_ids = []
        if level == 'dataset' and self.parent.current_experiment:
            excluded_ids = list(self.parent.current_experiment.excluded_datasets)
        elif level == 'session' and self.parent.current_dataset:
            excluded_ids = list(self.parent.current_dataset.excluded_sessions)

        restore_menu = None
        if excluded_ids:
            restore_menu = menu.addMenu(f"Restore {level.capitalize()}")
            for item_id in excluded_ids:
                restore_menu.addAction(item_id)

        selected = menu.exec(self.sender().mapToGlobal(pos))

        if selected == toggle_action and current_obj:
            current_obj.is_completed = not getattr(current_obj, 'is_completed', False)
            self.parent.has_unsaved_changes = True
            self.update_completion_status(level)
        elif selected == exclude_action and current_obj:
            if level == 'dataset':
                self.parent.exclude_dataset()
            else:
                self.parent.exclude_session()
        elif restore_menu and selected in restore_menu.actions():
            if level == 'dataset':
                self.parent.restore_dataset(selected.text())
            else:
                self.parent.restore_session(selected.text())

    def update_completion_status(self, level):
        """Update visual completion status for specified level"""
        combo = {
            'dataset': self.dataset_combo,
            'session': self.session_combo
        }.get(level)
        
        if combo and combo.currentIndex() >= 0:
            current_obj = {
                'dataset': self.parent.current_dataset,
                'session': self.parent.current_session
            }.get(level)
            
            # Set completion status in item data
            combo.setItemData(combo.currentIndex(),
                            getattr(current_obj, 'is_completed', False),
                            Qt.ItemDataRole.UserRole)
            combo.update()

    def update_all_completion_statuses(self):
        """Update all visual completion statuses"""
        for level in ('dataset', 'session'):
            self.update_completion_status(level)

    def _on_experiment_combo_changed(self, index):
        if self.parent.has_unsaved_changes:
            self.parent.data_manager.save_experiment()
        self.parent.data_manager.load_experiment(index)
    
    def _on_dataset_combo_changed(self):
        self.parent.data_manager.load_dataset(index=self.dataset_combo.currentIndex())


    def _on_session_combo_changed(self):
        self.parent.data_manager.load_session(index=self.session_combo.currentIndex())


    def update_experiment_combo(self):
        self.experiment_combo.clear()
        if self.parent.expts_dict_keys:
            for expt_id in self.parent.expts_dict_keys:
                self.experiment_combo.addItem(expt_id)
                index = self.experiment_combo.count() - 1
                self.experiment_combo.setItemData(index, expt_id, role=Qt.ItemDataRole.ToolTipRole)
        else:
            logging.warning("Cannot update experiments combo. No experiments loaded.")

    def update_dataset_combo(self):
        self.dataset_combo.clear()
        if self.parent.current_experiment:
            for dataset in self.parent.current_experiment.datasets:
                self.dataset_combo.addItem(dataset.formatted_name)
                index = self.dataset_combo.count() - 1
                self.dataset_combo.setItemData(index, dataset.formatted_name, role=Qt.ItemDataRole.ToolTipRole)
                self.dataset_combo.setItemData(index, getattr(dataset, 'is_completed', False), Qt.ItemDataRole.UserRole)
        else:
            logging.warning("Cannot update datasets combo. No experiment loaded.")

    def update_session_combo(self):
        self.session_combo.clear()
        if self.parent.current_dataset:
            for session in self.parent.current_dataset.sessions:
                self.session_combo.addItem(session.formatted_name)
                index = self.session_combo.count() - 1
                self.session_combo.setItemData(index, session.formatted_name, role=Qt.ItemDataRole.ToolTipRole)
                self.session_combo.setItemData(index, getattr(session, 'is_completed', False), Qt.ItemDataRole.UserRole)
        else:
            logging.warning("Cannot update sessions combo. No dataset loaded.")

    def update_all_data_combos(self):
        self.update_experiment_combo()
        self.update_dataset_combo()
        self.update_session_combo()
    