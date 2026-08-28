"""
Data Curation Manager Dialog
Allows users to manage experiments and datasets with create/import/delete/rename operations
and drag-and-drop dataset organization between experiments.
"""

# TODO: Data Curation Manager feature roadmap
# - Recycle bin / safe-delete: move datasets to a hidden .trash instead of immediate permanent deletion,
#   add restore & purge UI, and schedule automatic purge after configurable retention.
# - Background workers and progress/cancellation: run long operations (move/copy/import) in background
#   threads with progress dialogs and per-item error reporting.
# - Dry-run / preview modal for batch move/copy/delete: show affected items, conflicts,
#   and allow skip/overwrite/rename decisions before executing changes.
# - Validation & consistency checker: validate selected datasets for missing files, mismatched channels,
#   missing latency windows, etc., and provide quick-fix suggestions.
# - Add a Trim operation to remove 0-indexed recording numbers from all sessions in the selected data.
#   Make sure it can handle negative indexing to trim from the ends. Useful for trimming known bad recordings like the first/second ones.
# - Inline / bulk metadata editor: allow editing dataset metadata (date/animal/condition) from the tree
#   or an edit pane; support bulk edits and preview.
# - Batch rename tooling: pattern-based renaming (tokens or regex) with a preview before applying.
# - Duplicate detection & merge assistant: find likely duplicate datasets and offer safe merge options.
# - Tagging and saved views: let users tag datasets and save filterable views for recurring workflows.

import logging
import shlex
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDrag, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI

logger = logging.getLogger(__name__)


def auto_refresh(method):
    """
    Decorator that automatically calls self.load_data() after a method completes,
    but only if the method executes successfully without exceptions.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            # For PyQt signal connections, filter out unexpected boolean arguments
            # that can be passed by clicked signals ONLY if:
            # 1. There's exactly one argument
            # 2. That argument is a boolean
            # This prevents filtering out legitimate boolean parameters when multiple args exist
            filtered_args = args
            if len(args) == 1 and isinstance(args[0], bool):
                # This is likely a spurious signal argument (e.g., from clicked signal)
                # that should be filtered out
                filtered_args = ()

            result = method(self, *filtered_args, **kwargs)
            # Only refresh if the method completed successfully and auto-refresh is not suppressed
            if not getattr(self, "_suppress_autorefresh", False):
                self.load_data()
            return result
        except Exception as e:
            # If there was an error, still refresh to ensure UI consistency
            try:
                if not getattr(self, "_suppress_autorefresh", False):
                    self.load_data()
            except Exception as refresh_error:
                logger.error(f"Failed to refresh data after error in {method.__name__}: {refresh_error}")
            # Re-raise the original exception
            raise e

    return wrapper


class DatasetTreeWidget(QTreeWidget):
    """Custom QTreeWidget that handles dataset drag-and-drop operations."""

    dataset_moved = Signal(str, str, str, str)  # dataset_id, formatted_name, source_exp, target_exp
    dataset_move_batch_start = Signal()
    dataset_moved_batch = Signal(int, str)  # count, target_exp_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)
        # Enable automatic scrolling when dragging near the viewport edges
        try:
            self.setAutoScroll(True)
            # Margin in pixels to start autoscroll when cursor approaches edge
            self._auto_scroll_margin = 30
            try:
                # Use built-in setter if available
                self.setAutoScrollMargin(self._auto_scroll_margin)
            except Exception:
                # Non-fatal: setAutoScrollMargin may not be available on all platforms/PyQt versions
                logger.warning("setAutoScrollMargin not available; using default margin for autoscroll.")
        except Exception:
            # Non-fatal if attributes aren't present on the platform
            self._auto_scroll_margin = 30

    def startDrag(self, supportedActions):
        """Create a custom drag pixmap when dragging multiple dataset items.

        Shows a small collage of up to 3 dataset names and a count badge when
        multiple dataset items are selected so users can see how many items
        they're moving.
        """
        # Prefer datasets that are checked (checkboxes) as the dragged set. If
        # none are checked, fall back to the selected items. Only include
        # dataset children (items with a parent).
        checked_items = []
        for i in range(self.topLevelItemCount()):
            exp_item = self.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                try:
                    if ds_item.checkState(0) == Qt.CheckState.Checked:
                        checked_items.append(ds_item)
                except Exception:
                    # Some items may not support checkState (e.g., corrupted or custom widgets).
                    # Safe to ignore and skip these items for drag selection.
                    logger.warning("Failed to check state of item; skipping.")

        items = checked_items or [it for it in self.selectedItems() if it and it.parent()]

        # If still nothing, fallback to default behavior
        if not items:
            return super().startDrag(supportedActions)

        # Build pixmap for drag feedback
        try:
            pix = self._create_drag_pixmap(items)
        except Exception:
            pix = QPixmap()  # empty fallback

        # Create a QDrag and build mime data from the model indexes for each item
        drag = QDrag(self)
        try:
            idxs = []
            for it in items:
                try:
                    idx = self.indexFromItem(it)
                    if idx.isValid():
                        idxs.append(idx)
                except Exception:
                    # Ignore items that cannot be converted to valid indexes (e.g., corrupted or non-standard items).
                    pass

            if idxs:
                mime = self.model().mimeData(idxs)
                drag.setMimeData(mime)
        except Exception:
            # Fall back to widget-selected indexes if building mime from items fails
            try:
                mime = self.model().mimeData(list(self.selectedIndexes()))
                drag.setMimeData(mime)
            except Exception as e:
                # Safe to ignore: drag will proceed without mime data if this fails.
                logger.debug("Failed to set mime data from selected indexes: %r", e)

        if not pix.isNull():
            drag.setPixmap(pix)
            try:
                # Place the pixmap's top-left corner at the cursor click point
                drag.setHotSpot(pix.rect().topLeft())
            except Exception as e:
                logger.debug("Failed to set drag hotspot for pixmap: %r", e)
                # Safe to ignore: drag will proceed without custom hotspot

        # Execute the drag using the supported actions passed in
        drag.exec(supportedActions)

    def dragMoveEvent(self, event):
        """During drag, pan the viewport when cursor approaches the edges.

        This provides a smooth click-and-drag panning experience inside the
        scroll box so users can move items to targets that are outside the
        initially visible area.
        """
        try:
            pos = event.position().toPoint()
            vp_rect = self.viewport().rect()
            margin = getattr(self, "_auto_scroll_margin", 30)

            vsb = self.verticalScrollBar()
            hsb = self.horizontalScrollBar()

            # Reduce pan speed: use a smaller fraction of pageStep for smooth, slow panning
            scroll_amount_v = max(1, int(vsb.pageStep() * 0.05)) if vsb is not None else 5
            scroll_amount_h = max(1, int(hsb.pageStep() * 0.05)) if hsb is not None else 5

            # Vertical scrolling
            if pos.y() < vp_rect.top() + margin:
                try:
                    vsb.setValue(vsb.value() - scroll_amount_v)
                except Exception as e:
                    # Safe to ignore: vertical auto-scroll is a convenience feature.
                    logger.debug("Failed to auto-scroll up: %r", e)
            elif pos.y() > vp_rect.bottom() - margin:
                try:
                    vsb.setValue(vsb.value() + scroll_amount_v)
                except Exception as e:
                    # Safe to ignore: vertical auto-scroll is a convenience feature.
                    logger.debug("Failed to auto-scroll down: %r", e)

            # Horizontal scrolling (if needed)
            if pos.x() < vp_rect.left() + margin:
                try:
                    hsb.setValue(hsb.value() - scroll_amount_h)
                except Exception as e:
                    # Safe to ignore: horizontal auto-scroll is a convenience feature.
                    logger.debug("Failed to auto-scroll left: %r", e)
            elif pos.x() > vp_rect.right() - margin:
                try:
                    hsb.setValue(hsb.value() + scroll_amount_h)
                except Exception as e:
                    # Safe to ignore: horizontal auto-scroll is a convenience feature.
                    logger.debug("Failed to auto-scroll right: %r", e)
        except Exception:
            # Don't let autoscroll failures block drag
            pass

        # Continue with normal handling
        try:
            super().dragMoveEvent(event)
        except Exception:
            event.accept()

    def _create_drag_pixmap(self, items):
        """Create a compact pixmap representing the selected datasets.

        items: list of QTreeWidgetItem
        Returns QPixmap
        """
        # Limit the number of preview names shown
        max_preview = 3
        names = [it.text(0) for it in items[:max_preview]]
        count = len(items)

        # Pixmap sizing
        width = 260
        line_height = 18
        padding = 8
        height = padding * 2 + max(1, len(names)) * line_height

        pix = QPixmap(width, height)
        pix.fill(QColor(0, 0, 0, 0))  # transparent

        painter = QPainter(pix)
        try:
            # Background rounded rect
            pen = QPen(QColor(120, 120, 120, 200))
            painter.setPen(pen)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QColor(245, 245, 245, 230))
            rect = pix.rect().adjusted(0, 0, -1, -1)
            painter.drawRoundedRect(rect, 6, 6)

            # Draw dataset name previews
            font = QFont()
            font.setPointSize(9)
            painter.setFont(font)
            metrics = QFontMetrics(font)
            text_x = padding
            y = padding + metrics.ascent()
            elide_width = width - padding * 3 - 28

            for nm in names:
                elided = metrics.elidedText(nm, Qt.TextElideMode.ElideRight, elide_width)
                painter.setPen(QColor(30, 30, 30))
                painter.drawText(text_x, y, elided)
                y += line_height

            # If there are more items than shown, indicate with a trailing "+N"
            if count > max_preview:
                more_text = f"+{count - max_preview} more"
                painter.setPen(QColor(110, 110, 110))
                painter.drawText(text_x, y, more_text)

            # Draw circular count badge in top-right
            badge_d = 28
            badge_x = width - badge_d - padding
            badge_y = padding
            painter.setBrush(QColor(0, 120, 215))
            painter.setPen(QPen(Qt.GlobalColor.transparent))
            painter.drawEllipse(badge_x, badge_y, badge_d, badge_d)

            # Badge text
            badge_font = QFont()
            badge_font.setPointSize(9)
            badge_font.setBold(True)
            painter.setFont(badge_font)
            painter.setPen(QColor(255, 255, 255))
            fm = QFontMetrics(badge_font)
            txt = str(count)
            tx = badge_x + (badge_d - fm.horizontalAdvance(txt)) // 2
            ty = badge_y + (badge_d + fm.ascent() - fm.descent()) // 2
            painter.drawText(tx, ty, txt)

        finally:
            painter.end()

        return pix

    def dropEvent(self, event):
        """Handle drop events for moving datasets between experiments."""
        if event.source() != self:
            event.ignore()
            return

        drop_item = self.itemAt(event.position().toPoint())
        if not drop_item:
            event.ignore()
            return

        # Get the target experiment (either the dropped-on item or its parent)
        target_exp_item = drop_item.parent() if drop_item.parent() else drop_item

        target_exp_data = target_exp_item.data(0, Qt.ItemDataRole.UserRole)
        if not target_exp_data or target_exp_data.get("type") != "experiment":
            event.ignore()
            return

        target_exp_id = target_exp_data.get("id")
        if not target_exp_id:
            event.ignore()
            return

        # Determine which dataset items to move. Copy their identifying metadata first
        # to avoid referencing QTreeWidgetItem objects that may be deleted during move operations.
        items_info = []  # list of (dataset_id, formatted_name, source_exp_id)

        # Notify listeners that a batch move is starting so the UI can prepare
        try:
            self.dataset_move_batch_start.emit()
        except Exception:
            # If anything goes wrong emitting the signal, continue gracefully
            logger.warning("Failed to emit dataset_move_batch_start signal; proceeding with move.")

        # Collect checked dataset children across all experiments
        for i in range(self.topLevelItemCount()):
            exp_item = self.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                try:
                    if ds_item.checkState(0) == Qt.CheckState.Checked:
                        data = ds_item.data(0, Qt.ItemDataRole.UserRole)
                        if data and data.get("type") == "dataset":
                            ds_meta = data.get("metadata", {})
                            items_info.append((ds_meta.get("id", ""), ds_meta.get("formatted_name", ""), data.get("experiment_id")))
                except Exception:
                    # skip items that don't support check state
                    pass

        # If no checked items, fall back to the current selection
        if not items_info:
            for sel in self.selectedItems():
                data = sel.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "dataset":
                    ds_meta = data.get("metadata", {})
                    items_info.append((ds_meta.get("id", ""), ds_meta.get("formatted_name", ""), data.get("experiment_id")))

        moved_count = 0
        # Emit move for each dataset item found (supports items from different source experiments)
        for ds_id, ds_name, source_exp_id in items_info:
            # Don't move if it's already in the target experiment
            if source_exp_id != target_exp_id:
                self.dataset_moved.emit(ds_id, ds_name, source_exp_id, target_exp_id)
                moved_count += 1

        # Emit batch-complete signal so the UI can show a single consolidated message
        if moved_count > 0:
            self.dataset_moved_batch.emit(moved_count, target_exp_id)

        event.accept()


# HighlightDelegate removed


class DataCurationManager(QDialog):
    """
    Modal dialog for comprehensive data curation including experiment and dataset management.
    Uses preview/apply pattern with separate tabs for different operations.
    """

    data_structure_changed = Signal()  # Signal emitted when data structure changes

    def __init__(self, parent: MonstimGUI):
        super().__init__(parent)
        try:
            self.gui = parent

            # Track commands executed during this session for undo on cancel
            self.session_commands = []

            # Track if any changes were made during this session
            self._changes_made = False

            # Batch move counters and suppression flag for auto-refresh
            self._batch_successful_moves = 0
            self._batch_processed_moves = 0
            self._suppress_autorefresh = False

            self.setup_ui()
            self.load_data()

        except Exception as e:
            logger.error(f"Failed to initialize Data Curation Manager: {e}")
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize dialog:\n{e!s}")
            raise

    def setup_ui(self):
        """Set up the dialog UI with tabbed interface."""
        self.setWindowTitle("Data Curation Manager")
        self.setModal(True)
        # Match the working footprint of the Recording Exclusion Editor.  The
        # data tree is the primary task surface, so it needs room rather than
        # being squeezed beneath a wall of batch-operation buttons.
        self.resize(1000, 600)

        # Main layout with reduced margins
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        self.data_management_tab = self.create_data_management_tab()
        main_layout.addWidget(self.data_management_tab, 1)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        # Add some spacing (push remaining controls to the right)
        button_layout.addStretch()

        # Undo last change button - will be placed next to Undo All for proximity
        self.undo_last_button = QPushButton("Undo Last Change")
        self.undo_last_button.clicked.connect(self.undo_last_change)
        self.undo_last_button.setEnabled(False)
        self.undo_last_button.setToolTip("No changes to undo")
        button_layout.addWidget(self.undo_last_button)

        # Undo all and close buttons
        self.cancel_button = QPushButton("Undo All Changes")
        self.cancel_button.clicked.connect(self.cancel_all_changes)
        button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton("Done")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def create_data_management_tab(self) -> QWidget:
        """Create the data management tab with drag-and-drop functionality."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)

        # Header
        header_label = QLabel("Data Management")
        header_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        try:
            header_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        except Exception:
            logger.warning("Failed to set size policy for header label; using default.")
        layout.addWidget(header_label)

        # Instructions
        instructions = QLabel(
            "Right-click to access context menus. Drag datasets between experiments to reorganize your data structure. "
            "Use checkboxes for batch operations."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; margin-bottom: 10px;")
        try:
            instructions.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        except Exception:
            logger.warning("Failed to set size policy for instructions label; using default.")
        layout.addWidget(instructions)

        # Keep commands close to the data they operate on without letting them
        # dominate the dialog.  Less-common batch commands live in two compact
        # menus; their QAction instances retain the same enabled-state logic as
        # the former buttons.
        command_layout = QHBoxLayout()
        command_layout.setSpacing(6)

        self.create_blank_experiment_button = QToolButton()
        self.create_blank_experiment_button.setText("New Experiment")
        self.create_blank_experiment_button.clicked.connect(self.create_experiment)
        self.create_blank_experiment_button.setToolTip("Create a new empty experiment for organizing datasets")
        command_layout.addWidget(self.create_blank_experiment_button)

        self.import_new_experiment_button = QToolButton()
        self.import_new_experiment_button.setText("Import…")
        self.import_new_experiment_button.clicked.connect(self.import_experiment)
        self.import_new_experiment_button.setToolTip("Import a new experiment using the standard import workflow")
        command_layout.addWidget(self.import_new_experiment_button)

        self.rebuild_catalogs_button = QToolButton()
        self.rebuild_catalogs_button.setText("Rebuild Catalog")
        self.rebuild_catalogs_button.setToolTip("Re-scan all files in the currently selected experiment to recover data missing from the catalog")
        self.rebuild_catalogs_button.clicked.connect(self.force_rebuild_catalogs)
        command_layout.addWidget(self.rebuild_catalogs_button)

        self.select_all_button = QToolButton()
        self.select_all_button.setText("Select All")
        self.select_all_button.clicked.connect(self.select_all_datasets)
        command_layout.addWidget(self.select_all_button)

        self.clear_selection_button = QToolButton()
        self.clear_selection_button.setText("Clear")
        self.clear_selection_button.clicked.connect(self.clear_dataset_selection)
        self.clear_selection_button.setToolTip("Clear checked datasets")
        command_layout.addWidget(self.clear_selection_button)

        def add_selected_action(label, callback, tooltip):
            action = QAction(label, self)
            action.setToolTip(tooltip)
            action.triggered.connect(callback)
            action.setEnabled(False)
            return action

        self.validate_selected_button = add_selected_action(
            "Validate Selected",
            self.validate_selected_datasets,
            "Check checked datasets for readable metadata and session folders.",
        )
        self.exclude_selected_button = add_selected_action(
            "Exclude Selected",
            lambda: self.set_selected_datasets_included(False),
            "Hide checked datasets from analysis without deleting their files.",
        )
        self.restore_selected_button = add_selected_action(
            "Restore Selected",
            lambda: self.set_selected_datasets_included(True),
            "Return checked excluded datasets to analysis.",
        )
        self.mark_complete_button = add_selected_action(
            "Mark Complete",
            lambda: self.set_selected_datasets_completion(True),
            "Mark checked datasets complete without changing any sessions.",
        )
        self.mark_incomplete_button = add_selected_action(
            "Mark Incomplete",
            lambda: self.set_selected_datasets_completion(False),
            "Mark checked datasets incomplete without changing any sessions.",
        )
        self.move_selected_button = add_selected_action(
            "Move Selected To…",
            self.move_selected_datasets,
            "Move checked datasets to another experiment.",
        )
        self.copy_selected_button = add_selected_action(
            "Copy Selected To…",
            self.copy_selected_datasets,
            "Copy checked datasets to another experiment.",
        )
        self.delete_selected_button = add_selected_action(
            "Delete Selected",
            self.delete_selected_datasets,
            "Delete checked datasets.",
        )

        selected_menu = QMenu(self)
        selected_menu.addActions(
            [
                self.validate_selected_button,
                self.exclude_selected_button,
                self.restore_selected_button,
                self.mark_complete_button,
                self.mark_incomplete_button,
            ]
        )
        selected_commands_button = QToolButton()
        selected_commands_button.setText("Selected Actions")
        selected_commands_button.setToolTip("Actions for checked datasets")
        selected_commands_button.setMenu(selected_menu)
        selected_commands_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        command_layout.addWidget(selected_commands_button)

        organize_menu = QMenu(self)
        organize_menu.addActions(
            [
                self.move_selected_button,
                self.copy_selected_button,
                self.delete_selected_button,
            ]
        )
        organize_button = QToolButton()
        organize_button.setText("Organize")
        organize_button.setToolTip("Move, copy, or delete checked datasets")
        organize_button.setMenu(organize_menu)
        organize_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        command_layout.addWidget(organize_button)

        # QToolButton's native minimum can be disproportionately wide on
        # high-DPI Windows styles.  Deliberate compact widths keep this command
        # row below the 800 px working width while preserving readable labels.
        for button, width in (
            (self.create_blank_experiment_button, 110),
            (self.import_new_experiment_button, 75),
            (self.rebuild_catalogs_button, 110),
            (self.select_all_button, 70),
            (self.clear_selection_button, 60),
            (selected_commands_button, 125),
            (organize_button, 80),
        ):
            button.setFixedWidth(width)
        command_layout.addStretch()
        layout.addLayout(command_layout)

        # Search/filter box for quick filtering of datasets
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by name, id, animal, condition, date, date_added...")
        self.search_box.setClearButtonEnabled(True)
        self.search_box.textChanged.connect(self.on_search_text_changed)
        # The search field should be comfortably usable, but its placeholder is
        # intentionally descriptive and must not dictate the dialog width.
        # A 260 px minimum keeps this manager usable on laptop displays; the
        # field still grows with the window through its expanding size policy.
        self.search_box.setMinimumWidth(260)
        self.search_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Filter builder button opens advanced filter dialog
        self.filter_button = QPushButton("Filter")
        self.filter_button.setToolTip("Open filter builder")
        self.filter_button.clicked.connect(self.open_filter_dialog)

        search_layout.addWidget(self.filter_button)
        search_layout.addStretch(1)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_box)
        layout.addLayout(search_layout)

        # Tree widget for hierarchical display
        self.dataset_tree = DatasetTreeWidget()
        # Add Date columns so users can see/import timestamps and sort by them
        self.dataset_tree.setHeaderLabels(
            [
                "Name",
                "Contents",
                "Status",
                "Date Added",
                "Date Modified",
            ]
        )
        # Enable sorting by clicking column headers
        self.dataset_tree.setSortingEnabled(True)
        self.dataset_tree.itemChanged.connect(self.on_dataset_checkbox_changed)
        self.dataset_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dataset_tree.customContextMenuRequested.connect(self.show_dataset_context_menu)
        self.dataset_tree.dataset_moved.connect(self.on_dataset_dragged)
        self.dataset_tree.dataset_moved_batch.connect(self.on_dataset_dragged_batch)
        self.dataset_tree.dataset_move_batch_start.connect(self.on_dataset_drag_start)

        # Configure column widths - make first column stretch to fill available space
        header = self.dataset_tree.header()
        # header.setStretchLastSection(False)  # Don't auto-stretch the last column
        # Let the first column stretch to take remaining space and keep date columns compact
        try:
            # Allow user to drag-resize the first column interactively
            header.setMinimumSectionSize(40)
            for column, width in enumerate((150, 70, 70, 80, 80)):
                header.setSectionResizeMode(column, header.ResizeMode.Interactive)
                header.resizeSection(column, width)
        except Exception:
            # Fallback for older Qt versions
            header.setMinimumSectionSize(40)
            for column, width in enumerate((150, 70, 70, 80, 80)):
                header.setSectionResizeMode(column, header.Interactive)
                header.resizeSection(column, width)

        # Place tree and details pane side-by-side using a splitter for resizable/minimizable behavior
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(6)
        splitter.addWidget(self.dataset_tree)
        # Details pane shows metadata for selected item
        self.details_pane = self._create_details_pane()
        splitter.addWidget(self.details_pane)
        # Ensure the details pane expands vertically to fill available space
        try:
            self.details_pane.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        except Exception:
            logger.debug("Failed to set size policy for details pane; using default.")
        # Favor the tree horizontally, but let both panes expand
        try:
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
        except Exception:
            pass
        # Set initial sizes: favor tree and set details pane to its minimum collapse width
        try:
            min_w = self.details_pane.minimumWidth()
            splitter.setSizes([360, min_w])
        except Exception:
            pass
        # Make the splitter region dictate vertical growth; remove excess blank space above
        layout.addWidget(splitter, 1)

        # Wire selection changed to update details pane
        try:
            self.dataset_tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        except Exception:
            logger.debug("Failed to connect selection change handler for dataset_tree", exc_info=True)

        # Summary area
        self.dataset_summary = QLabel("No pending changes")
        self.dataset_summary.setStyleSheet("border: 1px solid gray; padding: 5px;")
        layout.addWidget(self.dataset_summary)
        # Ensure summary bar does not consume vertical space
        try:
            self.dataset_summary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        except Exception:
            logger.debug("Failed to set size policy for dataset summary; using default.")

        # TODO: Consider adding a small 'Views' / 'Saved Filters' pane here in the future
        # to let users save common filter criteria or tag-based views for fast access.

        return tab_widget

    def force_rebuild_catalogs(self):
        """Rebuild the active experiment catalog with visible progress."""
        self.gui.data_manager.rebuild_current_catalog()

    def on_dataset_drag_start(self):
        """Called before a batch of dataset moves begins."""
        # Reset counters and suppress auto-refresh while individual moves execute
        self._batch_successful_moves = 0
        self._batch_processed_moves = 0
        self._suppress_autorefresh = True
        # Track pending moves for batch execution
        self._in_batch_move = True
        self._pending_batch_moves = []
        # Show busy cursor while batch move is in progress to indicate work
        try:
            from PySide6.QtWidgets import QApplication

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        except Exception as e:
            # Setting the busy cursor is cosmetic; ignore failures but log for debugging.
            logger.debug(f"Failed to set busy cursor during batch move: {e}")

    def on_dataset_dragged_batch(self, moved_count: int, target_exp_id: str):
        """Called after a batch of dataset moves completes.

        Shows a single consolidated message and refreshes the dataset tree.
        """
        # We're done with the batch: execute buffered moves as a single command (atomic undo/redo)
        self._suppress_autorefresh = False

        pending = getattr(self, "_pending_batch_moves", []) or []

        # If there are buffered moves, execute them as a single batched command
        if pending:
            try:
                from monstim_gui.commands import MoveDatasetCommand, MoveDatasetsCommand

                if len(pending) == 1:
                    # Single move: use the existing single-dataset command for consistency
                    ds_id, ds_name, from_exp, to_exp = pending[0]
                    cmd = MoveDatasetCommand(self.gui, ds_id, ds_name, from_exp, to_exp)
                else:
                    # Multiple moves: use the new batched command
                    # Normalize the tuples to (id, name, from_exp, to_exp)
                    moves = [(p[0], p[1], p[2], target_exp_id) if len(p) == 3 else p for p in pending]
                    cmd = MoveDatasetsCommand(self.gui, moves)

                # Execute and record command
                try:
                    cmd.execute()
                    self.session_commands.append(cmd)
                    self._changes_made = True
                except Exception as e:
                    logger.error(f"Batched move command failed: {e}")
                    QMessageBox.critical(self, "Move Failed", f"Failed to move datasets:\n{e!s}")

                # Determine how many moves were successful
                succeeded = getattr(cmd, "_succeeded", None)
                success_count = len(succeeded) if succeeded is not None else (1 if len(pending) == 1 else moved_count)

                if success_count > 0:
                    QMessageBox.information(
                        self,
                        "Datasets Moved",
                        f"{success_count} dataset(s) moved to '{target_exp_id}' successfully.",
                    )

            finally:
                # Clear pending and batch flags
                self._pending_batch_moves = []
                self._in_batch_move = False
                # Restore cursor now that batch processing has completed
                try:
                    from PySide6.QtWidgets import QApplication

                    QApplication.restoreOverrideCursor()
                except Exception:
                    # Ignore errors restoring cursor: non-critical UI cleanup.
                    pass

        # Refresh dataset tree now that batch operations are done
        try:
            self.load_data()
        except Exception:
            # fallback to explicit update
            try:
                self.update_dataset_tree()
            except Exception as e:
                # Fallback failed; log the error but do not raise, as this is a non-critical UI update
                logger.error(f"Failed to update dataset tree in Data Curation Manager fallback: {e}")

    def load_data(self):
        """Load current experiment and dataset data."""
        try:
            logger.debug("Data Curation Manager: Loading data...")
            logger.debug(f"Available experiments: {list(self.gui.expts_dict_keys)}")
            logger.debug(f"Experiment paths: {dict(self.gui.expts_dict)}")

            # Store original state
            self.original_experiments = dict(self.gui.expts_dict)

            # Populate experiment tree
            try:
                self.update_dataset_tree()
            except Exception as tree_error:
                logger.error(f"Failed to update dataset tree: {tree_error}", exc_info=True)
                raise

            # Initialize button states
            try:
                self._update_button_states()
            except Exception as button_error:
                logger.error(f"Failed to update button states: {button_error}", exc_info=True)
                # Non-critical, continue

            logger.debug("Data Curation Manager: Data loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load data in Data Curation Manager: {e}", exc_info=True)
            QMessageBox.critical(self, "Data Loading Error", f"Failed to load experiment data:\n{e!s}")
            raise  # Re-raise to propagate the error

    def _save_tree_expansion_state(self):
        """Save the current expansion state of tree items."""
        expansion_state = {}
        selected_datasets = set()

        # Handle case where tree might be empty
        if self.dataset_tree.topLevelItemCount() == 0:
            return expansion_state, selected_datasets

        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            exp_id = exp_item.text(0)  # experiment name
            expansion_state[exp_id] = exp_item.isExpanded()

            # Also save which datasets are checked
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                if ds_item.checkState(0) == Qt.CheckState.Checked:
                    ds_data = ds_item.data(0, Qt.ItemDataRole.UserRole)
                    if ds_data and "metadata" in ds_data:
                        ds_key = (ds_data["experiment_id"], ds_data["metadata"].get("id", ""))
                        selected_datasets.add(ds_key)

        return expansion_state, selected_datasets

    def _restore_tree_expansion_state(self, expansion_state, selected_datasets):
        """Restore the expansion state and dataset selections."""
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            exp_id = exp_item.text(0)  # experiment name

            # Restore expansion state (default to expanded if not previously saved)
            if exp_id in expansion_state:
                exp_item.setExpanded(expansion_state[exp_id])
            else:
                exp_item.setExpanded(False)  # Default to collapsed for new experiments

            # Restore dataset selections
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                ds_data = ds_item.data(0, Qt.ItemDataRole.UserRole)
                if ds_data and "metadata" in ds_data:
                    ds_key = (ds_data["experiment_id"], ds_data["metadata"].get("id", ""))
                    if ds_key in selected_datasets:
                        ds_item.setCheckState(0, Qt.CheckState.Checked)

    def update_dataset_tree(self):
        """Update the dataset tree widget using lightweight metadata scanning."""
        try:
            logger.debug("Updating dataset tree...")

            # Save current state before clearing
            expansion_state, selected_datasets = self._save_tree_expansion_state()

            self.dataset_tree.clear()

            tree_items = 0
            for exp_id in self.gui.expts_dict_keys:
                try:
                    logger.debug(f"Processing experiment: {exp_id}")
                    exp_path = Path(self.gui.expts_dict[exp_id])

                    # Get experiment metadata using repository method
                    try:
                        exp_metadata = self._get_experiment_metadata(exp_path)
                        logger.debug(f"Retrieved metadata for {exp_id}: {exp_metadata.get('dataset_count', 0)} datasets")
                    except Exception as metadata_error:
                        logger.error(f"Failed to get metadata for '{exp_id}': {metadata_error}", exc_info=True)
                        raise

                    # Create experiment node with status and dates
                    dataset_count = exp_metadata.get("dataset_count", 0)
                    is_completed = bool(exp_metadata.get("is_completed", False))
                    exp_status = "Complete" if is_completed else "Incomplete"
                    exp_date_added = exp_metadata.get("date_added") or ""
                    exp_date_modified = exp_metadata.get("date_modified") or ""
                    exp_item = QTreeWidgetItem(
                        [
                            exp_id,
                            f"{dataset_count} datasets",
                            exp_status,
                            self._format_date(exp_date_added),
                            self._format_date(exp_date_modified),
                        ]
                    )
                    exp_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "experiment", "id": exp_id, "metadata": exp_metadata})
                    exp_item.setFlags(exp_item.flags() | Qt.ItemFlag.ItemIsDropEnabled)
                    # Bold experiment row to differentiate from datasets
                    try:
                        for _col in range(5):
                            fexp = exp_item.font(_col)
                            fexp.setBold(True)
                            exp_item.setFont(_col, fexp)
                    except Exception:
                        pass
                    # Tooltip shows full experiment name
                    try:
                        exp_item.setData(0, Qt.ItemDataRole.ToolTipRole, exp_id)
                    except Exception:
                        logger.warning(f"Failed to set tooltip for experiment '{exp_id}'", exc_info=True)

                    # Add dataset children using metadata
                    for ds_metadata in exp_metadata.get("datasets", []):
                        ds_name = ds_metadata.get("formatted_name", ds_metadata.get("id", "Unknown"))
                        session_count = ds_metadata.get("session_count", 0)
                        is_completed = bool(ds_metadata.get("is_completed", False))
                        status = "Complete" if is_completed else "Incomplete"
                        is_excluded = ds_metadata.get("id") in exp_metadata.get("excluded_datasets", [])
                        if is_excluded:
                            status = f"Excluded ({status})"

                        date_added = ds_metadata.get("date_added") or ""
                        date_modified = ds_metadata.get("date_modified") or ""
                        ds_item = QTreeWidgetItem(
                            [
                                ds_name,
                                f"{session_count} sessions",
                                status,
                                self._format_date(date_added),
                                self._format_date(date_modified),
                            ]
                        )
                        ds_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "dataset", "experiment_id": exp_id, "metadata": ds_metadata})
                        ds_item.setFlags(
                            ds_item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsSelectable
                        )
                        ds_item.setCheckState(0, Qt.CheckState.Unchecked)
                        # Tooltip shows full dataset name
                        try:
                            ds_item.setData(0, Qt.ItemDataRole.ToolTipRole, ds_name)
                        except Exception:
                            logger.warning(f"Failed to set tooltip for dataset '{ds_name}'", exc_info=True)

                        # Light styling for excluded datasets: italic + gray text
                        if is_excluded:
                            gray = QBrush(QColor(170, 170, 170))  # a soft gray
                            for col in range(5):
                                ds_item.setForeground(col, gray)
                                f = ds_item.font(col) if hasattr(ds_item, "font") else QFont()
                                f.setItalic(True)
                                ds_item.setFont(col, f)

                        exp_item.addChild(ds_item)

                    self.dataset_tree.addTopLevelItem(exp_item)
                    tree_items += 1
                    logger.debug(f"Added experiment to tree: {exp_id} with {len(exp_metadata.get('datasets', []))} datasets")

                except Exception as e:
                    logger.error(f"Failed to scan experiment '{exp_id}' metadata: {e}", exc_info=True)
                    # Add a placeholder item for the failed experiment
                    try:
                        exp_item = QTreeWidgetItem([exp_id, "Error loading", "Failed"])
                        exp_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "experiment", "id": exp_id})
                        self.dataset_tree.addTopLevelItem(exp_item)
                        tree_items += 1
                    except Exception as placeholder_error:
                        logger.error(f"Failed to add placeholder for failed experiment '{exp_id}': {placeholder_error}", exc_info=True)

            # Restore expansion state and selections after all items are added
            self._restore_tree_expansion_state(expansion_state, selected_datasets)

            # Apply active filter immediately so tree respects search box after reload
            try:
                txt = self.search_box.text() if hasattr(self, "search_box") else ""
                if txt:
                    self.on_search_text_changed(txt)
            except Exception:
                pass

            # Update button states after restoring selections (without triggering signals)
            self._update_button_states()

            logger.debug(f"Dataset tree updated with {tree_items} experiments")

            # Resize the Name column based on the longest experiment title up to a reasonable max
            try:
                header = self.dataset_tree.header()
                fm = self.dataset_tree.fontMetrics() if hasattr(self.dataset_tree, "fontMetrics") else QFontMetrics(self.dataset_tree.font())
                max_text = ""
                for i in range(self.dataset_tree.topLevelItemCount()):
                    t = self.dataset_tree.topLevelItem(i).text(0)
                    if len(t) > len(max_text):
                        max_text = t
                width = fm.horizontalAdvance(max_text) + 40  # padding for icon/checkbox
                width = min(max(300, width), 600)
                header.resizeSection(0, width)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed to update dataset tree: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load dataset information:\n{e!s}")

    def _collect_filter_terms(self) -> dict:
        """Collect unique values WITH counts for filter dialog from current experiments/datasets.

        Returns mapping key -> {value -> count}
        Excludes name (unique) by design for dialog.
        """
        terms: dict = {
            "id": {},
            "animal": {},
            "condition": {},
            "date": {},
            "added": {},
            "modified": {},
            "status": {},
            "experiment": {},
        }
        try:
            for exp_id in self.gui.expts_dict_keys:
                terms["experiment"][exp_id] = terms["experiment"].get(exp_id, 0) + 1
                # Pull experiment metadata
                exp_path = Path(self.gui.expts_dict[exp_id])
                exp_meta = self._get_experiment_metadata(exp_path)
                if exp_meta:
                    for k in ("date_added", "date_modified"):
                        v = exp_meta.get(k)
                        if v:
                            key = "added" if k == "date_added" else "modified"
                            val = self._format_date(v)
                            terms[key][val] = terms[key].get(val, 0) + 1
                    status = "Complete" if exp_meta.get("is_completed") else "Incomplete"
                    terms["status"][status] = terms["status"].get(status, 0) + 1
                # Dataset metadata
                for ds in exp_meta.get("datasets") or []:
                    if ds.get("id"):
                        val = str(ds.get("id"))
                        terms["id"][val] = terms["id"].get(val, 0) + 1
                    if ds.get("animal_id"):
                        val = str(ds.get("animal_id"))
                        terms["animal"][val] = terms["animal"].get(val, 0) + 1
                    if ds.get("condition"):
                        val = str(ds.get("condition"))
                        terms["condition"][val] = terms["condition"].get(val, 0) + 1
                    if ds.get("date"):
                        val = str(ds.get("date"))
                        terms["date"][val] = terms["date"].get(val, 0) + 1
                    if ds.get("date_added"):
                        val = self._format_date(ds.get("date_added"))
                        terms["added"][val] = terms["added"].get(val, 0) + 1
                    if ds.get("date_modified"):
                        val = self._format_date(ds.get("date_modified"))
                        terms["modified"][val] = terms["modified"].get(val, 0) + 1
                    st = "Complete" if ds.get("is_completed") else "Incomplete"
                    if ds.get("id") in (exp_meta.get("excluded_datasets") or []):
                        st = f"Excluded ({st})"
                    terms["status"][st] = terms["status"].get(st, 0) + 1
        except Exception:
            pass
        return terms

    def open_filter_dialog(self):
        """Open the advanced filter builder and apply selections to the search box."""
        try:
            from monstim_gui.dialogs.filter_dialog import FilterDialog
        except Exception as e:
            logger.error(f"Failed to import FilterDialog: {e}")
            return

        qualifiers = [
            ("id", "Dataset ID"),
            ("animal", "Animal"),
            ("condition", "Condition"),
            ("date", "Date (dataset)"),
            ("added", "Date Added"),
            ("modified", "Date Modified"),
            ("status", "Status"),
            ("experiment", "Experiment"),
        ]
        terms_with_counts = self._collect_filter_terms()
        dlg = FilterDialog(qualifiers, terms_with_counts, parent=self)
        if dlg.exec():
            q = dlg.result_query()
            if q:
                self.search_box.setText(q)

    def _format_date(self, text: str) -> str:
        """Format a date/time string to 'YYYY-MM-DD HH:MM'. Returns original if parsing fails or empty."""
        try:
            if not text:
                return ""
            s = str(text)
            # Normalize common formats like '2025-12-04T12:35:14' or '2025-12-04 12:35:14'
            s = s.replace("T", " ")
            # Keep up to minutes
            # Expected 'YYYY-MM-DD HH:MM:SS' or longer
            parts = s.split()
            if len(parts) >= 2:
                date_part = parts[0]
                time_part = parts[1]
                hm = time_part[:5] if len(time_part) >= 5 else time_part
                return f"{date_part} {hm}"
            return s
        except Exception:
            return text

    def _get_experiment_metadata(self, exp_path: Path) -> dict:
        """Get lightweight metadata about an experiment using repository method."""
        try:
            from monstim_signals.io.repositories import ExperimentRepository

            logger.debug(f"Creating ExperimentRepository for {exp_path}")
            exp_repo = ExperimentRepository(exp_path)
            logger.debug(f"Getting metadata for {exp_path.name}")
            metadata = exp_repo.get_metadata()
            logger.debug(f"Successfully retrieved metadata for {exp_path.name}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to get experiment metadata for {exp_path.name}: {e}", exc_info=True)
            return {
                "id": exp_path.name,
                "path": str(exp_path),
                "dataset_count": 0,
                "datasets": [],
                "is_completed": False,
                "error": str(e),
            }

    def _create_details_pane(self) -> QWidget:
        """Create a details pane widget that shows metadata for the selected experiment or dataset.
        The pane is constrained in width to avoid causing the window/splitter to expand when long text appears.
        """
        # Outer group box
        box = QGroupBox("Details")
        box_layout = QVBoxLayout()
        # Slightly reduced left margin to tighten title alignment
        box_layout.setContentsMargins(4, 6, 6, 6)
        box_layout.setSpacing(4)

        # Scrollable content to prevent vertical overflow while keeping width fixed
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Let the scroll area grow vertically with the dialog
        try:
            scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        except Exception:
            logger.debug("Failed to set size policy for details scroll area; using default.")

        # Inner content: wrap form in vertical and horizontal layouts
        content = QWidget()
        content_vbox = QVBoxLayout()
        # Reduce left buffer inside the content area as well
        content_vbox.setContentsMargins(2, 4, 2, 4)
        content_vbox.setSpacing(4)
        content_vbox.setSpacing(4)
        form = QFormLayout()
        # Tight margins and spacing; explicit label widgets for consistent alignment
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(3)
        form.setContentsMargins(0, 0, 0, 0)
        try:
            # Keep label and field on the same row; allow the field widget to wrap its text
            form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        except Exception:
            pass

        def _mk_label():
            lbl = QLabel("")
            lbl.setWordWrap(True)
            # Allow labels to wrap within available width and grow vertically to avoid clipping
            lbl.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
            try:
                lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            except Exception:
                logger.warning("Failed to set text interaction flags for label; using default.")
            return lbl

        self.detail_type = _mk_label()
        self.detail_id = _mk_label()
        self.detail_name = _mk_label()
        self.detail_path = _mk_label()
        self.detail_sessions = _mk_label()
        self.detail_status = _mk_label()
        self.detail_date = _mk_label()
        self.detail_date_added = _mk_label()
        self.detail_date_modified = _mk_label()

        # Monospace font for all value fields for consistent alignment
        try:
            mono_candidates = ["Consolas", "Courier New", "Monospace"]

            def _apply_mono(lbl: QLabel):
                f = lbl.font()
                for fam in mono_candidates:
                    try:
                        f.setFamily(fam)
                        break
                    except Exception:
                        continue
                lbl.setFont(f)

            for _lbl in (
                self.detail_type,
                self.detail_id,
                self.detail_name,
                self.detail_path,
                self.detail_sessions,
                self.detail_status,
                self.detail_date,
                self.detail_date_added,
                self.detail_date_modified,
            ):
                _apply_mono(_lbl)
        except Exception:
            pass

        # Improve path readability: use monospace and allow soft wrapping at separators
        try:
            f = self.detail_path.font()
            f.setFamily("Consolas")
            self.detail_path.setFont(f)
        except Exception:
            pass
        # Prefer a minimum-expanding vertical policy for path block to prevent clipping
        try:
            self.detail_path.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        except Exception:
            logger.debug("Failed to set size policy for path label; using default.")

        # Helper: build right-aligned, fixed-width label
        def _mk_title(text: str) -> QLabel:
            lbl = QLabel(text)
            try:
                f = lbl.font()
                f.setBold(True)
                lbl.setFont(f)
            except Exception:
                pass
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            # Narrower title width to reduce left buffer and keep values closer to titles
            lbl.setMinimumWidth(90)
            lbl.setMaximumWidth(120)
            lbl.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            return lbl

        # Add rows with explicit title labels to ensure consistent alignment
        form.addRow(_mk_title("Type:"), self.detail_type)
        form.addRow(_mk_title("ID:"), self.detail_id)
        form.addRow(_mk_title("Name:"), self.detail_name)
        form.addRow(_mk_title("Path:"), self.detail_path)
        form.addRow(_mk_title("Sessions:"), self.detail_sessions)
        form.addRow(_mk_title("Status:"), self.detail_status)
        form.addRow(_mk_title("Date (dataset):"), self.detail_date)
        form.addRow(_mk_title("Date Added:"), self.detail_date_added)
        form.addRow(_mk_title("Date Modified:"), self.detail_date_modified)

        # Add a vertical stretch to push form rows to the top
        content_vbox.addLayout(form)
        content_vbox.addStretch(1)
        content.setLayout(content_vbox)

        # Wrap content in an HBox and add a right-side stretch to keep left alignment
        hwrap = QWidget()
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        hbox.addWidget(content)
        hbox.addStretch(1)
        hwrap.setLayout(hbox)

        scroll.setWidget(hwrap)
        box_layout.addWidget(scroll)
        box.setLayout(box_layout)
        # Style title to avoid clipping and look nicer
        try:
            box.setStyleSheet("""
                QGroupBox {
                    padding: 6px;
                    border: 1px solid rgba(200,200,200,60);
                    border-radius: 6px;
                    margin-top: 12px; /* space for title */
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    padding: 0 6px;
                    color: palette(window-text);
                }
                QLabel { padding-left: 0px; margin-left: 0px; }
                """)
        except Exception:
            logger.debug("Failed to set stylesheet for details pane; using default.")

        # Constrain width so long text doesn't expand the splitter/window
        try:
            box.setMinimumWidth(200)
            box.setMaximumWidth(300)
        except Exception:
            logger.debug("Failed to set width constraints for details pane; using default.")

        return box

    def on_tree_selection_changed(self):
        """Populate details pane when tree selection changes."""
        try:
            sels = self.dataset_tree.selectedItems()
            if not sels:
                # Clear details
                self.detail_type.setText("")
                self.detail_id.setText("")
                self.detail_name.setText("")
                self.detail_path.setText("")
                self.detail_sessions.setText("")
                self.detail_status.setText("")
                self.detail_date.setText("")
                self.detail_date_added.setText("")
                self.detail_date_modified.setText("")
                return

            item = sels[0]
            data = item.data(0, Qt.ItemDataRole.UserRole) or {}

            # Helper: insert zero-width spaces to encourage wrapping at path separators
            def _wrap_path(text: str) -> str:
                if not text:
                    return ""
                t = str(text)
                # Insert soft wrap points after common separators
                t = t.replace("\\", "\\\u200b").replace("/", "/\u200b")
                t = t.replace("-", "-\u200b").replace("_", "_\u200b")
                return t

            if data.get("type") == "dataset":
                meta = data.get("metadata", {})
                self.detail_type.setText("Dataset")
                self.detail_id.setText(str(meta.get("id", "")))
                self.detail_name.setText(str(meta.get("formatted_name", "")))
                self.detail_path.setText(_wrap_path(meta.get("path", "")))
                self.detail_sessions.setText(str(meta.get("session_count", "")))
                self.detail_status.setText("Complete" if meta.get("is_completed") else "Incomplete")
                self.detail_date.setText(str(meta.get("date", "")))
                self.detail_date_added.setText(str(meta.get("date_added", "")))
                self.detail_date_modified.setText(str(meta.get("date_modified", "")))
            elif data.get("type") == "experiment":
                meta = data.get("metadata", {})
                self.detail_type.setText("Experiment")
                self.detail_id.setText(str(meta.get("id", "")))
                self.detail_name.setText(str(meta.get("id", "")))
                self.detail_path.setText(_wrap_path(meta.get("path", "")))
                self.detail_sessions.setText(str(meta.get("dataset_count", "")))
                self.detail_status.setText("Complete" if meta.get("is_completed") else "Incomplete")
                self.detail_date.setText("")
                self.detail_date_added.setText(str(meta.get("date_added", "")))
                self.detail_date_modified.setText(str(meta.get("date_modified", "")))
            else:
                # Unknown item: clear details
                self.detail_type.setText("")
                self.detail_id.setText("")
                self.detail_name.setText("")
                self.detail_path.setText("")
                self.detail_sessions.setText("")
                self.detail_status.setText("")
                self.detail_date.setText("")
                self.detail_date_added.setText("")
                self.detail_date_modified.setText("")

        except Exception as e:
            logger.error(f"Failed to populate details pane: {e}")

    def on_search_text_changed(self, text: str):
        """Filter tree items by search text with intelligent token matching.

        Supports:
        - plain tokens across name/id/animal/condition/date/added/modified
        - quoted phrases: "rev light"
        - field qualifiers: animal:WT, condition:"rev light", id:123, date:2025,
          added:2024-12, modified:2025, status:Complete, experiment:exp-1
        - alternatives within one qualifier: animal:(WT|KO)
        Tokens and qualifier groups are ANDed. Matching is case-insensitive and substring-based.
        """
        try:
            q = (text or "").strip()
            try:
                # shlex keeps qualified quoted values (animal:"rev light") and
                # grouped alternatives as one token while accepting ordinary text.
                tokens = shlex.split(q, posix=True)
            except ValueError:
                # Keep filtering responsive while the user is typing an unfinished quote.
                tokens = q.split()

            def norm(s):
                return str(s or "").strip().lower()

            key_map = {
                "name": "name",
                "id": "id",
                "animal": "animal_id",
                "animal_id": "animal_id",
                "cond": "condition",
                "condition": "condition",
                "date": "date",
                "added": "date_added",
                "date_added": "date_added",
                "modified": "date_modified",
                "date_modified": "date_modified",
                "experiment": "experiment_id",
                "exp": "experiment_id",
                "status": "status",
            }

            def qualifier_values(value: str) -> list[str]:
                """Return the OR alternatives encoded by the filter dialog."""
                if value.startswith("(") and value.endswith(")"):
                    return [part for part in value[1:-1].split("|") if part]
                return [value]

            # Token match helper includes status based on metadata or column text
            def token_matches(meta: dict, name_text: str, token: str, ds_data: dict, status_text: str) -> bool:
                t = token.strip()
                if not t:
                    return True
                if t.startswith('"') and t.endswith('"'):
                    phrase = norm(t[1:-1])
                    fields = [norm(name_text)] + [
                        norm(meta.get(k)) for k in ("id", "formatted_name", "animal_id", "condition", "date", "date_added", "date_modified")
                    ]
                    # include status column text and experiment id for phrase matching
                    fields.append(norm(status_text))
                    # include experiment id/name for phrase matching
                    fields.append(norm(ds_data.get("experiment_id")))
                    return any(phrase in v for v in fields)
                if ":" in t:
                    k, v = t.split(":", 1)
                    fkey = key_map.get(k.lower())
                    if not fkey:
                        tt = norm(t)
                        fields = [norm(name_text)] + [
                            norm(meta.get(x)) for x in ("id", "formatted_name", "animal_id", "condition", "date", "date_added", "date_modified")
                        ]
                        fields.append(norm(status_text))
                        fields.append(norm(ds_data.get("experiment_id")))
                        return any(tt in f for f in fields)
                    values = [norm(value) for value in qualifier_values(v)]
                    # experiment qualifier matches against dataset's experiment_id
                    if fkey == "experiment_id":
                        field_value = norm(ds_data.get("experiment_id"))
                        return any(value in field_value for value in values)
                    # status qualifier should match meta status or column text
                    if fkey == "status":
                        mv = norm(meta.get("status"))
                        sv = norm(status_text)
                        return any(value in mv or value in sv for value in values)
                    field_value = norm(meta.get(fkey))
                    return any(value in field_value for value in values)
                tt = norm(t)
                fields = [norm(name_text)] + [
                    norm(meta.get(x)) for x in ("id", "formatted_name", "animal_id", "condition", "date", "date_added", "date_modified")
                ]
                fields.append(norm(status_text))
                fields.append(norm(ds_data.get("experiment_id")))
                return any(tt in f for f in fields)

            any_visible_overall = False
            for i in range(self.dataset_tree.topLevelItemCount()):
                exp_item = self.dataset_tree.topLevelItem(i)
                any_child_visible = False
                # Evaluate experiment-level matching
                exp_data = exp_item.data(0, Qt.ItemDataRole.UserRole) or {}
                exp_meta = exp_data.get("metadata") or {}
                exp_name = exp_item.text(0) or ""
                exp_cols = [
                    exp_name,
                    exp_item.text(1) or "",
                    exp_item.text(2) or "",
                    exp_item.text(3) or "",
                    exp_item.text(4) or "",
                ]

                def exp_token_matches(token: str, exp_meta: dict, exp_name: str) -> bool:
                    t = token.strip()
                    if not t:
                        return True

                    def get_exp_field(key: str):
                        # Map qualifiers to experiment metadata fields
                        kmap = {
                            "name": "id",
                            "id": "id",
                            "experiment": "id",
                            "exp": "id",
                            "path": "path",
                            "date": "date",  # if present
                            "added": "date_added",
                            "date_added": "date_added",
                            "modified": "date_modified",
                            "date_modified": "date_modified",
                        }
                        if key == "status":
                            return "Complete" if exp_meta.get("is_completed") else "Incomplete"
                        fk = kmap.get(key)
                        return exp_meta.get(fk) if fk else None

                    if t.startswith('"') and t.endswith('"'):
                        phrase = norm(t[1:-1])
                        fields = [norm(exp_name)] + [norm(exp_meta.get(k)) for k in ("id", "path", "date", "date_added", "date_modified")]
                        return any(phrase in v for v in fields)
                    if ":" in t:
                        k, v = t.split(":", 1)
                        values = [norm(value) for value in qualifier_values(v)]
                        fv = norm(get_exp_field(k.lower()))
                        if fv:
                            return any(value in fv for value in values)
                        # Unknown qualifier: treat as plain token
                        tt = norm(t)
                        fields = [norm(exp_name)] + [norm(exp_meta.get(x)) for x in ("id", "path", "date", "date_added", "date_modified")]
                        return any(tt in f for f in fields)
                    # Plain token
                    tt = norm(t)
                    fields = [norm(exp_name)] + [norm(exp_meta.get(x)) for x in ("id", "path", "date", "date_added", "date_modified")]
                    return any(tt in f for f in fields)

                exp_visible = True if not tokens else all(exp_token_matches(t, exp_meta, exp_name) for t in tokens)
                # Build highlighted HTML per column for experiment
                if q:
                    exp_html_cols = self._build_highlight_html(exp_cols, tokens)
                    for col in range(min(5, len(exp_html_cols))):
                        exp_item.setData(col, Qt.ItemDataRole.UserRole + 3, exp_html_cols[col])
                else:
                    for col in range(5):
                        exp_item.setData(col, Qt.ItemDataRole.UserRole + 3, None)
                # Highlighting removed; no HTML injection
                for j in range(exp_item.childCount()):
                    ds_item = exp_item.child(j)
                    ds_data = ds_item.data(0, Qt.ItemDataRole.UserRole) or {}
                    meta = ds_data.get("metadata", {})
                    name_text = ds_item.text(0) or ""
                    # Prefer status from metadata; fallback to column text
                    status_text = ds_item.text(2) or ""
                    # Merge status into meta for downstream checks if missing
                    if not meta.get("status") and status_text:
                        meta = dict(meta)
                        meta["status"] = status_text
                    visible = True if not tokens else all(token_matches(meta, name_text, t, ds_data, status_text) for t in tokens)
                    ds_item.setHidden(not visible)
                    # Highlighting removed; no HTML injection
                    if visible:
                        any_child_visible = True

                # If an experiment qualifier is present and matches, show all its datasets
                has_experiment_token = any(t.split(":", 1)[0].lower() in ("experiment", "exp") for t in tokens if ":" in t)
                if has_experiment_token and exp_visible:
                    for j in range(exp_item.childCount()):
                        ds_item = exp_item.child(j)
                        ds_item.setHidden(False)
                    any_child_visible = True

                # Keep experiment visible if it matches or has visible children
                exp_item.setHidden(not (exp_visible or any_child_visible))
                if exp_visible or any_child_visible:
                    any_visible_overall = True

            # After filtering, clear selection and details if selection hidden
            sels = self.dataset_tree.selectedItems()
            if sels:
                sel = sels[0]
                if sel.isHidden():
                    self.dataset_tree.clearSelection()
                    self.on_tree_selection_changed()

            # Update summary status for active filter
            if q:
                if not any_visible_overall:
                    self.dataset_summary.setText("No results for filter")
                else:
                    visible_count = 0
                    for i in range(self.dataset_tree.topLevelItemCount()):
                        exp_item = self.dataset_tree.topLevelItem(i)
                        for j in range(exp_item.childCount()):
                            ds_item = exp_item.child(j)
                            if not ds_item.isHidden():
                                visible_count += 1
                    self.dataset_summary.setText(f"Filter active: {visible_count} dataset(s) visible")

        except Exception as e:
            logger.error(f"Failed to filter dataset tree: {e}")

    def _build_highlight_html(self, texts, tokens):
        """Return a list of HTML strings with matched substrings highlighted.

        tokens: already-parsed tokens including quoted phrases and qualifiers; qualifiers highlight their value.
        """
        try:

            def norm(s):
                return str(s or "")

            # Extract highlight words from tokens (handle quoted and key:value)
            highlights = []
            for t in tokens:
                t = t.strip()
                if not t:
                    continue
                if t.startswith('"') and t.endswith('"'):
                    highlights.append(t[1:-1])
                elif ":" in t:
                    _, v = t.split(":", 1)
                    if v:
                        highlights.append(v)
                else:
                    highlights.append(t)

            def highlight_text(text):
                s = norm(text)
                if not s or not highlights:
                    return s
                # Build case-insensitive replacements with span; avoid overlapping by scanning
                lower = s.lower()
                ranges = []
                for h in highlights:
                    hl = str(h).lower()
                    start = 0
                    while True:
                        idx = lower.find(hl, start)
                        if idx == -1:
                            break
                        ranges.append((idx, idx + len(hl)))
                        start = idx + len(hl)
                # Merge overlapping ranges
                ranges.sort()
                merged = []
                for st, en in ranges:
                    if not merged or st > merged[-1][1]:
                        merged.append((st, en))
                    else:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], en))
                # Build HTML
                out = []
                last = 0
                for st, en in merged:
                    out.append(s[last:st])
                    span = s[st:en]
                    out.append(f"<span style='background-color:#ffec99;color:#000;'>{span}</span>")
                    last = en
                out.append(s[last:])
                return "".join(out)

            return [highlight_text(t) for t in texts]
        except Exception:
            return texts

    def on_dataset_checkbox_changed(self, item, column):
        """Handle dataset checkbox changes in the tree."""
        # Only respond to checkbox changes on column 0 and only for dataset items (not experiments)
        if column != 0 or not item.parent():
            return

        # Update button states efficiently
        self._update_button_states()

    def _count_selected_datasets(self):
        """Efficiently count selected datasets."""
        count = 0
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                if ds_item.checkState(0) == Qt.CheckState.Checked:
                    count += 1
        return count

    def _update_button_states(self):
        """Update button states based on current selections without triggering signals."""
        selected_count = self._count_selected_datasets()

        has_selection = selected_count > 0
        self.move_selected_button.setEnabled(has_selection)
        self.copy_selected_button.setEnabled(has_selection)
        self.delete_selected_button.setEnabled(has_selection)
        self.validate_selected_button.setEnabled(has_selection)
        self.exclude_selected_button.setEnabled(has_selection)
        self.restore_selected_button.setEnabled(has_selection)
        self.mark_complete_button.setEnabled(has_selection)
        self.mark_incomplete_button.setEnabled(has_selection)

        # Update summary
        if has_selection:
            self.dataset_summary.setText(f"{selected_count} dataset(s) selected")
        else:
            self.dataset_summary.setText("No datasets selected")

        # Refresh Undo Last button state & tooltip
        try:
            self._refresh_undo_last_button()
        except Exception:
            # Non-fatal; ensure UI still updates
            logger.warning("Failed to refresh Undo Last button state; continuing without update.")

    def _refresh_undo_last_button(self):
        """Enable/disable the Undo Last button and set its tooltip to the last command name."""
        try:
            btn = getattr(self, "undo_last_button", None)
            if btn is None:
                return

            if not self.session_commands:
                btn.setEnabled(False)
                btn.setToolTip("No changes to undo")
                return

            # Last command is the most recently appended
            last_cmd = self.session_commands[-1]
            # Prefer a human-friendly command_name attribute if present
            cmd_name = getattr(last_cmd, "command_name", None) or last_cmd.__class__.__name__
            btn.setEnabled(True)
            btn.setToolTip(f"Undo last: {cmd_name}")
        except Exception as e:
            logger.error(f"Failed to refresh Undo Last button: {e}")

    @auto_refresh
    def on_dataset_dragged(self, dataset_id, formatted_name, source_exp_id, target_exp_id):
        """Handle dataset drag-and-drop operations."""
        # If we're in a batch move, accumulate move descriptors and defer execution until batch completion.
        if getattr(self, "_in_batch_move", False):
            try:
                self._pending_batch_moves.append((dataset_id, formatted_name, source_exp_id, target_exp_id))
                # Track processed moves for batch reporting
                try:
                    self._batch_processed_moves += 1
                except Exception:
                    self._batch_processed_moves = 1
            except Exception as e:
                logger.error(f"Failed to buffer pending dataset move: {e}")
            return

        # Execute the move command but do not announce success here; announcements occur after the full batch.
        try:
            from monstim_gui.commands import MoveDatasetCommand

            # Execute the move command
            command = MoveDatasetCommand(self.gui, dataset_id, formatted_name, source_exp_id, target_exp_id)

            command.execute()
            self.session_commands.append(command)
            self._changes_made = True

            # Track successful move for batch summary
            try:
                self._batch_successful_moves += 1
            except Exception:
                self._batch_successful_moves = 1

        except Exception as e:
            logger.error(f"Failed to move dataset via drag-and-drop: {e}")
            # Track processed move even on failure for accurate batch reporting
            try:
                self._batch_processed_moves += 1
            except Exception:
                self._batch_processed_moves = 1
            QMessageBox.critical(self, "Move Failed", f"Failed to move dataset '{formatted_name}':\n{e!s}")

    @auto_refresh
    def create_experiment(self):
        """Create a new empty experiment immediately."""
        try:
            from PySide6.QtWidgets import QInputDialog

            from monstim_gui.commands import CreateExperimentCommand

            name, ok = QInputDialog.getText(self, "Create Experiment", "Enter experiment name:")
            if ok and name.strip():
                name = name.strip()

                # Check for naming conflicts
                if name in self.gui.expts_dict_keys:
                    QMessageBox.warning(self, "Name Conflict", f"An experiment named '{name}' already exists.")
                    return

                # Execute command immediately
                command = CreateExperimentCommand(self.gui, name)
                try:
                    command.execute()
                    self.session_commands.append(command)
                    self._changes_made = True  # Mark that changes were made
                    QMessageBox.information(self, "Experiment Created", f"Empty experiment '{name}' has been created successfully.")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to create experiment:\n{e!s}")
                    raise  # Let decorator handle the refresh

        except Exception as e:
            logger.error(f"Error in create_experiment: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create experiment:\n{e!s}")
            raise  # Let decorator handle the refresh

    @auto_refresh
    def import_experiment(self):
        """Import experiment using existing functionality (minimal wrapper).

        This uses the application's DataManager.import_expt_data() method.
        All data changes auto-save via the command pattern.
        """

        # Helper to disable/enable dataset UI during long operations
        def _set_dataset_ui_enabled(enabled: bool):
            try:
                self.dataset_tree.setEnabled(enabled)
            except Exception:
                # Ignore errors if the widget is missing or in a transient state; safe to skip in UI enable/disable.
                logger.warning("Failed to set dataset_tree enabled state; widget may be missing or deleted.")

            for btn_name in (
                "create_blank_experiment_button",
                "import_new_experiment_button",
                "validate_selected_button",
                "exclude_selected_button",
                "restore_selected_button",
                "mark_complete_button",
                "mark_incomplete_button",
                "select_all_button",
                "clear_selection_button",
                "move_selected_button",
                "copy_selected_button",
                "delete_selected_button",
            ):
                btn = getattr(self, btn_name, None)
                if btn is not None:
                    try:
                        btn.setEnabled(enabled)
                    except Exception as e:
                        # It is safe to ignore errors here (e.g., widget may be deleted during teardown),
                        # but log them for debugging purposes.
                        logger.error(f"Failed to set enabled state for {btn_name}: {e}")

        # Disable dataset UI to 'pause' dataset management while importing
        _set_dataset_ui_enabled(False)
        from PySide6.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # Suppress auto-refresh until import completes (may run in background thread)
        self._suppress_autorefresh = True

        # Data changes auto-save via command pattern, proceed with import

        # Use existing import functionality
        try:
            self.gui.data_manager.import_expt_data()
        except Exception as e:
            logger.error(f"Failed to start import: {e}")
            QMessageBox.critical(self, "Import Error", f"Failed to start import:\n{e!s}")
            # Restore UI and cursor
            try:
                QApplication.restoreOverrideCursor()
            except Exception as e:
                logger.error(f"Failed to restore cursor after import error: {e}")
            _set_dataset_ui_enabled(True)
            self._suppress_autorefresh = False
            raise

        # If importation started a background thread, re-enable UI only after it finishes
        dm = getattr(self.gui, "data_manager", None)
        thread = getattr(dm, "thread", None) if dm is not None else None

        def _finish_import():
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                # Ignore errors restoring cursor; not critical if cursor was not set
                logger.warning("Failed to restore cursor after import; cursor state may be inconsistent.")
            try:
                _set_dataset_ui_enabled(True)
            except Exception as e:
                # Non-critical: UI may already be disabled or window closed. Log for debugging.
                logger.error("Failed to re-enable dataset UI: %s", e)
            # clear suppression and perform a single refresh
            self._suppress_autorefresh = False
            try:
                self.load_data()
            except Exception:
                try:
                    self.update_dataset_tree()
                except Exception as e:
                    # Suppress all errors here to avoid crashing the UI; log for diagnostics.
                    logger.warning(f"Failed to update dataset tree after import: {e}")

        if thread is not None and getattr(thread, "isRunning", lambda: False)():
            # Connect finish handlers
            try:
                thread.finished.connect(_finish_import)
                thread.canceled.connect(_finish_import)
            except Exception:
                # If connecting fails, finish immediately
                _finish_import()
        else:
            # No background thread started; finish now
            _finish_import()

    def select_all_datasets(self):
        """Select all datasets in the tree."""
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                ds_item.setCheckState(0, Qt.CheckState.Checked)

    def clear_dataset_selection(self):
        """Clear all dataset selections."""
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                ds_item.setCheckState(0, Qt.CheckState.Unchecked)

    def _selected_dataset_data(self):
        """Return metadata for checked dataset rows."""
        selected = []
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                if ds_item.checkState(0) == Qt.CheckState.Checked:
                    data = ds_item.data(0, Qt.ItemDataRole.UserRole)
                    if data and data.get("type") == "dataset":
                        selected.append(data)
        return selected

    def validate_selected_datasets(self):
        """Check selected datasets without loading their recordings into the GUI."""
        selected = self._selected_dataset_data()
        if not selected:
            return

        from monstim_signals.io.repositories import DatasetRepository

        failures = []
        for data in selected:
            metadata = data["metadata"]
            path = Path(metadata["path"])
            try:
                if not path.is_dir():
                    raise FileNotFoundError("dataset folder is missing")
                DatasetRepository(path).get_metadata()
                if not any(child.is_dir() for child in path.iterdir()):
                    raise ValueError("no session folders found")
            except Exception as exc:
                failures.append(f"{metadata.get('formatted_name', path.name)}: {exc!s}")

        if failures:
            QMessageBox.warning(
                self,
                "Validation Found Issues",
                f"{len(failures)} of {len(selected)} dataset(s) need attention:\n\n" + "\n".join(failures),
            )
        else:
            QMessageBox.information(self, "Validation Complete", f"All {len(selected)} selected dataset(s) passed validation.")

    @auto_refresh
    def set_selected_datasets_included(self, include):
        """Batch include or exclude checked datasets using undoable commands."""
        from monstim_gui.commands import BatchCommand, ToggleDatasetInclusionCommand

        selected = self._selected_dataset_data()
        commands = []
        for data in selected:
            metadata = data["metadata"]
            commands.append(
                ToggleDatasetInclusionCommand(
                    self.gui,
                    data["experiment_id"],
                    metadata["id"],
                    exclude=not include,
                )
            )
        if commands:
            action = "Restore" if include else "Exclude"
            command = BatchCommand(f"{action} {len(commands)} dataset(s)", commands)
            command.execute()
            self.session_commands.append(command)
            self._changes_made = True

    @auto_refresh
    def set_selected_datasets_completion(self, completed):
        """Set completion on checked datasets without changing child-session annotations."""
        from monstim_gui.commands import BatchCommand, ToggleCompletionStatusCommand

        commands = []
        for data in self._selected_dataset_data():
            metadata = data["metadata"]
            if bool(metadata.get("is_completed")) == completed:
                continue
            target = SimpleNamespace(id=metadata["id"], is_completed=metadata.get("is_completed", False))
            command = ToggleCompletionStatusCommand(
                self.gui,
                "dataset",
                target,
                experiment_id=data["experiment_id"],
                new_status=completed,
                dataset_path=Path(metadata["path"]),
            )
            commands.append(command)

        if commands:
            action = "Mark Complete" if completed else "Mark Incomplete"
            command = BatchCommand(f"{action} for {len(commands)} dataset(s)", commands)
            command.execute()
            self.session_commands.append(command)
            self._changes_made = True

    @auto_refresh
    def move_selected_datasets(self):
        """Move selected datasets to another experiment immediately."""
        from PySide6.QtWidgets import QInputDialog

        from monstim_gui.commands import MoveDatasetsCommand

        # Get selected datasets
        selected_datasets = []
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                if ds_item.checkState(0) == Qt.CheckState.Checked:
                    data = ds_item.data(0, Qt.ItemDataRole.UserRole)
                    selected_datasets.append(data)

        if not selected_datasets:
            return

        # Show target experiment selection dialog
        available_experiments = list(self.gui.expts_dict_keys)

        target_exp, ok = QInputDialog.getItem(self, "Move Datasets", "Select target experiment:", available_experiments, 0, False)

        if ok and target_exp:
            moves = [(data["metadata"]["id"], data["metadata"]["formatted_name"], data["experiment_id"], target_exp) for data in selected_datasets]
            command = MoveDatasetsCommand(self.gui, moves)
            try:
                command.execute()
                successful_moves = len(command._succeeded)
            except Exception as exc:
                QMessageBox.critical(self, "Move Failed", f"Failed to move selected datasets:\n{exc!s}")
                raise
            if successful_moves:
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Datasets Moved",
                    f"{successful_moves} dataset(s) moved to '{target_exp}' successfully.",
                )

    @auto_refresh
    def copy_selected_datasets(self):
        """Copy selected datasets to another experiment immediately."""
        from PySide6.QtWidgets import QInputDialog

        from monstim_gui.commands import BatchCommand, CopyDatasetCommand

        # Get selected datasets
        selected_datasets = []
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                if ds_item.checkState(0) == Qt.CheckState.Checked:
                    data = ds_item.data(0, Qt.ItemDataRole.UserRole)
                    selected_datasets.append(data)

        if not selected_datasets:
            return

        # Show target experiment selection dialog
        available_experiments = list(self.gui.expts_dict_keys)

        target_exp, ok = QInputDialog.getItem(self, "Copy Datasets", "Select target experiment:", available_experiments, 0, False)

        if ok and target_exp:
            commands = [
                CopyDatasetCommand(
                    self.gui,
                    data["metadata"]["id"],
                    data["metadata"]["formatted_name"],
                    data["experiment_id"],
                    target_exp,
                )
                for data in selected_datasets
            ]
            command = BatchCommand(f"Copy {len(commands)} dataset(s) to '{target_exp}'", commands)
            try:
                command.execute()
            except Exception as exc:
                QMessageBox.critical(self, "Copy Failed", f"Failed to copy selected datasets:\n{exc!s}")
                raise
            if commands:
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Datasets Copied",
                    f"{len(commands)} dataset(s) copied to '{target_exp}' successfully.",
                )

    @auto_refresh
    def delete_selected_datasets(self):
        """Delete selected datasets."""
        # Get selected datasets
        selected_datasets = []
        for i in range(self.dataset_tree.topLevelItemCount()):
            exp_item = self.dataset_tree.topLevelItem(i)
            for j in range(exp_item.childCount()):
                ds_item = exp_item.child(j)
                if ds_item.checkState(0) == Qt.CheckState.Checked:
                    data = ds_item.data(0, Qt.ItemDataRole.UserRole)
                    selected_datasets.append(data)

        if not selected_datasets:
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Dataset Deletion",
            f"Are you sure you want to delete {len(selected_datasets)} dataset(s)?\n\nThis will permanently remove the dataset files from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            from monstim_gui.commands import DeleteDatasetCommand

            successful_deletions = 0
            errors = []

            for ds_data in selected_datasets:
                try:
                    ds_metadata = ds_data["metadata"]
                    cmd = DeleteDatasetCommand(
                        self.gui,
                        ds_metadata["id"],
                        ds_metadata.get("formatted_name", ds_metadata["id"]),
                        ds_data["experiment_id"],
                    )
                    cmd.execute()
                    self.session_commands.append(cmd)
                    successful_deletions += 1
                except Exception as e:
                    errors.append(f"Failed to delete '{ds_metadata['formatted_name']}': {e!s}")

            if successful_deletions > 0:
                self._changes_made = True  # Mark that changes were made

            if errors:
                error_msg = "\n".join(errors)
                QMessageBox.warning(
                    self,
                    "Some Deletions Failed",
                    f"{successful_deletions} dataset(s) deleted successfully.\n\nErrors:\n{error_msg}",
                )
            else:
                QMessageBox.information(
                    self,
                    "Datasets Deleted",
                    f"{successful_deletions} dataset(s) deleted successfully.",
                )

    def show_dataset_context_menu(self, position):
        """Show context menu for dataset tree items."""

        item = self.dataset_tree.itemAt(position)
        if not item:
            return

        # Check if this is a dataset item (has parent) or experiment item (no parent)
        is_dataset = item.parent() is not None
        is_experiment = item.parent() is None

        menu = QMenu(self)

        if is_dataset:
            # Dataset-specific actions
            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data:
                move_action = menu.addAction("Move to...")
                move_action.triggered.connect(lambda: self.context_move_dataset(data))

                copy_action = menu.addAction("Copy to...")
                copy_action.triggered.connect(lambda: self.context_copy_dataset(data))

                menu.addSeparator()

                edit_metadata_action = menu.addAction("Edit Dataset Metadata...")
                edit_metadata_action.setStatusTip("Edit this dataset's date, animal ID, and condition")
                edit_metadata_action.triggered.connect(lambda: self.context_edit_dataset_metadata(data))

                duplicate_action = menu.addAction("Duplicate in Same Experiment")
                duplicate_action.triggered.connect(lambda: self.context_duplicate_dataset(data))

                menu.addSeparator()

                # Toggle inclusion/exclusion reflecting experiment annot
                parent_meta = item.parent().data(0, Qt.ItemDataRole.UserRole).get("metadata", {})
                excluded_list = parent_meta.get("excluded_datasets", [])
                is_excluded = data["metadata"].get("id") in excluded_list
                if is_excluded:
                    include_action = menu.addAction("Include Dataset")
                    include_action.triggered.connect(lambda: self.context_toggle_dataset_inclusion(data, True))
                else:
                    exclude_action = menu.addAction("Exclude Dataset")
                    exclude_action.triggered.connect(lambda: self.context_toggle_dataset_inclusion(data, False))

                menu.addSeparator()

                delete_action = menu.addAction("Delete Dataset")
                delete_action.triggered.connect(lambda: self.context_delete_dataset(data))

        elif is_experiment:
            # Experiment-specific actions
            exp_name = item.text(0)

            rename_action = menu.addAction("Rename Experiment")
            rename_action.triggered.connect(lambda: self.context_rename_experiment(exp_name))

            menu.addSeparator()

            delete_action = menu.addAction("Delete Experiment")
            delete_action.triggered.connect(lambda: self.context_delete_experiment(exp_name))

        # Show the menu
        if menu.actions():
            menu.exec(self.dataset_tree.mapToGlobal(position))

    @auto_refresh
    def context_move_dataset(self, dataset_data):
        """Move a single dataset via context menu immediately."""
        from PySide6.QtWidgets import QInputDialog

        from monstim_gui.commands import MoveDatasetCommand

        available_experiments = list(self.gui.expts_dict_keys)

        target_exp, ok = QInputDialog.getItem(self, "Move Dataset", "Select target experiment:", available_experiments, 0, False)

        if ok and target_exp:
            ds_metadata = dataset_data["metadata"]
            command = MoveDatasetCommand(self.gui, ds_metadata["id"], ds_metadata["formatted_name"], dataset_data["experiment_id"], target_exp)

            try:
                command.execute()
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Dataset Moved",
                    f"Dataset '{ds_metadata['formatted_name']}' has been moved to '{target_exp}'.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Move Failed", f"Failed to move dataset:\n{e!s}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def context_copy_dataset(self, dataset_data):
        """Copy a single dataset via context menu immediately."""
        from PySide6.QtWidgets import QInputDialog

        from monstim_gui.commands import CopyDatasetCommand

        available_experiments = list(self.gui.expts_dict_keys)

        target_exp, ok = QInputDialog.getItem(self, "Copy Dataset", "Select target experiment:", available_experiments, 0, False)

        if ok and target_exp:
            ds_metadata = dataset_data["metadata"]
            command = CopyDatasetCommand(self.gui, ds_metadata["id"], ds_metadata["formatted_name"], dataset_data["experiment_id"], target_exp)

            try:
                command.execute()
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Dataset Copied",
                    f"Dataset '{ds_metadata['formatted_name']}' has been copied to '{target_exp}'.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Copy Failed", f"Failed to copy dataset:\n{e!s}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def context_edit_dataset_metadata(self, dataset_data):
        """Edit one dataset's identity metadata through the standard undoable editor."""
        from monstim_gui.commands import EditDatasetMetadataCommand
        from monstim_gui.dialogs.dataset_metadata_editor import DatasetMetadataEditor
        from monstim_signals.io.repositories import DatasetRepository

        metadata = dataset_data["metadata"]
        try:
            dataset = DatasetRepository(Path(metadata["path"])).load(lazy_open_h5=True)
            dialog = DatasetMetadataEditor(dataset, parent=self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return

            command = EditDatasetMetadataCommand(
                gui=self.gui,
                dataset=dataset,
                old_date=dialog.old_date,
                new_date=dialog.new_date,
                old_animal_id=dialog.old_animal_id,
                new_animal_id=dialog.new_animal_id,
                old_condition=dialog.old_condition,
                new_condition=dialog.new_condition,
                old_folder_name=dialog.old_folder_name,
                new_folder_name=dialog.new_folder_name,
            )
            command.execute()
            self.session_commands.append(command)
            self._changes_made = True
        except Exception as exc:
            logger.exception("Failed to edit metadata for dataset '%s'", metadata.get("id"))
            QMessageBox.critical(self, "Metadata Update Failed", f"Failed to edit dataset metadata:\n{exc!s}")
            raise

    def context_duplicate_dataset(self, dataset_data):
        """Duplicate a dataset within the same experiment asynchronously."""
        from monstim_gui.commands import CopyDatasetCommand

        ds_metadata = dataset_data["metadata"]
        original_name = ds_metadata["formatted_name"]

        experiment_id = dataset_data["experiment_id"]
        command = CopyDatasetCommand(
            self.gui,
            ds_metadata["id"],
            ds_metadata["formatted_name"],
            experiment_id,
            experiment_id,
        )

        experiment_path = Path(self.gui.expts_dict[experiment_id])
        original_datasets = {item.name for item in experiment_path.iterdir() if item.is_dir()}

        def completed():
            try:
                command.finalize_copy(original_datasets)
                self.session_commands.append(command)
                self._changes_made = True
                QMessageBox.information(
                    self,
                    "Dataset Duplicated",
                    f"Dataset '{original_name}' was duplicated as '{command.copied_folder_name}'.",
                )
            except Exception as exc:
                QMessageBox.critical(self, "Duplication Failed", f"Failed to finalize dataset duplication:\n{exc!s}")

        def failed(message):
            QMessageBox.critical(self, "Duplication Failed", f"Failed to duplicate dataset:\n{message}")

        self.gui.data_manager.copy_dataset_async(
            ds_metadata["id"],
            ds_metadata["formatted_name"],
            experiment_id,
            experiment_id,
            None,
            completed,
            failed,
        )

    @auto_refresh
    def context_toggle_dataset_inclusion(self, dataset_data, include):
        """Toggle dataset inclusion/exclusion using a command."""
        from monstim_gui.commands import ToggleDatasetInclusionCommand

        ds_metadata = dataset_data["metadata"]
        try:
            cmd = ToggleDatasetInclusionCommand(
                self.gui,
                dataset_data["experiment_id"],
                ds_metadata["id"],
                exclude=not include,
            )
            cmd.execute()
            self.session_commands.append(cmd)
            self._changes_made = True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update dataset state:\n{e!s}")
            raise  # Let decorator handle the refresh

    @auto_refresh
    def context_delete_dataset(self, dataset_data):
        """Delete a single dataset via context menu using command."""
        from monstim_gui.commands import DeleteDatasetCommand

        ds_metadata = dataset_data["metadata"]

        reply = QMessageBox.question(
            self,
            "Confirm Dataset Deletion",
            f"Are you sure you want to delete dataset '{ds_metadata['formatted_name']}'?\n\n"
            "This will permanently remove the dataset files from disk.\n\n"
            "WARNING: This action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                cmd = DeleteDatasetCommand(
                    self.gui,
                    ds_metadata["id"],
                    ds_metadata.get("formatted_name", ds_metadata["id"]),
                    dataset_data["experiment_id"],
                )
                cmd.execute()
                self.session_commands.append(cmd)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Dataset Deleted",
                    f"Dataset '{ds_metadata['formatted_name']}' has been permanently deleted.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Deletion Failed", f"Failed to delete dataset:\n{e!s}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def context_rename_experiment(self, exp_name):
        """Rename an experiment via context menu."""
        from PySide6.QtWidgets import QInputDialog

        from monstim_gui.commands import RenameExperimentCommand

        new_name, ok = QInputDialog.getText(self, "Rename Experiment", f"Enter new name for experiment '{exp_name}':", text=exp_name)

        if ok and new_name.strip() and new_name.strip() != exp_name:
            new_name = new_name.strip()

            # Validate name for invalid directory characters
            invalid_chars = r'<>:"/\\|?*'
            found_invalid = [c for c in new_name if c in invalid_chars]
            if found_invalid:
                invalid_str = ", ".join(f"'{c}' ({ord(c)})" if c != "\\" else "'\\\\' (backslash)" for c in set(found_invalid))
                QMessageBox.warning(
                    self,
                    "Invalid Characters",
                    f"The experiment name contains invalid characters for a directory name:\n\n"
                    f"Invalid characters found: {invalid_str}\n\n"
                    f'The following characters are not allowed: < > : " / \\ | ? *',
                )
                return

            # Check for naming conflicts
            if new_name in self.gui.expts_dict_keys:
                QMessageBox.warning(self, "Name Conflict", f"An experiment named '{new_name}' already exists.")
                return

            # Execute command immediately
            command = RenameExperimentCommand(self.gui, exp_name, new_name)
            try:
                command.execute()
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(self, "Experiment Renamed", f"Experiment '{exp_name}' has been renamed to '{new_name}'.")
            except Exception as e:
                QMessageBox.critical(self, "Rename Error", f"Failed to rename experiment:\n{e!s}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def context_delete_experiment(self, exp_name):
        """Delete an experiment via context menu immediately."""
        from monstim_gui.commands import DeleteExperimentCommand

        reply = QMessageBox.question(
            self,
            "Confirm Experiment Deletion",
            f"Are you sure you want to delete experiment '{exp_name}'?\n\n"
            "This will permanently remove all datasets and files in this experiment.\n\n"
            "WARNING: This action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            command = DeleteExperimentCommand(self.gui, exp_name)
            try:
                command.execute()
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(self, "Experiment Deleted", f"Experiment '{exp_name}' has been permanently deleted.")
            except Exception as e:
                QMessageBox.critical(self, "Deletion Failed", f"Failed to delete experiment:\n{e!s}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def undo_last_change(self):
        """Undo the most recently recorded command and remove it from session_commands."""
        if not self.session_commands:
            QMessageBox.information(self, "Nothing to Undo", "There are no recorded changes to undo.")
            return

        # Pop the last command and attempt to undo it. If undo fails, re-add it.
        cmd = self.session_commands.pop()
        cmd_name = getattr(cmd, "command_name", None) or cmd.__class__.__name__
        try:
            cmd.undo()
            QMessageBox.information(self, "Undo Last Change", f"Undid last operation: {cmd_name}")

            # If no remaining commands, mark no outstanding changes
            self._changes_made = len(self.session_commands) > 0

        except Exception as e:
            # Re-insert to preserve history since undo failed
            try:
                self.session_commands.append(cmd)
            except Exception as append_error:
                # Failed to re-insert command into history; log and continue.
                logger.error(f"Failed to re-insert command into session_commands after undo failure: {append_error}")
            QMessageBox.critical(self, "Undo Failed", f"Failed to undo last change '{cmd_name}':\n{e!s}")
            # Re-raise to let auto_refresh ensure a refresh and to surface the error
            raise

    @auto_refresh
    def cancel_all_changes(self):
        """Undo all changes made during this session and keep the dialog open."""
        if not self.session_commands:
            QMessageBox.information(self, "No Changes", "There are no recorded changes to undo.")
            return

        reply = QMessageBox.question(
            self,
            "Undo All Changes",
            f"This will undo {len(self.session_commands)} operation(s) performed in this session.\n\nAre you sure you want to undo all changes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Undo all commands in reverse order
                for command in reversed(self.session_commands):
                    try:
                        command.undo()
                    except Exception as e:
                        logger.exception("Undo failed for a command during Undo All: %s", e)

                QMessageBox.information(self, "Changes Undone", f"All {len(self.session_commands)} operation(s) have been successfully undone.")

                # Clear command history so there are no duplicate undos later
                self.session_commands.clear()
                self._changes_made = False
                try:
                    self._refresh_undo_last_button()
                except Exception as e:
                    logger.exception("Failed to refresh undo/last button after undoing all changes: %s", e)

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Undoing Changes",
                    f"An error occurred while undoing changes:\n{e!s}\n\nSome changes may not have been fully reversed.",
                )
        # Keep dialog open for user to resolve

    def accept(self):
        """On close via Done, prompt to apply/undo if there are pending changes."""
        if getattr(self, "session_commands", None):
            choice = self._confirm_close_with_pending_changes()
            if choice == "cancel":
                return  # keep dialog open
            if choice == "undo":
                # Undo all then close
                try:
                    for cmd in reversed(self.session_commands):
                        try:
                            cmd.undo()
                        except Exception as e:
                            logger.exception("Undo failed during close: %s", e)
                    self.session_commands.clear()
                    self._changes_made = False
                except Exception as e:
                    # Suppress unexpected errors during undo to avoid crashing the dialog,
                    # but log them for diagnostics. User is not notified here because individual undo failures are already logged above.
                    logger.exception("Unexpected error during undo-all in accept(): %s", e)

        # Emit change signal if there were changes this session
        if getattr(self, "_changes_made", False):
            self.data_structure_changed.emit()

        super().accept()

    def reject(self):
        """Intercept window close to confirm pending changes before exiting."""
        if getattr(self, "session_commands", None):
            choice = self._confirm_close_with_pending_changes()
            if choice == "cancel":
                return  # abort close
            if choice == "undo":
                try:
                    for cmd in reversed(self.session_commands):
                        try:
                            cmd.undo()
                        except Exception as e:
                            logger.exception("Undo failed during close: %s", e)
                    self.session_commands.clear()
                    self._changes_made = False
                except Exception:
                    # Suppress unexpected errors during undo to avoid crashing the dialog,
                    # but log them for diagnostics. User is not notified here because individual undo failures are already logged above.
                    logger.exception("Unexpected error during undo-all in reject()")

        if getattr(self, "_changes_made", False):
            self.data_structure_changed.emit()

        super().reject()

    def _confirm_close_with_pending_changes(self) -> str:
        """Prompt the user when there are pending changes. Returns 'keep', 'undo', or 'cancel'."""
        try:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Pending Changes")
            msg.setText("You have changes from this session. What would you like to do?")
            keep_btn = msg.addButton("Keep Changes and Close", QMessageBox.ButtonRole.AcceptRole)
            undo_btn = msg.addButton("Undo All and Close", QMessageBox.ButtonRole.DestructiveRole)
            _ = msg.addButton(QMessageBox.StandardButton.Cancel)
            msg.exec()
            clicked = msg.clickedButton()
            if clicked == keep_btn:
                return "keep"
            if clicked == undo_btn:
                return "undo"
            return "cancel"
        except Exception:
            # Fallback: cancel if prompt fails
            return "cancel"
