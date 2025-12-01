"""
Data Curation Manager Dialog
Allows users to manage experiments and datasets with create/import/delete/rename operations
and drag-and-drop dataset organization between experiments.
"""

# TODOs: Data Curation Manager feature roadmap
# - Thumbnail / quick-preview column: show a small sparkline or thumbnail per-dataset
#   (representative session trace) to help visually triage datasets before moving/deleting.
# - Search / filter box: add a text input above the tree to filter experiments/datasets by
#   name, date, animal_id, condition, or tags.
# - Dry-run / preview modal for batch move/copy/delete: show affected items, conflicts,
#   and allow skip/overwrite/rename decisions before executing changes.
# - Multi-select drag & drop ghosting: when dragging multiple checked datasets, show a
#   drag ghost with count and optionally a preview to make multi-drag operations clear.
# - Batch rename tooling: pattern-based renaming (tokens or regex) with a preview before applying.
# - Recycle bin / safe-delete: move datasets to a hidden .trash instead of immediate permanent deletion,
#   add restore & purge UI, and schedule automatic purge after configurable retention.
# - Inline / bulk metadata editor: allow editing dataset metadata (date/animal/condition) from the tree
#   or an edit pane; support bulk edits and preview.
# - Validation & consistency checker: validate selected datasets for missing files, mismatched channels,
#   missing latency windows, etc., and provide quick-fix suggestions.
# - Background workers and progress/cancellation: run long operations (move/copy/import) in background
#   threads with progress dialogs and per-item error reporting.
# - Duplicate detection & merge assistant: find likely duplicate datasets and offer safe merge options.
# - Tagging and saved views: let users tag datasets and save filterable views for recurring workflows.

import logging
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QDrag, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI


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
                logging.error(f"Failed to refresh data after error in {method.__name__}: {refresh_error}")
            # Re-raise the original exception
            raise e

    return wrapper


class DatasetTreeWidget(QTreeWidget):
    """Custom QTreeWidget that handles dataset drag-and-drop operations."""

    dataset_moved = pyqtSignal(str, str, str, str)  # dataset_id, formatted_name, source_exp, target_exp
    dataset_move_batch_start = pyqtSignal()
    dataset_moved_batch = pyqtSignal(int, str)  # count, target_exp_id

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
                pass
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
                    pass

        if checked_items:
            items = checked_items
        else:
            # Use selection as a fallback
            items = [it for it in self.selectedItems() if it and it.parent()]

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
                mime = self.model().mimeData([idx for idx in self.selectedIndexes()])
                drag.setMimeData(mime)
            except Exception:
                pass

        if not pix.isNull():
            drag.setPixmap(pix)
            try:
                # Place the pixmap's top-left corner at the cursor click point
                drag.setHotSpot(pix.rect().topLeft())
            except Exception as e:
                logging.debug("Failed to set drag hotspot for pixmap: %r", e)
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
                except Exception:
                    pass
            elif pos.y() > vp_rect.bottom() - margin:
                try:
                    vsb.setValue(vsb.value() + scroll_amount_v)
                except Exception:
                    pass

            # Horizontal scrolling (if needed)
            if pos.x() < vp_rect.left() + margin:
                try:
                    hsb.setValue(hsb.value() - scroll_amount_h)
                except Exception:
                    pass
            elif pos.x() > vp_rect.right() - margin:
                try:
                    hsb.setValue(hsb.value() + scroll_amount_h)
                except Exception:
                    pass
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
        if drop_item.parent():  # Dropped on a dataset
            target_exp_item = drop_item.parent()
        else:  # Dropped on an experiment
            target_exp_item = drop_item

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
            pass

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
                            items_info.append(
                                (ds_meta.get("id", ""), ds_meta.get("formatted_name", ""), data.get("experiment_id"))
                            )
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


class DataCurationManager(QDialog):
    """
    Modal dialog for comprehensive data curation including experiment and dataset management.
    Uses preview/apply pattern with separate tabs for different operations.
    """

    data_structure_changed = pyqtSignal()  # Signal emitted when data structure changes

    def __init__(self, parent: "MonstimGUI"):
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
            logging.error(f"Failed to initialize Data Curation Manager: {e}")
            QMessageBox.critical(self, "Initialization Error", f"Failed to initialize dialog:\n{str(e)}")
            raise

    def setup_ui(self):
        """Set up the dialog UI with tabbed interface."""
        self.setWindowTitle("Data Curation Manager")
        self.setModal(True)
        self.resize(1000, 700)

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

        # TODO: Search/filter UI
        # - Add a QLineEdit here to filter the tree by dataset/experiment name, date,
        #   animal_id, condition, or tags. Implement tokenized matching and optionally
        #   highlight matched substrings in the tree.

        # Header
        header_label = QLabel("Data Management")
        header_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(header_label)

        # Instructions
        instructions = QLabel(
            "Right-click to access context menus. Drag datasets between experiments to reorganize your data structure. Use checkboxes for batch operations."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Batch operations
        batch_layout = QHBoxLayout()

        # Quick experiment creation
        self.create_blank_experiment_button = QPushButton("Create Blank Experiment")
        self.create_blank_experiment_button.clicked.connect(self.create_experiment)
        self.create_blank_experiment_button.setToolTip("Create a new empty experiment for organizing datasets")
        batch_layout.addWidget(self.create_blank_experiment_button)

        # Import new experiment button (to the right of Create Blank Experiment)
        self.import_new_experiment_button = QPushButton("Import New Experiment")
        self.import_new_experiment_button.clicked.connect(self.import_experiment)
        self.import_new_experiment_button.setToolTip("Import a new experiment using the standard import workflow")
        self.import_new_experiment_button.setMaximumHeight(self.create_blank_experiment_button.maximumHeight())
        batch_layout.addWidget(self.import_new_experiment_button)

        # TODO: Batch operations - add buttons for additional curation tools
        # - Batch Rename
        # - Validate Selected
        # - Move to Trash / Restore (Trash view)
        # These should be inserted here aligned with the existing batch controls.

        batch_layout.addStretch()

        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all_datasets)
        batch_layout.addWidget(self.select_all_button)

        self.clear_selection_button = QPushButton("Clear Selection")
        self.clear_selection_button.clicked.connect(self.clear_dataset_selection)
        batch_layout.addWidget(self.clear_selection_button)

        batch_layout.addStretch()

        self.move_selected_button = QPushButton("Move Selected To...")
        self.move_selected_button.clicked.connect(self.move_selected_datasets)
        self.move_selected_button.setEnabled(False)
        batch_layout.addWidget(self.move_selected_button)

        self.copy_selected_button = QPushButton("Copy Selected To...")
        self.copy_selected_button.clicked.connect(self.copy_selected_datasets)
        self.copy_selected_button.setEnabled(False)
        batch_layout.addWidget(self.copy_selected_button)

        self.delete_selected_button = QPushButton("Delete Selected")
        self.delete_selected_button.clicked.connect(self.delete_selected_datasets)
        self.delete_selected_button.setEnabled(False)
        batch_layout.addWidget(self.delete_selected_button)

        layout.addLayout(batch_layout)

        # Tree widget for hierarchical display
        self.dataset_tree = DatasetTreeWidget()
        self.dataset_tree.setHeaderLabels(["Experiment / Dataset", "Sessions", "Status"])
        self.dataset_tree.itemChanged.connect(self.on_dataset_checkbox_changed)
        self.dataset_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.dataset_tree.customContextMenuRequested.connect(self.show_dataset_context_menu)
        self.dataset_tree.dataset_moved.connect(self.on_dataset_dragged)
        self.dataset_tree.dataset_moved_batch.connect(self.on_dataset_dragged_batch)
        self.dataset_tree.dataset_move_batch_start.connect(self.on_dataset_drag_start)

        # Configure column widths - make first column stretch to fill available space
        header = self.dataset_tree.header()
        # header.setStretchLastSection(False)  # Don't auto-stretch the last column
        header.setSectionResizeMode(0, header.ResizeMode.ResizeToContents)  # Experiment/Dataset column
        header.setSectionResizeMode(1, header.ResizeMode.ResizeToContents)  # Sessions column

        layout.addWidget(self.dataset_tree)

        # Summary area
        self.dataset_summary = QLabel("No pending changes")
        self.dataset_summary.setStyleSheet("border: 1px solid gray; padding: 5px;")
        layout.addWidget(self.dataset_summary)

        # TODO: Consider adding a small 'Views' / 'Saved Filters' pane here in the future
        # to let users save common filter criteria or tag-based views for fast access.

        return tab_widget

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
            from PyQt6.QtWidgets import QApplication

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        except Exception:
            pass

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
                    logging.error(f"Batched move command failed: {e}")
                    QMessageBox.critical(self, "Move Failed", f"Failed to move datasets:\n{str(e)}")

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
                    from PyQt6.QtWidgets import QApplication

                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass

        # Refresh dataset tree now that batch operations are done
        try:
            self.load_data()
        except Exception:
            # fallback to explicit update
            try:
                self.update_dataset_tree()
            except Exception:
                pass

    def load_data(self):
        """Load current experiment and dataset data."""
        try:
            logging.debug("Data Curation Manager: Loading data...")
            logging.debug(f"Available experiments: {list(self.gui.expts_dict_keys)}")
            logging.debug(f"Experiment paths: {dict(self.gui.expts_dict)}")

            # Store original state
            self.original_experiments = dict(self.gui.expts_dict)

            # Populate experiment tree
            self.update_dataset_tree()

            # Initialize button states
            self._update_button_states()

            logging.debug("Data Curation Manager: Data loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load data in Data Curation Manager: {e}")
            QMessageBox.critical(self, "Data Loading Error", f"Failed to load experiment data:\n{str(e)}")

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
            logging.debug("Updating dataset tree...")

            # Save current state before clearing
            expansion_state, selected_datasets = self._save_tree_expansion_state()

            self.dataset_tree.clear()

            tree_items = 0
            for exp_id in self.gui.expts_dict_keys:
                try:
                    exp_path = Path(self.gui.expts_dict[exp_id])

                    # Get experiment metadata using repository method
                    exp_metadata = self._get_experiment_metadata(exp_path)

                    # Create experiment node
                    dataset_count = exp_metadata.get("dataset_count", 0)
                    exp_item = QTreeWidgetItem([exp_id, f"{dataset_count} datasets", ""])
                    exp_item.setData(
                        0, Qt.ItemDataRole.UserRole, {"type": "experiment", "id": exp_id, "metadata": exp_metadata}
                    )
                    exp_item.setFlags(exp_item.flags() | Qt.ItemFlag.ItemIsDropEnabled)

                    # Add dataset children using metadata
                    for ds_metadata in exp_metadata.get("datasets", []):
                        ds_name = ds_metadata.get("formatted_name", ds_metadata.get("id", "Unknown"))
                        session_count = ds_metadata.get("session_count", 0)
                        status = "Complete" if ds_metadata.get("is_completed", False) else "Incomplete"
                        is_excluded = ds_metadata.get("id") in exp_metadata.get("excluded_datasets", [])
                        if is_excluded:
                            status = f"Excluded ({status})"

                        ds_item = QTreeWidgetItem([ds_name, f"{session_count} sessions", status])
                        ds_item.setData(
                            0, Qt.ItemDataRole.UserRole, {"type": "dataset", "experiment_id": exp_id, "metadata": ds_metadata}
                        )
                        ds_item.setFlags(
                            ds_item.flags()
                            | Qt.ItemFlag.ItemIsUserCheckable
                            | Qt.ItemFlag.ItemIsDragEnabled
                            | Qt.ItemFlag.ItemIsSelectable
                        )
                        ds_item.setCheckState(0, Qt.CheckState.Unchecked)

                        # Light styling for excluded datasets: italic + gray text
                        if is_excluded:
                            gray = QBrush(QColor(170, 170, 170))  # a soft gray
                            for col in range(3):
                                ds_item.setForeground(col, gray)
                                f = ds_item.font(col) if hasattr(ds_item, "font") else QFont()
                                f.setItalic(True)
                                ds_item.setFont(col, f)

                        exp_item.addChild(ds_item)

                    self.dataset_tree.addTopLevelItem(exp_item)
                    tree_items += 1
                    logging.debug(f"Added experiment to tree: {exp_id} with {len(exp_metadata.get('datasets', []))} datasets")

                except Exception as e:
                    logging.error(f"Failed to scan experiment '{exp_id}' metadata: {e}")
                    # Add a placeholder item for the failed experiment
                    exp_item = QTreeWidgetItem([exp_id, "Error loading", "Failed"])
                    exp_item.setData(0, Qt.ItemDataRole.UserRole, {"type": "experiment", "id": exp_id})
                    self.dataset_tree.addTopLevelItem(exp_item)
                    tree_items += 1

            # Restore expansion state and selections after all items are added
            self._restore_tree_expansion_state(expansion_state, selected_datasets)

            # Update button states after restoring selections (without triggering signals)
            self._update_button_states()

            logging.debug(f"Dataset tree updated with {tree_items} experiments")

        except Exception as e:
            logging.error(f"Failed to update dataset tree: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load dataset information:\n{str(e)}")

    def _get_experiment_metadata(self, exp_path: Path) -> dict:
        """Get lightweight metadata about an experiment using repository method."""
        try:
            from monstim_signals.io.repositories import ExperimentRepository

            exp_repo = ExperimentRepository(exp_path)
            return exp_repo.get_metadata()

        except Exception as e:
            logging.error(f"Failed to get experiment metadata for {exp_path.name}: {e}")
            return {
                "id": exp_path.name,
                "path": str(exp_path),
                "dataset_count": 0,
                "datasets": [],
                "is_completed": False,
                "error": str(e),
            }

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
            pass

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
            logging.error(f"Failed to refresh Undo Last button: {e}")

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
                logging.error(f"Failed to buffer pending dataset move: {e}")
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
            logging.error(f"Failed to move dataset via drag-and-drop: {e}")
            # Track processed move even on failure for accurate batch reporting
            try:
                self._batch_processed_moves += 1
            except Exception:
                self._batch_processed_moves = 1
            QMessageBox.critical(self, "Move Failed", f"Failed to move dataset '{formatted_name}':\n{str(e)}")

    @auto_refresh
    def create_experiment(self):
        """Create a new empty experiment immediately."""
        try:
            from PyQt6.QtWidgets import QInputDialog

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
                    QMessageBox.information(
                        self, "Experiment Created", f"Empty experiment '{name}' has been created successfully."
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to create experiment:\n{str(e)}")
                    raise  # Let decorator handle the refresh

        except Exception as e:
            logging.error(f"Error in create_experiment: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create experiment:\n{str(e)}")
            raise  # Let decorator handle the refresh

    @auto_refresh
    def import_experiment(self):
        """Import experiment using existing functionality (minimal wrapper).

        This uses the application's DataManager.import_expt_data() method and
        preserves the existing confirmation for unsaved changes.
        """

        # Helper to disable/enable dataset UI during long operations
        def _set_dataset_ui_enabled(enabled: bool):
            try:
                self.dataset_tree.setEnabled(enabled)
            except Exception:
                pass

            for btn_name in (
                "create_blank_experiment_button",
                "import_new_experiment_button",
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
                    except Exception:
                        pass

        # Disable dataset UI to 'pause' dataset management while importing
        _set_dataset_ui_enabled(False)
        from PyQt6.QtWidgets import QApplication

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        # Suppress auto-refresh until import completes (may run in background thread)
        self._suppress_autorefresh = True

        # Check for unsaved changes
        if getattr(self.gui, "has_unsaved_changes", False):
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes to data. Save them before importing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Use DataManager save if available
                try:
                    self.gui.data_manager.save_experiment()
                except Exception as e:
                    logging.error(f"Failed to save before import: {e}")
            elif reply == QMessageBox.StandardButton.Cancel:
                # Restore UI and cursor
                try:
                    QApplication.restoreOverrideCursor()
                except Exception:
                    pass
                _set_dataset_ui_enabled(True)
                self._suppress_autorefresh = False
                return

        # Use existing import functionality
        try:
            self.gui.data_manager.import_expt_data()
        except Exception as e:
            logging.error(f"Failed to start import: {e}")
            QMessageBox.critical(self, "Import Error", f"Failed to start import:\n{str(e)}")
            # Restore UI and cursor
            try:
                QApplication.restoreOverrideCursor()
            except Exception:
                pass
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
                pass
            try:
                _set_dataset_ui_enabled(True)
            except Exception:
                pass
            # clear suppression and perform a single refresh
            self._suppress_autorefresh = False
            try:
                self.load_data()
            except Exception:
                try:
                    self.update_dataset_tree()
                except Exception:
                    pass

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

    @auto_refresh
    def move_selected_datasets(self):
        """Move selected datasets to another experiment immediately."""
        from PyQt6.QtWidgets import QInputDialog

        from monstim_gui.commands import MoveDatasetCommand

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
        available_experiments = [exp_id for exp_id in self.gui.expts_dict_keys]

        target_exp, ok = QInputDialog.getItem(
            self, "Move Datasets", "Select target experiment:", available_experiments, 0, False
        )

        if ok and target_exp:
            # Execute move commands immediately
            successful_moves = 0
            for ds_data in selected_datasets:
                ds_metadata = ds_data["metadata"]
                command = MoveDatasetCommand(
                    self.gui, ds_metadata["id"], ds_metadata["formatted_name"], ds_data["experiment_id"], target_exp
                )
                try:
                    command.execute()
                    self.session_commands.append(command)
                    successful_moves += 1
                except Exception as e:
                    QMessageBox.warning(self, "Move Failed", f"Failed to move '{ds_metadata['formatted_name']}':\n{str(e)}")

            if successful_moves > 0:
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Datasets Moved",
                    f"{successful_moves} dataset(s) moved to '{target_exp}' successfully.",
                )

    @auto_refresh
    def copy_selected_datasets(self):
        """Copy selected datasets to another experiment immediately."""
        from PyQt6.QtWidgets import QInputDialog

        from monstim_gui.commands import CopyDatasetCommand

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
        available_experiments = [exp_id for exp_id in self.gui.expts_dict_keys]

        target_exp, ok = QInputDialog.getItem(
            self, "Copy Datasets", "Select target experiment:", available_experiments, 0, False
        )

        if ok and target_exp:
            # Execute copy commands immediately
            successful_copies = 0
            for ds_data in selected_datasets:
                ds_metadata = ds_data["metadata"]
                command = CopyDatasetCommand(
                    self.gui, ds_metadata["id"], ds_metadata["formatted_name"], ds_data["experiment_id"], target_exp
                )
                try:
                    command.execute()
                    self.session_commands.append(command)
                    successful_copies += 1
                except Exception as e:
                    QMessageBox.warning(self, "Copy Failed", f"Failed to copy '{ds_metadata['formatted_name']}':\n{str(e)}")

            if successful_copies > 0:
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Datasets Copied",
                    f"{successful_copies} dataset(s) copied to '{target_exp}' successfully.",
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
            f"Are you sure you want to delete {len(selected_datasets)} dataset(s)?\n\n"
            "This will permanently remove the dataset files from disk.",
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
                    errors.append(f"Failed to delete '{ds_metadata['formatted_name']}': {str(e)}")

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
        from PyQt6.QtWidgets import QInputDialog

        from monstim_gui.commands import MoveDatasetCommand

        available_experiments = [exp_id for exp_id in self.gui.expts_dict_keys]

        target_exp, ok = QInputDialog.getItem(
            self, "Move Dataset", "Select target experiment:", available_experiments, 0, False
        )

        if ok and target_exp:
            ds_metadata = dataset_data["metadata"]
            command = MoveDatasetCommand(
                self.gui, ds_metadata["id"], ds_metadata["formatted_name"], dataset_data["experiment_id"], target_exp
            )

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
                QMessageBox.critical(self, "Move Failed", f"Failed to move dataset:\n{str(e)}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def context_copy_dataset(self, dataset_data):
        """Copy a single dataset via context menu immediately."""
        from PyQt6.QtWidgets import QInputDialog

        from monstim_gui.commands import CopyDatasetCommand

        available_experiments = [exp_id for exp_id in self.gui.expts_dict_keys]

        target_exp, ok = QInputDialog.getItem(
            self, "Copy Dataset", "Select target experiment:", available_experiments, 0, False
        )

        if ok and target_exp:
            ds_metadata = dataset_data["metadata"]
            command = CopyDatasetCommand(
                self.gui, ds_metadata["id"], ds_metadata["formatted_name"], dataset_data["experiment_id"], target_exp
            )

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
                QMessageBox.critical(self, "Copy Failed", f"Failed to copy dataset:\n{str(e)}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def context_duplicate_dataset(self, dataset_data):
        """Duplicate a dataset within the same experiment immediately."""
        from PyQt6.QtWidgets import QInputDialog

        from monstim_gui.commands import CopyDatasetCommand

        ds_metadata = dataset_data["metadata"]
        original_name = ds_metadata["formatted_name"]

        new_name, ok = QInputDialog.getText(
            self, "Duplicate Dataset", f"Enter name for duplicate of '{original_name}':", text=f"{original_name}_copy"
        )

        if ok and new_name.strip():
            experiment_id = dataset_data["experiment_id"]
            command = CopyDatasetCommand(
                self.gui,
                ds_metadata["id"],
                ds_metadata["formatted_name"],
                experiment_id,
                experiment_id,
                new_name=new_name.strip(),
            )

            try:
                command.execute()
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self,
                    "Dataset Duplicated",
                    f"Dataset has been duplicated as '{new_name.strip()}'.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Duplication Failed", f"Failed to duplicate dataset:\n{str(e)}")
                raise  # Let decorator handle the refresh

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
            QMessageBox.critical(self, "Error", f"Failed to update dataset state:\n{str(e)}")
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
                QMessageBox.critical(self, "Deletion Failed", f"Failed to delete dataset:\n{str(e)}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def context_rename_experiment(self, exp_name):
        """Rename an experiment via context menu."""
        from PyQt6.QtWidgets import QInputDialog

        from monstim_gui.commands import RenameExperimentCommand

        new_name, ok = QInputDialog.getText(
            self, "Rename Experiment", f"Enter new name for experiment '{exp_name}':", text=exp_name
        )

        if ok and new_name.strip() and new_name.strip() != exp_name:
            new_name = new_name.strip()

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
                QMessageBox.information(
                    self, "Experiment Renamed", f"Experiment '{exp_name}' has been renamed to '{new_name}'."
                )
            except Exception as e:
                QMessageBox.critical(self, "Rename Error", f"Failed to rename experiment:\n{str(e)}")
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
                QMessageBox.critical(self, "Deletion Failed", f"Failed to delete experiment:\n{str(e)}")
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
            except Exception:
                pass
            QMessageBox.critical(self, "Undo Failed", f"Failed to undo last change '{cmd_name}':\n{str(e)}")
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
            f"This will undo {len(self.session_commands)} operation(s) performed in this session.\n\n"
            "Are you sure you want to undo all changes?",
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
                        logging.exception("Undo failed for a command during Undo All: %s", e)

                QMessageBox.information(
                    self, "Changes Undone", f"All {len(self.session_commands)} operation(s) have been successfully undone."
                )

                # Clear command history so there are no duplicate undos later
                self.session_commands.clear()
                self._changes_made = False
                try:
                    self._refresh_undo_last_button()
                except Exception:
                    pass

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Undoing Changes",
                    f"An error occurred while undoing changes:\n{str(e)}\n\nSome changes may not have been fully reversed.",
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
                            logging.exception("Undo failed during close: %s", e)
                    self.session_commands.clear()
                    self._changes_made = False
                except Exception:
                    pass

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
                            logging.exception("Undo failed during close: %s", e)
                    self.session_commands.clear()
                    self._changes_made = False
                except Exception:
                    pass

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
