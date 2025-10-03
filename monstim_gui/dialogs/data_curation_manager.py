"""
Data Curation Manager Dialog
Allows users to manage experiments and datasets with create/import/delete/rename operations
and drag-and-drop dataset organization between experiments.
"""

import logging
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
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
            # Only refresh if the method completed successfully
            self.load_data()
            return result
        except Exception as e:
            # If there was an error, still refresh to ensure UI consistency
            try:
                self.load_data()
            except Exception as refresh_error:
                logging.error(f"Failed to refresh data after error in {method.__name__}: {refresh_error}")
            # Re-raise the original exception
            raise e

    return wrapper


class DatasetTreeWidget(QTreeWidget):
    """Custom QTreeWidget that handles dataset drag-and-drop operations."""

    dataset_moved = pyqtSignal(str, str, str, str)  # dataset_id, formatted_name, source_exp, target_exp

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)

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

        # Get the dragged dataset items
        selected_items = self.selectedItems()
        for item in selected_items:
            item_data = item.data(0, Qt.ItemDataRole.UserRole)
            if item_data and item_data.get("type") == "dataset":
                source_exp_id = item_data.get("experiment_id")
                ds_metadata = item_data.get("metadata", {})

                # Don't move if it's already in the target experiment
                if source_exp_id != target_exp_id:
                    self.dataset_moved.emit(
                        ds_metadata.get("id", ""), ds_metadata.get("formatted_name", ""), source_exp_id, target_exp_id
                    )

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

        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Create tabs
        self.experiment_tab = self.create_experiment_management_tab()
        self.dataset_tab = self.create_dataset_management_tab()

        self.tab_widget.addTab(self.experiment_tab, "Experiment Management")
        self.tab_widget.addTab(self.dataset_tab, "Dataset Organization")

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.reset_button = QPushButton("Reset All")
        self.reset_button.clicked.connect(self.reset_all_changes)
        button_layout.addWidget(self.reset_button)

        # Add some spacing
        button_layout.addStretch()

        self.cancel_button = QPushButton("Undo All Changes")
        self.cancel_button.clicked.connect(self.cancel_all_changes)
        button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton("Done")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)

        main_layout.addLayout(button_layout)

    def create_experiment_management_tab(self) -> QWidget:
        """Create the experiment management tab."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Header
        header_label = QLabel("Experiment Management")
        header_label.setStyleSheet("font-weight: bold; font-size: 11pt; margin-bottom: 0px;")
        layout.addWidget(header_label)

        # Create horizontal splitter - this should take up all remaining space
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)  # Give splitter stretch factor of 1

        # Left side: Experiment list and operations
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(5)

        # Operations buttons
        ops_layout = QHBoxLayout()
        ops_layout.setSpacing(4)

        self.create_experiment_button = QPushButton("Create Empty")
        self.create_experiment_button.clicked.connect(self.create_experiment)
        self.create_experiment_button.setMaximumHeight(30)
        ops_layout.addWidget(self.create_experiment_button)

        self.import_experiment_button = QPushButton("Import")
        self.import_experiment_button.clicked.connect(self.import_experiment)
        self.import_experiment_button.setMaximumHeight(30)
        ops_layout.addWidget(self.import_experiment_button)

        self.rename_experiment_button = QPushButton("Rename")
        self.rename_experiment_button.clicked.connect(self.rename_experiment)
        self.rename_experiment_button.setEnabled(False)
        self.rename_experiment_button.setMaximumHeight(30)
        ops_layout.addWidget(self.rename_experiment_button)

        self.delete_experiment_button = QPushButton("Delete")
        self.delete_experiment_button.clicked.connect(self.delete_experiment)
        self.delete_experiment_button.setEnabled(False)
        self.delete_experiment_button.setMaximumHeight(30)
        ops_layout.addWidget(self.delete_experiment_button)

        left_layout.addLayout(ops_layout)

        # Experiment list - make sure it expands to fill space
        self.experiment_list = QListWidget()
        self.experiment_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.experiment_list.itemSelectionChanged.connect(self.on_experiment_selection_changed)
        self.experiment_list.setMinimumWidth(280)
        self.experiment_list.setMaximumWidth(350)
        left_layout.addWidget(self.experiment_list, 1)  # Give list stretch factor

        splitter.addWidget(left_widget)

        # Right side: Preview area
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)
        right_layout.setSpacing(5)

        preview_label = QLabel("Preview")
        preview_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        right_layout.addWidget(preview_label)

        self.experiment_preview = QLabel("Select an experiment to see details")
        self.experiment_preview.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.experiment_preview.setStyleSheet(
            """
            border: 1px solid #666;
            background-color: #2d2d2d;
            padding: 10px;
            font-size: 11px;
            color: #cccccc;
        """
        )
        self.experiment_preview.setWordWrap(True)
        self.experiment_preview.setMinimumHeight(100)
        right_layout.addWidget(self.experiment_preview, 1)  # Give preview stretch factor

        splitter.addWidget(right_widget)
        # Set proportional sizes
        splitter.setSizes([330, 670])  # More space to preview
        splitter.setStretchFactor(0, 0)  # Fixed left panel size
        splitter.setStretchFactor(1, 1)  # Allow right panel to stretch
        splitter.setCollapsible(0, False)  # Don't allow left panel to collapse
        splitter.setCollapsible(1, False)  # Don't allow right panel to collapse

        return tab_widget

    def create_dataset_management_tab(self) -> QWidget:
        """Create the dataset organization tab with drag-and-drop functionality."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)

        # Header
        header_label = QLabel("Dataset Organization")
        header_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(header_label)

        # Instructions
        instructions = QLabel(
            "Drag datasets between experiments to reorganize your data structure. Use checkboxes for batch operations."
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

        return tab_widget

    def load_data(self):
        """Load current experiment and dataset data."""
        try:
            logging.debug("Data Curation Manager: Loading data...")
            logging.debug(f"Available experiments: {list(self.gui.expts_dict_keys)}")
            logging.debug(f"Experiment paths: {dict(self.gui.expts_dict)}")

            # Store original state
            self.original_experiments = dict(self.gui.expts_dict)

            # Populate experiment list
            self.update_experiment_list()
            self.update_dataset_tree()

            # Initialize button states
            self._update_button_states()

            logging.debug("Data Curation Manager: Data loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load data in Data Curation Manager: {e}")
            QMessageBox.critical(self, "Data Loading Error", f"Failed to load experiment data:\n{str(e)}")

    def update_experiment_list(self):
        """Update the experiment list widget."""
        logging.debug("Updating experiment list...")
        self.experiment_list.clear()

        exp_count = 0
        for exp_id in self.gui.expts_dict_keys:
            item = QListWidgetItem(exp_id)
            item.setData(Qt.ItemDataRole.UserRole, exp_id)
            self.experiment_list.addItem(item)
            exp_count += 1
            logging.debug(f"Added experiment to list: {exp_id}")

        logging.debug(f"Experiment list updated with {exp_count} experiments")

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
                exp_item.setExpanded(True)  # Default to expanded for new experiments

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

    def on_experiment_selection_changed(self):
        """Handle experiment selection changes."""
        try:
            current_item = self.experiment_list.currentItem()
            has_selection = current_item is not None

            self.rename_experiment_button.setEnabled(has_selection)
            self.delete_experiment_button.setEnabled(has_selection)

            if has_selection:
                exp_id = current_item.data(Qt.ItemDataRole.UserRole)

                try:
                    # Use lightweight metadata instead of loading full experiment
                    from pathlib import Path

                    exp_path = Path(self.gui.expts_dict[exp_id])
                    exp_metadata = self._get_experiment_metadata(exp_path)

                    self.update_experiment_preview_from_metadata(exp_id, exp_metadata)

                except Exception as e:
                    logging.error(f"Failed to load experiment '{exp_id}' metadata for preview: {e}")
                    self.experiment_preview.setText(
                        f"<b>Experiment:</b> {exp_id}<br><br><i>Error loading experiment details</i>"
                    )
            else:
                self.experiment_preview.setText("Select an experiment to see details")

        except Exception as e:
            logging.error(f"Error in experiment selection changed: {e}")
            self.experiment_preview.setText("Error loading experiment details")

    def update_experiment_preview_from_metadata(self, exp_id: str, exp_metadata: dict):
        """Update the experiment preview area using lightweight metadata."""
        try:
            preview_text = f"<b>Experiment:</b> {exp_id}<br><br>"

            dataset_count = exp_metadata.get("dataset_count", 0)
            preview_text += f"<b>Datasets:</b> {dataset_count}<br><br>"

            datasets = exp_metadata.get("datasets", [])
            if datasets:
                preview_text += "<b>Dataset Details:</b><br>"
                for ds_metadata in datasets:
                    try:
                        ds_name = ds_metadata.get("formatted_name", ds_metadata.get("id", "Unknown"))
                        session_count = ds_metadata.get("session_count", 0)
                        is_completed = ds_metadata.get("is_completed", False)
                        status = "✓" if is_completed else "○"
                        preview_text += f"• {status} {ds_name} ({session_count} sessions)<br>"
                    except Exception as e:
                        logging.warning(f"Error processing dataset metadata for preview: {e}")
                        ds_name = ds_metadata.get("id", "Unknown")
                        preview_text += f"• ? {ds_name} (error loading details)<br>"
            else:
                preview_text += "<i>No datasets in this experiment</i>"

            # Add experiment path info
            exp_path = exp_metadata.get("path", "Unknown")
            preview_text += f"<br><small><b>Path:</b> {exp_path}</small>"

            self.experiment_preview.setText(preview_text)

        except Exception as e:
            logging.error(f"Error updating experiment preview from metadata: {e}")
            self.experiment_preview.setText(f"<b>Experiment:</b> {exp_id}<br><br><i>Error loading preview</i>")

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

    @auto_refresh
    def on_dataset_dragged(self, dataset_id, formatted_name, source_exp_id, target_exp_id):
        """Handle dataset drag-and-drop operations."""
        try:
            from monstim_gui.commands import MoveDatasetCommand

            # Execute the move command
            command = MoveDatasetCommand(self.gui, dataset_id, formatted_name, source_exp_id, target_exp_id)

            command.execute()
            self.session_commands.append(command)
            self._changes_made = True

            # Show success message
            QMessageBox.information(
                self,
                "Dataset Moved",
                f"Dataset '{formatted_name}' has been moved from '{source_exp_id}' to '{target_exp_id}'.",
            )

        except Exception as e:
            logging.error(f"Failed to move dataset via drag-and-drop: {e}")
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
        """Import experiment using existing functionality."""
        try:
            # Check for unsaved changes
            if self.gui.has_unsaved_changes:
                reply = QMessageBox.question(
                    self,
                    "Unsaved Changes",
                    "You have unsaved changes. Save them before importing?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                )

                if reply == QMessageBox.StandardButton.Yes:
                    self.gui.data_manager.save_experiment()
                elif reply == QMessageBox.StandardButton.Cancel:
                    return

            # Use existing import functionality
            try:
                self.gui.data_manager.import_expt_data()
                QMessageBox.information(self, "Import Complete", "Experiment imported successfully. Data has been refreshed.")
            except Exception as e:
                logging.error(f"Failed to import experiment: {e}")
                QMessageBox.critical(self, "Import Error", f"Failed to import experiment:\n{str(e)}")
                raise  # Let decorator handle the refresh

        except Exception as e:
            logging.error(f"Error in import_experiment: {e}")
            QMessageBox.critical(self, "Error", f"Unexpected error during import:\n{str(e)}")
            raise  # Let decorator handle the refresh

    @auto_refresh
    def rename_experiment(self):
        """Rename selected experiment using command pattern."""
        from PyQt6.QtWidgets import QInputDialog

        from monstim_gui.commands import RenameExperimentCommand

        current_item = self.experiment_list.currentItem()
        if not current_item:
            return

        old_name = current_item.data(Qt.ItemDataRole.UserRole)

        # Get new name from user
        new_name, ok = QInputDialog.getText(
            self, "Rename Experiment", f"Enter new name for experiment '{old_name}':", text=old_name
        )

        if ok and new_name.strip() and new_name.strip() != old_name:
            new_name = new_name.strip()

            # Check for naming conflicts
            if new_name in self.gui.expts_dict_keys:
                QMessageBox.warning(self, "Name Conflict", f"An experiment named '{new_name}' already exists.")
                return

            # Execute command immediately
            command = RenameExperimentCommand(self.gui, old_name, new_name)
            try:
                command.execute()
                self.session_commands.append(command)
                self._changes_made = True  # Mark that changes were made
                QMessageBox.information(
                    self, "Experiment Renamed", f"Experiment '{old_name}' has been renamed to '{new_name}'."
                )
            except Exception as e:
                QMessageBox.critical(self, "Rename Error", f"Failed to rename experiment:\n{str(e)}")
                raise  # Let decorator handle the refresh

    @auto_refresh
    def delete_experiment(self):
        """Delete selected experiment immediately."""
        from monstim_gui.commands import DeleteExperimentCommand

        current_item = self.experiment_list.currentItem()
        if not current_item:
            return

        exp_id = current_item.data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Confirm Experiment Deletion",
            f"Are you sure you want to delete experiment '{exp_id}'?\n\n"
            "This will permanently remove all datasets and files in this experiment.\n\n"
            "WARNING: This action cannot be undone!",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            command = DeleteExperimentCommand(self.gui, exp_id)
            try:
                command.execute()
                self.session_commands.append(command)
                QMessageBox.information(self, "Experiment Deleted", f"Experiment '{exp_id}' has been permanently deleted.")
            except Exception as e:
                QMessageBox.critical(self, "Deletion Failed", f"Failed to delete experiment:\n{str(e)}")
                raise  # Let decorator handle the refresh

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

    def reset_all_changes(self):
        """Reset/clear all session commands - used by reset button."""
        if self.session_commands:
            reply = QMessageBox.question(
                self,
                "Reset Changes",
                f"This will clear the record of {len(self.session_commands)} operation(s) from this session.\n\n"
                "The changes will remain applied, but you won't be able to undo them using 'Cancel All Changes'.\n\n"
                "Are you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.session_commands.clear()
                QMessageBox.information(self, "Reset Complete", "Session command history has been cleared.")

        QMessageBox.information(self, "Changes Reset", "All pending changes have been cleared.")

    def cancel_all_changes(self):
        """Undo all changes made during this session."""
        if not self.session_commands:
            self.reject()
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
                    command.undo()

                QMessageBox.information(
                    self, "Changes Undone", f"All {len(self.session_commands)} operation(s) have been successfully undone."
                )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Undoing Changes",
                    f"An error occurred while undoing changes:\n{str(e)}\n\n" "Some changes may not have been fully reversed.",
                )
                raise  # Let decorator handle the refresh
            finally:
                self.reject()

    def accept(self):
        """Override accept to emit signal if changes were made."""
        # Track if any changes were made during this session
        changes_made = hasattr(self, "_changes_made") and self._changes_made

        if changes_made:
            self.data_structure_changed.emit()

        super().accept()

    def reject(self):
        """Override reject to emit signal if changes were made."""
        # Track if any changes were made during this session
        changes_made = hasattr(self, "_changes_made") and self._changes_made

        if changes_made:
            self.data_structure_changed.emit()

        super().reject()
