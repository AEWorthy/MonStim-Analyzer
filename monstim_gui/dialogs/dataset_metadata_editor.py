"""
Dataset metadata editor dialog for MonStim Analyzer.
Allows users to edit dataset date, animal ID, and experimental condition.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

if TYPE_CHECKING:
    from monstim_signals import Dataset


class DatasetMetadataEditor(QDialog):
    """Dialog for editing dataset metadata (date, animal ID, condition)."""

    def __init__(self, dataset: "Dataset", parent=None):
        super().__init__(parent)
        self.dataset: "Dataset" = dataset
        self.setWindowTitle("Edit Dataset Metadata")
        self.setModal(True)
        self.resize(500, 300)

        # Store metadata for command creation
        self.old_date = self.dataset.annot.date
        self.old_animal_id = self.dataset.annot.animal_id
        self.old_condition = self.dataset.annot.condition
        self.old_folder_name = self.dataset.id if self.dataset.repo else None

        # These will be set when user clicks Save
        self.new_date = None
        self.new_animal_id = None
        self.new_condition = None
        self.new_folder_name = None

        # Initialize UI
        self._setup_ui()
        self._populate_fields()

    def _setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout()

        # Title and current name display
        title_label = QLabel("Edit Dataset Metadata")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Current dataset info group
        info_group = QGroupBox("Current Dataset")
        info_layout = QVBoxLayout()

        self.folder_name_label = QLabel(f"Folder Name: {self.dataset.id}")
        self.current_display_label = QLabel(f"Current Display: {self.dataset.formatted_name}")

        info_layout.addWidget(self.folder_name_label)
        info_layout.addWidget(self.current_display_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Metadata editing group
        edit_group = QGroupBox("Edit Metadata")
        form_layout = QFormLayout()

        # Date field
        self.date_edit = QLineEdit()
        self.date_edit.setPlaceholderText("YYYY-MM-DD or YYMMDD or YYYYMMDD")
        self.date_edit.setToolTip(
            "Enter date in one of these formats:\n"
            "• YYYY-MM-DD (e.g., 2024-08-15)\n"
            "• YYMMDD (e.g., 240815)\n"
            "• YYYYMMDD (e.g., 20240815)"
        )
        form_layout.addRow("Date:", self.date_edit)

        # Animal ID field
        self.animal_id_edit = QLineEdit()
        self.animal_id_edit.setPlaceholderText("e.g., BEM3, XX000.0, C537.6")
        self.animal_id_edit.setToolTip("Animal identifier (letters, numbers, and dots allowed)")
        form_layout.addRow("Animal ID:", self.animal_id_edit)

        # Condition field
        self.condition_edit = QLineEdit()
        self.condition_edit.setPlaceholderText("e.g., test condition, pre-dec mcurves_long-")
        self.condition_edit.setToolTip("Experimental condition or description")
        form_layout.addRow("Condition:", self.condition_edit)

        edit_group.setLayout(form_layout)
        layout.addWidget(edit_group)

        # Preview group
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout()

        self.folder_preview_label = QLabel("New Folder Name: (enter values above)")
        self.folder_preview_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        preview_layout.addWidget(self.folder_preview_label)

        self.preview_label = QLabel("New Display Name: (enter values above)")
        self.preview_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        preview_layout.addWidget(self.preview_label)

        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Connect text changes to preview update
        self.date_edit.textChanged.connect(self._update_preview)
        self.animal_id_edit.textChanged.connect(self._update_preview)
        self.condition_edit.textChanged.connect(self._update_preview)

        # Buttons
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save Changes")
        self.save_button.clicked.connect(self._save_changes)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _populate_fields(self):
        """Populate the form fields with current dataset metadata."""
        # Get current values, handling both None and "UNDEFINED"
        current_date = self.dataset.annot.date
        current_animal_id = self.dataset.annot.animal_id
        current_condition = self.dataset.annot.condition

        # Populate fields if values exist and are not "UNDEFINED"
        if current_date and current_date != "UNDEFINED":
            self.date_edit.setText(current_date)

        if current_animal_id and current_animal_id != "UNDEFINED":
            self.animal_id_edit.setText(current_animal_id)

        if current_condition and current_condition != "UNDEFINED":
            self.condition_edit.setText(current_condition)

        # Update preview
        self._update_preview()

    def _update_preview(self):
        """Update the preview of what the new display name will be."""
        date = self._parse_date_input(self.date_edit.text().strip())
        animal_id = self.animal_id_edit.text().strip()
        condition = self.condition_edit.text().strip()

        if date and animal_id and condition:
            # Format folder name (replace spaces and special characters for filesystem compatibility)
            folder_name = self._format_folder_name(date, animal_id, condition)
            self.folder_preview_label.setText(f"New Folder Name: {folder_name}")
            self.folder_preview_label.setStyleSheet("font-weight: bold; color: #00aa00;")

            preview_name = f"{date} {animal_id} {condition}"
            self.preview_label.setText(f"New Display Name: {preview_name}")
            self.preview_label.setStyleSheet("font-weight: bold; color: #00aa00;")
        else:
            self.folder_preview_label.setText(f"New Folder Name: {self.dataset.id} (no change)")
            self.folder_preview_label.setStyleSheet("font-weight: bold; color: #888888;")

            self.preview_label.setText(f"New Display Name: {self.dataset.id} (using folder name)")
            self.preview_label.setStyleSheet("font-weight: bold; color: #888888;")

    def _parse_date_input(self, date_input):
        """Parse various date input formats and return standardized format."""
        if not date_input:
            return None

        # Remove any non-alphanumeric characters for parsing
        clean_input = re.sub(r"[^0-9]", "", date_input)

        # Check for YYMMDD format (6 digits)
        if len(clean_input) == 6 and clean_input.isdigit():
            yy = int(clean_input[:2])
            # Pivot year logic: years 00-69 -> 2000-2069, 70-99 -> 1970-1999
            if yy <= 69:
                year = 2000 + yy
            else:
                year = 1900 + yy
            month = int(clean_input[2:4])
            day = int(clean_input[4:6])
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"

        # Check for YYYYMMDD format (8 digits)
        elif len(clean_input) == 8 and clean_input.isdigit():
            year = int(clean_input[:4])
            month = int(clean_input[4:6])
            day = int(clean_input[6:8])
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"

        # Check if it's already in YYYY-MM-DD format
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", date_input):
            parts = date_input.split("-")
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                return date_input

        return None

    def _format_folder_name(self, date, animal_id, condition):
        """Format the folder name for filesystem compatibility."""
        # For folder names, we typically use the short date format (YYMMDD)
        # Convert YYYY-MM-DD back to YYMMDD for folder names
        if len(date) == 10 and date.count("-") == 2:  # YYYY-MM-DD format
            year, month, day = date.split("-")
            short_date = year[2:] + month + day  # Convert to YYMMDD
        else:
            short_date = date

        # Create folder name with single spaces (standard format)
        folder_name = f"{short_date} {animal_id} {condition}"

        # Replace any characters that might be problematic for filenames
        # but keep spaces as they are expected in the standard format
        problematic_chars = '<>:"|?*'
        for char in problematic_chars:
            folder_name = folder_name.replace(char, "_")

        return folder_name

    def _validate_inputs(self):
        """Validate all input fields."""
        errors = []

        date_input = self.date_edit.text().strip()
        animal_id_input = self.animal_id_edit.text().strip()

        # Validate date if provided
        if date_input and not self._parse_date_input(date_input):
            errors.append("Invalid date format. Use YYYY-MM-DD, YYMMDD, or YYYYMMDD.")

        # Validate animal ID format if provided
        if animal_id_input and not re.match(r"^[a-zA-Z0-9.]+$", animal_id_input):
            errors.append("Animal ID can only contain letters, numbers, and dots.")

        # No specific validation for condition - any text is allowed

        return errors

    def _save_changes(self):
        """Validate inputs and prepare metadata for command execution."""
        # Validate inputs
        errors = self._validate_inputs()
        if errors:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please correct the following errors:\n\n" + "\n".join(f"• {error}" for error in errors),
            )
            return

        # Get the input values
        date_input = self.date_edit.text().strip()
        animal_id_input = self.animal_id_edit.text().strip()
        condition_input = self.condition_edit.text().strip()

        # Parse and store the values
        self.new_date = self._parse_date_input(date_input) if date_input else None
        self.new_animal_id = animal_id_input if animal_id_input else None
        self.new_condition = condition_input if condition_input else None

        # Check if we need to rename the folder
        should_rename_folder = self.new_date and self.new_animal_id and self.new_condition and self.dataset.repo is not None

        if should_rename_folder:
            # Calculate new folder name
            new_folder_name = self._format_folder_name(self.new_date, self.new_animal_id, self.new_condition)
            old_folder_path = Path(self.dataset.repo.folder)

            # Check if rename is needed (folder name changed)
            if old_folder_path.name != new_folder_name:
                new_folder_path = old_folder_path.parent / new_folder_name

                # Check if target folder already exists
                if new_folder_path.exists():
                    QMessageBox.warning(
                        self,
                        "Folder Exists",
                        f"A folder with the name '{new_folder_name}' already exists.\n"
                        f"Please choose different metadata values or rename the existing folder first.",
                    )
                    return

                self.new_folder_name = new_folder_name
            else:
                # Folder name won't change
                self.new_folder_name = None
        else:
            self.new_folder_name = None

        # Accept the dialog - the command will execute the changes
        self.accept()
