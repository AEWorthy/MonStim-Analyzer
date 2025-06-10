from .base import *


class SelectChannelsDialog(QDialog):
    def __init__(self, data: Experiment | Dataset | Session, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Select Channels")

        self.data = data
        self.channel_names = data.channel_names

        self.selected_channels = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Add checkbox header
        header_layout = QVBoxLayout()
        header_layout.addWidget(QLabel(f"Select channels for\n'{self.data.formatted_name}'"))
        header_layout.addWidget(QLabel("\nSelect channels:"))
        layout.addLayout(header_layout)

        # Add checkboxes for each channel in the dataset
        self.checkboxes = []
        for name in self.channel_names:
            checkbox = QCheckBox(name, self)

            # Check the checkbox if the channel is already selected
            excluded = getattr(self.data, 'excluded_channels', [])
            if self.data.channel_names.index(name) not in excluded:
                checkbox.setChecked(True)

            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)

        # Add button box (OK and Cancel buttons)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        layout.addWidget(button_box)

        # Connect signals
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Final layout setup
        self.setLayout(layout)

    def get_selected_channel_indexes(self):
        # Return the indexes of the channels where checkboxes are checked
        return [i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]
