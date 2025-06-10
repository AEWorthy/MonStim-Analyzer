from .base import *


class ChangeChannelNamesDialog(QDialog):
    def __init__(self, channel_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Channel Names")
        self.setModal(True)
        layout = QGridLayout(self)

        self.channel_inputs = {}
        for i, channel_name in enumerate(channel_names):
            layout.addWidget(QLabel(f"Channel {i+1}:"), i, 0)
            self.channel_inputs[channel_name] = QLineEdit(channel_name)
            layout.addWidget(self.channel_inputs[channel_name], i, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, len(channel_names), 0, 1, 2)

    def get_new_names(self):
        return {old: input.text() for old, input in self.channel_inputs.items()}


class InvertChannelPolarityDialog(QDialog):
    def __init__(self, data: Experiment | Dataset | Session, parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Invert Channel Polarity")

        self.data = data
        self.channel_names = data.channel_names

        self.selected_channels = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Add checkbox header
        header_layout = QVBoxLayout()
        header_layout.addWidget(QLabel(f"Invert selected channel polarities for\n'{self.data.formatted_name}'"))
        header_layout.addWidget(QLabel("\nSelect channels to invert:"))
        layout.addLayout(header_layout)

        # Add checkboxes for each channel in the dataset
        self.checkboxes = []
        for name in self.channel_names:
            checkbox = QCheckBox(name, self)
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
