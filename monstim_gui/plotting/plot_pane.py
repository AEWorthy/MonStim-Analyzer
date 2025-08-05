import logging
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PyQt6.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI


class PlotPane(QGroupBox):
    def __init__(self, parent: "MonstimGUI"):
        super().__init__("Plot Pane", parent)
        self.parent = parent
        self.layout = QVBoxLayout()

        # Set the default background color for pyqtgraph to white and text to black
        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')

        # Create the main graphics layout widget
        self.graphics_layout: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self.graphics_layout.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.graphics_layout.setMinimumSize(800, 400)
        self.graphics_layout.setAntialiasing(True)
        self.layout.addWidget(self.graphics_layout)

        # Store references to current plots
        self.current_plots = []
        self.current_plot_items = []

        self.setLayout(self.layout)
        logging.debug("PyQtGraph canvas created and added to layout.")

    def clear_plots(self):
        """Clear all current plots"""
        self.graphics_layout.clear()
        self.current_plots = []
        self.current_plot_items = []
