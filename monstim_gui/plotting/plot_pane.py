import logging
from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtWidgets import QGroupBox, QSizePolicy, QVBoxLayout

from monstim_gui.core.application_state import app_state

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI


class PlotPane(QGroupBox):
    def __init__(self, parent: "MonstimGUI", use_opengl: bool = None):
        super().__init__("Plot Pane", parent)
        self.parent = parent
        self.layout = QVBoxLayout()

        # Use application state setting if use_opengl not explicitly provided
        if use_opengl is None:
            use_opengl = app_state.should_use_opengl_acceleration()

        # Create the main graphics layout widget with optional OpenGL acceleration
        if use_opengl:
            try:
                # Try to enable OpenGL acceleration using PyQtGraph's global configuration
                pg.setConfigOption("useOpenGL", True)
                pg.setConfigOption("antialias", True)
                pg.setConfigOption("enableExperimental", False)
                self.graphics_layout: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
                logging.info("OpenGL acceleration enabled for plot rendering")
            except Exception as e:
                logging.warning(f"Failed to enable OpenGL acceleration: {e}. Falling back to software rendering.")
                pg.setConfigOption("useOpenGL", False)
                pg.setConfigOption("antialias", True)
                pg.setConfigOption("enableExperimental", False)
                self.graphics_layout: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        else:
            pg.setConfigOption("useOpenGL", False)
            pg.setConfigOption("antialias", True)
            pg.setConfigOption("enableExperimental", False)
            self.graphics_layout: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
            logging.info("OpenGL acceleration disabled (software rendering)")

        self.graphics_layout.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.graphics_layout.setMinimumSize(800, 400)
        self.graphics_layout.setAntialiasing(True)
        self.graphics_layout.setUpdatesEnabled(True)

        self.layout.addWidget(self.graphics_layout)

        # Store references to current plots
        self.current_plots = []
        self.current_plot_items = []

        self.setLayout(self.layout)
        logging.debug("PyQtGraph canvas created and added to layout.")

    def clear_plots(self):
        """Clear all current plots with comprehensive cleanup."""
        try:
            logging.debug(f"Clearing plots. Current plot count: {len(self.current_plots)}")

            # Clear references first to help garbage collector
            self.current_plots = []
            self.current_plot_items = []

            # Clear the graphics layout
            if self.graphics_layout:
                try:
                    logging.debug("Calling graphics_layout.clear()...")
                    self.graphics_layout.clear()
                    logging.debug("Graphics layout cleared successfully.")
                except RuntimeError as e:
                    # Widget may have been destroyed
                    logging.debug(f"Graphics layout already destroyed or invalid: {e}")
                except AttributeError as e:
                    logging.debug(f"Graphics layout attribute error during clear: {e}")
                except Exception as e:
                    logging.warning(f"Unexpected error clearing graphics layout: {e}", exc_info=True)
            else:
                logging.warning("Graphics layout is None, cannot clear")

        except Exception as e:
            logging.error(f"CRITICAL: Error in clear_plots: {e}", exc_info=True)
