from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QStatusBar, QSizePolicy

from .menu_bar import MenuBar
from .data_selection_widget import DataSelectionWidget
from .reports_widget import ReportsWidget
from ..plotting import PlotWidget, PlotPane

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui_main import MonstimGUI


def setup_main_layout(parent: 'MonstimGUI') -> dict:
    """Create and apply the main window layout.

    Parameters
    ----------
    parent : QMainWindow
        The window that will hold all widgets.

    Returns
    -------
    dict
        Dictionary containing all created widgets.
    """
    # Central widget and main layout
    central_widget = QWidget()
    parent.setCentralWidget(central_widget)
    main_layout = QHBoxLayout(central_widget)
    main_layout.setSpacing(10)
    main_layout.setContentsMargins(20, 20, 20, 20)

    # Widgets
    menu_bar = MenuBar(parent)
    data_selection_widget = DataSelectionWidget(parent)
    reports_widget = ReportsWidget(parent)
    plot_pane = PlotPane(parent)
    plot_widget = PlotWidget(parent)

    # Left panel holding controls
    left_panel = QWidget()
    left_panel.setFixedWidth(370)
    
    left_layout = QVBoxLayout(left_panel)
    left_layout.setSpacing(10)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.addWidget(data_selection_widget)
    left_layout.addWidget(reports_widget)
    left_layout.addWidget(plot_widget)
    left_panel.setLayout(left_layout)
    left_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    plot_pane.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    main_layout.addWidget(left_panel)
    main_layout.addWidget(plot_pane)

    parent.setMenuBar(menu_bar)

    status_bar = QStatusBar()
    parent.setStatusBar(status_bar)

    return {
        "menu_bar": menu_bar,
        "data_selection_widget": data_selection_widget,
        "reports_widget": reports_widget,
        "plot_pane": plot_pane,
        "plot_widget": plot_widget,
        "status_bar": status_bar,
    }
