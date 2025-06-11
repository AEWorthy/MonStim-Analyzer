from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QStatusBar

from .menu_bar import MenuBar
from .data_selection_widget import DataSelectionWidget
from .reports_widget import ReportsWidget
from .plotting.plotting_widget import PlotWidget, PlotPane


def setup_main_layout(parent):
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
    left_layout = QVBoxLayout(left_panel)
    left_layout.setSpacing(10)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.addWidget(data_selection_widget)
    left_layout.addWidget(reports_widget)
    left_layout.addWidget(plot_widget)
    left_layout.addStretch(1)

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
