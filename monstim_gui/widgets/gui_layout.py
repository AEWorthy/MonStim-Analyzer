from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QSizePolicy, QSplitter, QStatusBar, QVBoxLayout, QWidget

from ..core.ui_scaling import get_responsive_margins, get_responsive_spacing, ui_scaling
from ..plotting import PlotPane, PlotWidget
from .data_selection_widget import DataSelectionWidget
from .menu_bar import MenuBar
from .reports_widget import ReportsWidget

if TYPE_CHECKING:
    from gui_main import MonstimGUI


def setup_main_layout(parent: MonstimGUI) -> dict:
    """Create and apply the main window layout.

    Parameters
    ----------
    parent : QMainWindow
        The window that will hold all widgets.

    Returns
    -------
    dict
        dictionary containing all created widgets.
    """
    # Central widget and main layout
    central_widget = QWidget()
    parent.setCentralWidget(central_widget)
    main_layout = QHBoxLayout(central_widget)

    # Apply responsive spacing and margins
    spacing = get_responsive_spacing(8)
    margins = get_responsive_margins(10)
    main_layout.setSpacing(spacing)
    main_layout.setContentsMargins(*margins)

    # Widgets
    menu_bar = MenuBar(parent)
    data_selection_widget = DataSelectionWidget(parent)
    reports_widget = ReportsWidget(parent)
    plot_pane = PlotPane(parent)
    plot_widget = PlotWidget(parent)

    # Left panel holding controls.  Its baseline width is the widest section's
    # expanded minimum, so collapsing a section never makes the sidebar jump.
    left_panel = QWidget()
    left_panel.setObjectName("mainSidebar")

    left_layout = QVBoxLayout(left_panel)
    left_spacing = get_responsive_spacing(10)
    left_layout.setSpacing(left_spacing)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    left_layout.addWidget(data_selection_widget)
    left_layout.addWidget(reports_widget)
    left_layout.addWidget(plot_widget)
    left_panel.setLayout(left_layout)
    intended_width = max(
        ui_scaling.get_optimal_panel_width(300, 600),
        *(section.expanded_minimum_width() for section in (data_selection_widget, reports_widget, plot_widget)),
    )
    # Start at the widest section's intended width, while still letting a
    # user make the sidebar narrower through the splitter if desired.
    left_panel.setMinimumWidth(ui_scaling.get_optimal_panel_width(220, 600))
    left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    plot_pane.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    splitter = QSplitter(Qt.Orientation.Horizontal)
    splitter.setObjectName("mainContentSplitter")
    splitter.setChildrenCollapsible(False)
    splitter.setHandleWidth(get_responsive_spacing(6))
    splitter.addWidget(left_panel)
    splitter.addWidget(plot_pane)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([intended_width, intended_width * 2])
    main_layout.addWidget(splitter)
    central_widget.setMinimumWidth(intended_width + plot_pane.minimumSizeHint().width() + splitter.handleWidth() + margins[0] + margins[2])

    parent.setMenuBar(menu_bar)

    status_bar = QStatusBar()
    parent.setStatusBar(status_bar)

    return {
        "menu_bar": menu_bar,
        "data_selection_widget": data_selection_widget,
        "reports_widget": reports_widget,
        "plot_pane": plot_pane,
        "plot_widget": plot_widget,
        "main_splitter": splitter,
        "status_bar": status_bar,
    }
