from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from monstim_gui.widgets.collapsible_group_box import CollapsibleGroupBox


def test_collapsible_group_box_hides_and_restores_direct_content(qapplication):
    section = CollapsibleGroupBox("Example")
    layout = QVBoxLayout(section)
    content = QLabel("Section content")
    layout.addWidget(content)
    section.show()

    assert section.is_expanded()
    assert content.isVisible()

    expanded_width = section.expanded_minimum_width()

    section.toggle_button.click()

    assert not section.is_expanded()
    assert not content.isVisible()
    assert section.expanded_minimum_width() == expanded_width

    section.toggle_button.click()

    assert section.is_expanded()
    assert content.isVisible()


def test_sidebar_sections_default_to_requested_state(qapplication):

    from monstim_gui.plotting.plotting_widget import PlotWidget
    from monstim_gui.widgets.data_selection_widget import DataSelectionWidget
    from monstim_gui.widgets.reports_widget import ReportsWidget

    class _Parent(QWidget):
        def __init__(self):
            super().__init__()
            self.report_manager = type(
                "ReportManager",
                (),
                {
                    "show_session_report": lambda: None,
                    "show_dataset_report": lambda: None,
                    "show_experiment_report": lambda: None,
                    "show_mmax_report": lambda: None,
                },
            )()
            self.plot_controller = type(
                "PlotController",
                (),
                {
                    "plot_data": lambda: None,
                    "get_raw_data": lambda: None,
                },
            )()

    gui = _Parent()
    data_selection_widget = DataSelectionWidget(gui)
    assert data_selection_widget.is_expanded()
    assert not ReportsWidget(gui).is_expanded()
    assert PlotWidget(gui).is_expanded()
    assert data_selection_widget.experiment_combo.sizeAdjustPolicy().name == "AdjustToMinimumContentsLengthWithIcon"
