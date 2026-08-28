from types import SimpleNamespace

from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from monstim_gui.plotting.plot_options import ChannelSelectorWidget


def test_channel_selector_uses_natural_horizontal_width(qapplication):
    gui_main = SimpleNamespace(
        plot_widget=SimpleNamespace(view="session"),
        current_session=SimpleNamespace(num_channels=6),
    )
    parent = QWidget()
    layout = QVBoxLayout(parent)
    selector = ChannelSelectorWidget(gui_main, parent)
    layout.addWidget(selector)

    parent.resize(600, selector.sizeHint().height())
    parent.show()
    qapplication.processEvents()

    assert selector.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Preferred
    assert selector.width() < parent.width()
    assert all(cb.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed for cb in selector.checkboxes)
