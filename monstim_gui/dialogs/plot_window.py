from .base import *


class PlotWindowDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Window")
        # self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'plot.png')))
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.canvas = None
        self.toolbar = None

    def create_canvas(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)  # Type: FigureCanvas
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(400, 200)
        self.layout.addWidget(self.canvas)

    def set_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

    def closeEvent(self, event):
        if self.toolbar:
            self.toolbar.deleteLater()
        if self.canvas:
            self.canvas.deleteLater()
        event.accept()
