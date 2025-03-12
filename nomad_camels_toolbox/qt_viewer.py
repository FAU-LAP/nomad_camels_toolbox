import sys
from importlib import resources

import PySide6
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt

import pyqtgraph as pg

from utils.exception_hook import exception_hook
import graphics

from data_reader import read_camels_file

# these are the colors used by matplotlib, they are used as default colors in light mode
matplotlib_default_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# these are the symbols recognized by pyqtgraph
symbols = {
    "circle": "o",
    "square": "s",
    "triangle": "t",
    "diamond": "d",
    "plus": "+",
    "upwards triangle": "t1",
    "right triangle": "t2",
    "left triangle": "t3",
    "pentagon": "p",
    "hexagon": "h",
    "star": "star",
    "cross": "x",
    "arrow_up": "arrow_up",
    "arrow_right": "arrow_right",
    "arrow_down": "arrow_down",
    "arrow_left": "arrow_left",
    "crosshair": "crosshair",
    "none": None,
}

# these are the linestyles recognized by pyqtgraph
linestyles = {
    "solid": Qt.PenStyle.SolidLine,
    "dashed": Qt.PenStyle.DashLine,
    "dash-dot": Qt.PenStyle.DashDotLine,
    "dash-dot-dot": Qt.PenStyle.DashDotDotLine,
    "dotted": Qt.PenStyle.DotLine,
    "none": Qt.PenStyle.NoPen,
}


dark_palette = QtGui.QPalette()
dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
dark_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColorConstants.White)
dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColorConstants.White)
dark_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColorConstants.White)
dark_palette.setColor(QtGui.QPalette.Text, QtGui.QColorConstants.White)
dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
dark_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColorConstants.White)
dark_palette.setColor(QtGui.QPalette.BrightText, QtGui.QColorConstants.Red)
dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
dark_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColorConstants.Black)

light_palette = QtGui.QPalette(QtGui.QColor(225, 225, 225), QtGui.QColor(238, 238, 238))
light_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))


def set_theme(dark_mode=False):
    main_app = QtWidgets.QApplication.instance()
    if dark_mode:
        pg.setConfigOptions(background="k", foreground="w")
        palette = dark_palette
    else:
        pg.setConfigOptions(background="w", foreground="k")
        palette = light_palette
    main_app.setPalette(palette)
    main_app.setStyle("Fusion")


class DragDropGraphicLayoutWidget(pg.GraphicsLayoutWidget):
    dropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.dropped.emit(links)
        else:
            event.ignore()


class CAMELS_Viewer(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NOMAD CAMELS Data Viewer")
        self.setWindowIcon(
            QtGui.QIcon(str(resources.files(graphics) / "CAMELS_icon.png"))
        )
        self.graphics_view = DragDropGraphicLayoutWidget()
        self.graphics_view.dropped.connect(self.load_data)
        self.left_widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        self.left_widget.setLayout(layout)
        # main_layout = QtWidgets.QHBoxLayout()
        # self.setLayout(main_layout)
        self.setCentralWidget(QtWidgets.QSplitter())
        self.centralWidget().addWidget(self.left_widget)
        self.centralWidget().addWidget(self.graphics_view)
        # self.centralWidget().setLayout(main_layout)

        self.histogram = pg.HistogramLUTItem()
        self.image_plot = self.graphics_view.addPlot()
        self.graphics_view.addItem(self.histogram)
        self.histogram.autoHistogramRange()
        self.graphics_view.nextRow()
        self.roi_plot = self.graphics_view.addPlot()
        self.roi_plot.hide()
        self.graphics_view.nextRow()
        self.roi_intensity_plot = self.graphics_view.addPlot()
        self.roi_intensity_plot.hide()

        self.load_measurement_button = QtWidgets.QPushButton("Load Measurement")
        self.load_measurement_button.clicked.connect(self.load_measurement)
        layout.addWidget(self.load_measurement_button, 0, 0)

        self.plot_table = QtWidgets.QTableWidget()
        labels = [
            "plot?",
            "X",
            "Y",
            "color",
            "symbol",
            "linestyle",
            "file",
            "file-entry",
        ]
        self.plot_table.setColumnCount(len(labels))
        self.plot_table.setHorizontalHeaderLabels(labels)
        self.plot_table.verticalHeader().hide()
        self.plot_table.resizeColumnsToContents()

        layout.addWidget(self.plot_table, 1, 0)

        self.data = {}

        self.adjustSize()

    def add_table_row(self, data, fname="", entry_name=""):
        row = self.plot_table.rowCount()
        self.plot_table.setRowCount(row + 1)
        self.plot_table.setItem(row, 0, QtWidgets.QTableWidgetItem())
        self.plot_table.item(row, 0).setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        self.plot_table.item(row, 0).setCheckState(Qt.Checked)
        data_keys = list(data.keys())
        box = QtWidgets.QComboBox()
        box.addItems(data_keys)
        self.plot_table.setCellWidget(row, 1, box)
        box = QtWidgets.QComboBox()
        box.addItems(data_keys)
        self.plot_table.setCellWidget(row, 2, box)
        box = QtWidgets.QComboBox()
        box.addItems(matplotlib_default_colors)
        self.plot_table.setCellWidget(row, 3, box)
        box = QtWidgets.QComboBox()
        box.addItems(list(symbols.keys()))
        self.plot_table.setCellWidget(row, 4, box)
        box = QtWidgets.QComboBox()
        box.addItems(list(linestyles.keys()))
        self.plot_table.setCellWidget(row, 5, box)
        self.plot_table.setItem(row, 6, QtWidgets.QTableWidgetItem(fname))
        self.plot_table.setItem(row, 7, QtWidgets.QTableWidgetItem(entry_name))
        self.plot_table.resizeColumnsToContents()

    def load_measurement(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            self.load_data(file_paths)

    def load_data(self, file_paths):
        for file_path in file_paths:
            data = read_camels_file(file_path)
            self.data[file_path] = data
            self.add_table_row(data=data, fname=file_path)


def run_viewer():
    app = QtWidgets.QApplication([])
    set_theme()
    sys.excepthook = exception_hook
    window = CAMELS_Viewer()
    window.show()
    app.exec_()


if __name__ == "__main__":
    run_viewer()
