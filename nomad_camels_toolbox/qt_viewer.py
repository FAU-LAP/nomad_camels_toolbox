import sys
from importlib import resources

import PySide6
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
import h5py

import pyqtgraph as pg

from utils.exception_hook import exception_hook
import graphics

from data_reader import read_camels_file, PANDAS_INSTALLED

# these are the colors used by matplotlib, they are used as default colors in light mode
matplotlib_default_colors = {
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "gray": "#7f7f7f",
    "ocher": "#bcbd22",
    "turquoise": "#17becf",
}

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
        self.setWindowTitle("Data Viewer - NOMAD CAMELS Toolbox")
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

        self.image_plot = self.graphics_view.addPlot()

        self.histogram = pg.HistogramLUTItem()
        self.graphics_view.addItem(self.histogram)
        self.histogram.autoHistogramRange()
        self.histogram.hide()

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
            "data-set",
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
        self.plot_items = []

        self.adjustSize()

    def add_table_row(self, data, fname="", entry_name=""):
        row = self.plot_table.rowCount()
        self.plot_table.setRowCount(row + 1)
        self.plot_table.setItem(row, 0, QtWidgets.QTableWidgetItem())
        self.plot_table.item(row, 0).setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        self.plot_table.item(row, 0).setCheckState(Qt.Checked)

        data_sets = list(data.keys())
        box = QtWidgets.QComboBox()
        box.addItems(data_sets)
        box.currentTextChanged.connect(
            lambda text=None, x=row: self._update_x_y_comboboxes(x)
        )
        self.plot_table.setCellWidget(row, 3, box)

        box = QtWidgets.QComboBox()
        box.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        self.plot_table.setCellWidget(row, 1, box)
        box = QtWidgets.QComboBox()
        box.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )

        self.plot_table.setCellWidget(row, 2, box)
        box = QtWidgets.QComboBox()
        box.addItems(list(matplotlib_default_colors.keys()))
        box.setCurrentIndex(row % len(matplotlib_default_colors))
        box.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        self.plot_table.setCellWidget(row, 4, box)
        box = QtWidgets.QComboBox()
        box.addItems(list(symbols.keys()))
        box.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        self.plot_table.setCellWidget(row, 5, box)
        box = QtWidgets.QComboBox()
        box.addItems(list(linestyles.keys()))
        box.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )

        self.plot_table.setCellWidget(row, 6, box)
        self.plot_table.setItem(row, 7, QtWidgets.QTableWidgetItem(fname))
        self.plot_table.setItem(row, 8, QtWidgets.QTableWidgetItem(entry_name))

        self.plot_table.resizeColumnsToContents()

        self._update_x_y_comboboxes(row)

    def _update_x_y_comboboxes(self, row):
        fname = self.plot_table.item(row, 7).text()
        entry_name = self.plot_table.item(row, 8).text()
        data = self.data[f"{fname}_{entry_name}"]
        data_set_name = self.plot_table.cellWidget(row, 3).currentText()
        data_set = data[data_set_name]
        data_keys = list(data_set.keys())
        box = self.plot_table.cellWidget(row, 1)
        box.clear()
        box.addItems(data_keys)
        box = self.plot_table.cellWidget(row, 2)
        box.clear()
        box.addItems(data_keys)
        self._add_or_change_plot_data(row)

    def load_measurement(self):
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            self.load_data(file_paths)

    def load_data(self, file_paths):
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                keys = list(f.keys())
                if len(keys) > 1:
                    remaining_keys = []
                    for key in keys:
                        if not key.startswith("NeXus_"):
                            remaining_keys.append(key)
                    if len(remaining_keys) > 1:
                        key = ask_for_input_box(remaining_keys)
                    else:
                        key = remaining_keys[0]
                else:
                    key = keys[0]
            data = read_camels_file(file_path, entry_key=key, read_all_datasets=True)
            self.data[f"{file_path}_{key}"] = data
            self.add_table_row(data=data, fname=file_path, entry_name=key)
        self.update_plot()

    def update_plot(self):
        self.image_plot.clear()
        self.roi_plot.clear()
        self.roi_intensity_plot.clear()
        self.plot_items.clear()
        for row in range(self.plot_table.rowCount()):
            self._add_or_change_plot_data(row)

    def _add_or_change_plot_data(self, number):
        x_data = self.plot_table.cellWidget(number, 1).currentText()
        y_data = self.plot_table.cellWidget(number, 2).currentText()
        if not x_data or not y_data:
            return
        color = matplotlib_default_colors[
            self.plot_table.cellWidget(number, 4).currentText()
        ]
        symbol = self.plot_table.cellWidget(number, 5).currentText()
        linestyle = self.plot_table.cellWidget(number, 6).currentText()
        file_name = self.plot_table.item(number, 7).text()
        entry_name = self.plot_table.item(number, 8).text()
        data_set = self.plot_table.cellWidget(number, 3).currentText()
        data = self.data[f"{file_name}_{entry_name}"][data_set]
        try:
            x = data[x_data]
            y = data[y_data]
        except KeyError:
            return
        if PANDAS_INSTALLED:
            x = x.to_numpy()
            y = y.to_numpy()
        if number >= len(self.plot_items):
            item = pg.PlotCurveItem(
                x,
                y,
                pen=pg.mkPen(
                    color, width=2, symbol=symbols[symbol], style=linestyles[linestyle]
                ),
            )
            self.image_plot.addItem(item)
            self.plot_items.append(item)
        else:
            item = self.plot_items[number]
            item.setData(x, y)
            item.setPen(
                pg.mkPen(
                    color, width=2, symbol=symbols[symbol], style=linestyles[linestyle]
                )
            )


def ask_for_input_box(values):
    item, ok = QtWidgets.QInputDialog.getItem(
        None, "Select Entry", "Select one of the following:", values, editable=False
    )
    if ok and item:
        return item
    return values[0]


def run_viewer():
    app = QtWidgets.QApplication([])
    set_theme()
    sys.excepthook = exception_hook
    window = CAMELS_Viewer()
    window.show()
    app.exec_()


if __name__ == "__main__":
    run_viewer()
