import sys
from importlib import resources

import PySide6
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt
import h5py
import numpy as np
import copy

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

bolder_font = QtGui.QFont()
bolder_font.setBold(True)
bolder_font.setPointSize(11)


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

        self.xy_plot = self.graphics_view.addPlot()

        self.graphics_view.nextRow()
        self.image_plot = self.graphics_view.addPlot()
        self.image_plot.hide()
        self.image = pg.ImageItem()
        self.image_plot.addItem(self.image)

        pen = pg.mkPen("r", width=2)
        self.image_ROI = pg.RectROI(
            [0, 0],
            [1, 1],
            movable=True,
            translateSnap=True,
            maxBounds=None,
            parent=self.image_plot,
            pen=pen,
        )
        self.image_ROI.removeHandle(self.image_ROI.getHandles()[-1])
        self.image_plot.addItem(self.image_ROI)
        self.image_ROI.sigRegionChanged.connect(self._image_roi_moved)

        self.image_info_text = pg.LabelItem()
        # self.image_plot.addItem(self.image_info_text)
        self.image_info_text.setParentItem(self.image_plot)
        self.image_info_text.setText("test")
        self.image_info_text.anchor(itemPos=(0.4, 0.5), parentPos=(0.4, 0.5))

        self.intensity_line_lo = pg.InfiniteLine(pos=1, pen="r", movable=True)
        self.intensity_line_hi = pg.InfiniteLine(pos=2, pen="r", movable=True)

        # self.scatter = pg.ScatterPlotItem()
        # self.image_plot.addItem(self.scatter)
        # self.dummy_image = pg.ImageItem()

        # self.colorbar = pg.ColorBarItem()
        # self.colorbar.setImageItem(self.dummy_image)

        self.histogram = pg.HistogramLUTItem()
        self.graphics_view.addItem(self.histogram)
        self.histogram.autoHistogramRange()
        self.histogram.hide()
        self.histogram.setImageItem(self.image)

        self.graphics_view.nextRow()
        self.roi_intensity_plot = self.graphics_view.addPlot()
        self.roi_intensity_plot.hide()

        self.pos_line_1d = pg.InfiniteLine(pos=0, pen="r", movable=True)
        self.pos_line_1d.sigPositionChanged.connect(self._pos_line_moved)
        self.roi_intensity_plot.addItem(self.pos_line_1d)

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
        self.plot_table.clicked.connect(self.check_change)

        self.multi_selection_widget = QtWidgets.QWidget()

        self.image_xlabel = QtWidgets.QLabel()
        self.image_ylabel = QtWidgets.QLabel()
        self.image_xlabel.setFont(bolder_font)
        self.image_ylabel.setFont(bolder_font)
        self.image_x_values = []
        self.image_y_values = []
        self.last_x = 0
        self.last_y = 0

        layout.addWidget(self.plot_table, 1, 0, 1, 2)
        layout.addWidget(self.image_xlabel, 2, 0)
        layout.addWidget(self.image_ylabel, 2, 1)
        layout.addWidget(self.multi_selection_widget, 10, 0, 1, 2)
        self.options_layout = layout

        self.data = {}
        self.plot_items = []
        self.image_data = None

        self.showMaximized()

        if not PANDAS_INSTALLED:
            raise Exception(
                "Pandas is not installed. Please install pandas to use all functionality"
            )

    def check_change(self, index):
        c = index.column()
        if c == 0:
            self._add_or_change_plot_data(index.row())
        self.make_multi_selection_widget(index.row())

    def make_multi_selection_widget(self, number):
        x_selection = self.plot_table.cellWidget(number, 1).currentText()
        y_selection = self.plot_table.cellWidget(number, 2).currentText()
        data = self._get_current_data(number)
        widget = Multi_Selection_Widget(
            data, x_selection=x_selection, y_selection=y_selection
        )
        self.options_layout.replaceWidget(self.multi_selection_widget, widget)
        self.multi_selection_widget.deleteLater()
        self.multi_selection_widget = widget
        self.multi_selection_widget.filter_signal.connect(self.update_image)
        self.multi_selection_widget.x_selection_signal.connect(
            lambda text=None: self.update_image(number)
        )
        self.multi_selection_widget.y_selection_signal.connect(
            lambda text=None: self.update_image(number)
        )

    def add_table_row(self, data, fname="", entry_name=""):
        row = self.plot_table.rowCount()
        self.plot_table.setRowCount(row + 1)
        self.plot_table.setItem(row, 0, QtWidgets.QTableWidgetItem())
        self.plot_table.item(row, 0).setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        self.plot_table.item(row, 0).setCheckState(Qt.Checked)

        data_sets = list(data.keys())
        box1 = QtWidgets.QComboBox()
        box1.addItems(data_sets)
        self.plot_table.setCellWidget(row, 3, box1)

        box2 = QtWidgets.QComboBox()
        self.plot_table.setCellWidget(row, 1, box2)
        box3 = QtWidgets.QComboBox()

        self.plot_table.setCellWidget(row, 2, box3)
        box4 = QtWidgets.QComboBox()
        box4.addItems(list(matplotlib_default_colors.keys()))
        box4.setCurrentIndex(row % len(matplotlib_default_colors))
        self.plot_table.setCellWidget(row, 4, box4)
        box5 = QtWidgets.QComboBox()
        box5.addItems(list(symbols.keys()))
        box5.setCurrentText("none")
        self.plot_table.setCellWidget(row, 5, box5)
        box6 = QtWidgets.QComboBox()
        box6.addItems(list(linestyles.keys()))

        self.plot_table.setCellWidget(row, 6, box6)
        self.plot_table.setItem(row, 7, QtWidgets.QTableWidgetItem(fname))
        self.plot_table.setItem(row, 8, QtWidgets.QTableWidgetItem(entry_name))

        self.plot_table.resizeColumnsToContents()

        self._update_x_y_comboboxes(row)
        box1.currentTextChanged.connect(
            lambda text=None, x=row: self._update_x_y_comboboxes(x)
        )
        box2.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        box3.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        box4.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        box5.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        box6.currentTextChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )

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
            data = read_camels_file(
                file_path, entry_key=key, read_all_datasets=True, return_dataframe=False
            )
            self.data[f"{file_path}_{key}"] = data
            self.add_table_row(data=data, fname=file_path, entry_name=key)
        # self.update_plot()

    def update_plot(self):
        self.image_plot.clear()
        self.xy_plot.clear()
        self.roi_intensity_plot.clear()
        self.plot_items.clear()
        for row in range(self.plot_table.rowCount()):
            self._add_or_change_plot_data(row)

    def _get_current_data(self, number, as_dataframe=False):
        file_name = self.plot_table.item(number, 7).text()
        entry_name = self.plot_table.item(number, 8).text()
        data_set = self.plot_table.cellWidget(number, 3).currentText()
        data = copy.deepcopy(self.data[f"{file_name}_{entry_name}"][data_set])
        if as_dataframe:
            import pandas as pd

            for key in data.keys():
                if data[key].ndim > 1:
                    data[key] = [np.array(x) for x in data[key]]
            return pd.DataFrame(data)
        return data

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
        data = self._get_current_data(number)
        try:
            x = data[x_data]
            y = data[y_data]
        except KeyError:
            return
        try:
            x = x.astype(float)
            y = y.astype(float)
        except ValueError:
            print("Could not convert data to float.")
            return
        self.intensity_line_lo.sigPositionChanged.disconnect()
        self.intensity_line_hi.sigPositionChanged.disconnect()
        self.intensity_line_lo.setValue(x.min())
        self.intensity_line_hi.setValue(x.max())
        if x.ndim == 1 and y.ndim == 1:
            if number >= len(self.plot_items):
                item = pg.PlotDataItem(
                    x,
                    y,
                    pen=pg.mkPen(
                        color,
                        width=2,
                        style=linestyles[linestyle],
                    ),
                )
                self.xy_plot.addItem(item)
                self.plot_items.append(item)
            else:
                item = self.plot_items[number]
                item.setData(x, y)
                item.setPen(
                    pg.mkPen(
                        color,
                        width=2,
                        style=linestyles[linestyle],
                    )
                )
            item.setSymbol(symbols[symbol])
            item.setSymbolBrush(pg.mkBrush(color))
            item.setSymbolPen(pg.mkPen(color))
            do_plot = self.plot_table.item(number, 0).checkState()
            if do_plot == Qt.Checked:
                item.show()
            else:
                item.hide()
            self.xy_plot.autoRange()
            self.image_plot.hide()
            self.roi_intensity_plot.hide()
        elif x.ndim == 2 and y.ndim == 2:
            self.make_multi_selection_widget(number)
            self.image_plot.show()
            self.image_plot.setTitle(f"integrated intensity {y_data}")
        else:
            print("Could not plot data, please check the data shapes.")
            return

    def update_intensity_line(self):
        self.image_plot.hide()
        self.histogram.hide()

    def update_image(self, number):
        if not self._update_intensities(number):
            return
        self.intensity_line_hi.sigPositionChanged.disconnect()
        self.intensity_line_lo.sigPositionChanged.disconnect()
        self.intensity_line_lo.sigPositionChanged.connect(
            lambda stat=None, val=number: self._update_intensities(val)
        )
        self.intensity_line_hi.sigPositionChanged.connect(
            lambda stat=None, val=number: self._update_intensities(val)
        )
        self._current_image_number = number
        y_name = self.multi_selection_widget.y_image_box.currentText()
        if y_name == "None":
            self.update_intensity_line()
            return

        # self.image.setImage(np.array([[0, 1], [2, 3]]), levels=[0, 3])
        self.image.show()
        self.image_plot.show()
        self.image_plot.enableAutoRange()
        self.histogram.show()
        self.image_plot.autoRange()
        self.image_ROI.setPos((0, 0))
        self._image_roi_moved()

    def _update_intensities(self, number):
        data = self._get_current_data(number, as_dataframe=True)
        x_name = self.plot_table.cellWidget(number, 1).currentText()
        y_name = self.plot_table.cellWidget(number, 2).currentText()
        x_ax = self.multi_selection_widget.x_image_box.currentText()
        y_ax = self.multi_selection_widget.y_image_box.currentText()
        self.image.clear()
        self.roi_intensity_plot.clear()
        if x_ax == y_ax:
            self.image_info_text.setText("Select different axes for the image.")
            self.image_info_text.show()
            return False
        try:
            if y_ax == "None":
                sorted_data = data.sort_values([x_ax])
            else:
                sorted_data = data.sort_values([x_ax, y_ax])
        except Exception:
            self.image_info_text.setText(
                "Could not make an image of the axes,\nplease check the data and your selection."
            )
            self.image_info_text.show()
            return False
        x_data = sorted_data[x_name]
        y_data = sorted_data[y_name]
        x_ax_data = sorted_data[x_ax]
        if y_ax != "None":
            y_ax_data = sorted_data[y_ax]
        else:
            y_ax_data = np.zeros((1))
        if x_ax_data.ndim != 1 or y_ax_data.ndim != 1:
            self.image_info_text.setText("Please select 1D data for x and y axes.")
            self.image_info_text.show()
            return False
        lo_pos = self.intensity_line_lo.value()
        hi_pos = self.intensity_line_hi.value()
        if lo_pos > hi_pos:
            self.intensity_line_lo.setValue(hi_pos)
        intensities = []
        for i, y_val in enumerate(y_data):
            x_val = x_data[i]
            lo_filtered = np.where(x_val >= lo_pos)
            x_lo = x_val[lo_filtered]
            y_lo = y_val[lo_filtered]
            hi_filtered = np.where(x_lo <= hi_pos)
            x_filtered = x_lo[hi_filtered]
            y_filtered = y_lo[hi_filtered]

            val = np.trapezoid(y_filtered, x=x_filtered)
            intensities.append(val)
        intensities = np.array(intensities)
        try:
            self.image_x_values = sorted(list(set(x_ax_data)))
            self.image_y_values = sorted(list(set(y_ax_data)))
            xlen = len(self.image_x_values)
            ylen = len(self.image_y_values)
            self.image_data = np.reshape(intensities, (xlen, ylen))
        except Exception as e:
            print(e)
            self.image_info_text.setText(
                "Error: incompatible data shapes.\nYou may need to select other axes for the image."
            )
            self.image_info_text.show()
            return False

        self.sorted_data = sorted_data
        self.image_info_text.hide()
        if ylen > 1:
            self.image.setImage(
                self.image_data, levels=[np.min(intensities), np.max(intensities)]
            )
        else:
            try:
                self.roi_intensity_plot.plot(
                    x_ax_data, intensities, pen=pg.mkPen(width=2), symbol="o"
                )
                self.roi_intensity_plot.show()
                self.roi_intensity_plot.autoRange()
                self.roi_intensity_plot.addItem(self.pos_line_1d)
            except Exception as e:
                print(e)
                self.image_info_text.setText(f"Error: {e}")
                self.image_info_text.show()
                return False
        return True

    def _pos_line_moved(self):
        x = self.pos_line_1d.value()
        if min(self.image_x_values) <= x <= max(self.image_x_values):
            self.last_x = x
        else:
            self.pos_line_1d.setValue(self.last_x)
            return
        closest_x = np.abs(np.array(self.image_x_values) - x).argmin()
        xpos = self.image_x_values[closest_x]
        x_ax_name = self.multi_selection_widget.x_image_box.currentText()
        x_text = f"{x_ax_name} = {xpos}"
        self.image_xlabel.setText(x_text)
        self.image_ylabel.setText("")

        x_name = self.plot_table.cellWidget(self._current_image_number, 1).currentText()
        y_name = self.plot_table.cellWidget(self._current_image_number, 2).currentText()
        x_data = self.sorted_data[x_name][self.sorted_data[x_ax_name] == xpos]
        y_data = self.sorted_data[y_name][self.sorted_data[x_ax_name] == xpos]
        try:
            x_data = x_data.to_numpy()[0]
            y_data = y_data.to_numpy()[0]
        except Exception:
            pass
        self.xy_plot.clear()
        self.xy_plot.plot(x_data, y_data)
        self.xy_plot.addItem(self.intensity_line_lo)
        self.xy_plot.addItem(self.intensity_line_hi)

    def _image_roi_moved(self):
        x, y = [int(val) for val in self.image_ROI.pos()]
        if 0 <= x < len(self.image_x_values) and 0 <= y < len(self.image_y_values):
            self.last_x = x
            self.last_y = y
        else:
            self.image_ROI.setPos((self.last_x, self.last_y))
            return
        xpos = self.image_x_values[x]
        ypos = self.image_y_values[y]
        x_ax_name = self.multi_selection_widget.x_image_box.currentText()
        y_ax_name = self.multi_selection_widget.y_image_box.currentText()
        x_text = f"{x_ax_name} = {xpos}"
        y_text = f"{y_ax_name} = {ypos}"
        self.image_xlabel.setText(x_text)
        self.image_ylabel.setText(y_text)

        x_name = self.plot_table.cellWidget(self._current_image_number, 1).currentText()
        y_name = self.plot_table.cellWidget(self._current_image_number, 2).currentText()
        x_data = self.sorted_data[x_name][
            (self.sorted_data[x_ax_name] == xpos)
            & (self.sorted_data[y_ax_name] == ypos)
        ]
        y_data = self.sorted_data[y_name][
            (self.sorted_data[x_ax_name] == xpos)
            & (self.sorted_data[y_ax_name] == ypos)
        ]
        try:
            x_data = x_data.to_numpy()[0]
            y_data = y_data.to_numpy()[0]
        except Exception:
            pass
        self.xy_plot.clear()
        self.xy_plot.plot(x_data, y_data)
        self.xy_plot.addItem(self.intensity_line_lo)
        self.xy_plot.addItem(self.intensity_line_hi)


class Multi_Selection_Widget(QtWidgets.QWidget):
    filter_signal = QtCore.Signal(dict)
    x_selection_signal = QtCore.Signal(str)
    y_selection_signal = QtCore.Signal(str)

    def __init__(self, data, parent=None, x_selection=None, y_selection=None):
        super().__init__(parent)
        layout = QtWidgets.QGridLayout()
        self.data = data
        self.setLayout(layout)
        self.x_selection = x_selection
        self.y_selection = y_selection

        self.x_image_box = QtWidgets.QComboBox()
        self.y_image_box = QtWidgets.QComboBox()

        self.keys = list(data.keys())
        self.keys.remove(self.x_selection)
        if self.x_selection != self.y_selection:
            self.keys.remove(self.y_selection)
        self.x_image_box.addItems(self.keys)
        self.y_image_box.addItems(["None"] + self.keys)
        self.x_image_box.currentTextChanged.connect(
            lambda text=None: self.x_selection_signal.emit(
                self.x_image_box.currentText()
            )
        )
        self.y_image_box.currentTextChanged.connect(
            lambda text=None: self.y_selection_signal.emit(
                self.y_image_box.currentText()
            )
        )

        layout.addWidget(QtWidgets.QLabel("image X:"), 0, 0)
        layout.addWidget(self.x_image_box, 0, 1)
        layout.addWidget(QtWidgets.QLabel("image Y:"), 1, 0)
        layout.addWidget(self.y_image_box, 1, 1)

        self.filter_checks = {}
        self.filter_boxes = {}
        i = 2
        for key in self.keys:
            if key == self.x_selection or key == self.y_selection:
                continue
            if data[key].ndim != 1:
                continue
            if len(set(data[key])) < 2:
                continue
            check = QtWidgets.QCheckBox(f"filter {key}")
            self.filter_checks[key] = check
            box = QtWidgets.QComboBox()
            self.filter_boxes[key] = box
            box.addItems(sorted([str(x) for x in set(data[key])]))
            box.currentTextChanged.connect(self._update_filters)
            check.stateChanged.connect(self._update_filters)
            layout.addWidget(check, i, 0)
            layout.addWidget(box, i, 1)
            i += 1

    def get_filters(self):
        filters = {}
        for key in self.filter_checks:
            if self.filter_checks[key].isChecked():
                filters[key] = self.filter_boxes[key].currentText()
        return filters

    def _update_filters(self):
        filters = self.get_filters()
        self.filter_signal.emit(filters)


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
