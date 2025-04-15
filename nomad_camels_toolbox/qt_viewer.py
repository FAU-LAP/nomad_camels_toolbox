"""
CAMELS Data Viewer - NOMAD CAMELS Toolbox

This module implements a graphical user interface to visualize CAMELS data files.
It uses PySide6 for the GUI, pyqtgraph for interactive plotting, and h5py/numpy for
data handling. The viewer supports drag-and-drop file loading, multiple plot types,
and interactive image/intensity analysis.
"""

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
from utils.string_evaluation import evaluate_string
import graphics

from data_reader import read_camels_file, PANDAS_INSTALLED


dark_mode = False


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

# Symbols recognized by pyqtgraph.
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

# Linestyles recognized by pyqtgraph.
linestyles = {
    "solid": Qt.PenStyle.SolidLine,
    "dashed": Qt.PenStyle.DashLine,
    "dash-dot": Qt.PenStyle.DashDotLine,
    "dash-dot-dot": Qt.PenStyle.DashDotDotLine,
    "dotted": Qt.PenStyle.DotLine,
    "none": Qt.PenStyle.NoPen,
}

# Define dark theme palette settings.
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

# Define light theme palette settings.
light_palette = QtGui.QPalette(QtGui.QColor(225, 225, 225), QtGui.QColor(238, 238, 238))
light_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))

# Define a bolder and larger font style for specific labels.
bolder_font = QtGui.QFont()
bolder_font.setBold(True)
bolder_font.setPointSize(11)


def set_theme(set_dark_mode=None):
    """
    Set the application's theme based on the dark_mode flag.

    Parameters:
        dark_mode (bool): If True, set dark theme. Otherwise, use light theme.
    """
    global dark_mode
    if set_dark_mode is not None:
        dark_mode = set_dark_mode
    main_app = QtWidgets.QApplication.instance()
    if dark_mode:
        # For dark mode, configure pyqtgraph with dark background.
        pg.setConfigOptions(background="k", foreground="w")
        palette = dark_palette
    else:
        # For light mode, configure pyqtgraph with light background.
        pg.setConfigOptions(background="w", foreground="k")
        palette = light_palette
    main_app.setPalette(palette)
    main_app.setStyle("Fusion")


class DragDropGraphicLayoutWidget(pg.GraphicsLayoutWidget):
    """
    Custom GraphicsLayoutWidget that supports drag-and-drop of files.

    Signal:
        dropped (list): Emits a list of file paths dropped onto the widget.
    """

    dropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        """
        Handle the drag enter event to allow file drops.
        """
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """
        Handle the drag move event by setting the drop action (for visuals).
        """
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        Handle the drop event by extracting file paths and emitting them.
        """
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            # Convert each dropped URL to local file path.
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            # Emit the list of file paths.
            self.dropped.emit(links)
        else:
            event.ignore()


def _get_color(color):
    """
    Get the RGB color value for a given color name.

    Parameters:
        color (str): Name of the color.

    Returns:
        tuple: RGB color value.
    """
    if color in ["red", "r"]:
        if dark_mode:
            return (75, 0, 0)
        return (255, 180, 180)
    if color in ["green", "g"]:
        if dark_mode:
            return (0, 75, 0)
        return (180, 255, 180)
    if color in ["white", "w"]:
        if dark_mode:
            return (0, 0, 0)
        return (255, 255, 255)
    if color in ["black", "b"]:
        if dark_mode:
            return (255, 255, 255)
        return (0, 0, 0)


class Box_and_Line_Widget(QtWidgets.QWidget):
    textChanged = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.box = QtWidgets.QComboBox()

        self.line = QtWidgets.QLineEdit()

        layout.addWidget(self.box)
        layout.addWidget(self.line)

        self.box.currentTextChanged.connect(self._text_changed)
        self.line.textChanged.connect(self._line_changed)
        self._box_max_width = self.box.maximumWidth()

        self._box_items = []

    def clear(self):
        """
        Clear the combo box and line edit.
        """
        self.box.clear()
        self.line.clear()

    def _line_changed(self, text):
        if not text:
            self.line.setStyleSheet(f'background-color: rgb{_get_color("white")}')
            return
        try:
            evaluate_string(text, self._box_items)
            self.line.setStyleSheet(f'background-color: rgb{_get_color("green")}')
            self._text_changed(text)
        except Exception:
            self.line.setStyleSheet(f'background-color: rgb{_get_color("red")}')

    def _text_changed(self, text):
        """
        Emit the textChanged signal with the current text.

        Parameters:
            text (str): The current text from the combo box or line edit.
        """
        if self.box.currentText() == "custom value":
            text = self.line.text()
            self.box.maximumWidth
            self.box.setMaximumWidth(20)
            self.line.show()
        else:
            text = self.box.currentText()
            self.box.setMaximumWidth(self._box_max_width)
            self.line.hide()
        self.textChanged.emit(text)

    def addItems(self, items):
        """
        Add items to the combo box.

        Parameters:
            items (list): List of items to add to the combo box.
        """
        self.box.addItems(items + ["custom value"])
        self._box_items = items

    def setText(self, text):
        """
        Set the text of the line edit.

        Parameters:
            text (str): The text to set in the line edit.
        """
        if text in self._box_items:
            self.box.setCurrentText(text)
        else:
            self.box.setCurrentText("custom value")
            self.line.setText(text)

    def currentText(self):
        """
        Get the current text from the combo box or line edit.

        Returns:
            str: The current text.
        """
        if self.box.currentText() == "custom value":
            return self.line.text()
        return self.box.currentText()


class CAMELS_Viewer(QtWidgets.QMainWindow):
    """
    Main window class for the CAMELS Data Viewer application.

    This class creates a user interface which allows the user to load CAMELS data
    files, visualize 1D plots or integrated image data, and interactively filter and
    analyze the data.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Viewer - NOMAD CAMELS Toolbox")
        self.setWindowIcon(
            QtGui.QIcon(str(resources.files(graphics) / "CAMELS_icon.png"))
        )

        # Create the main graphics view with drag and drop functionality.
        self.graphics_view = DragDropGraphicLayoutWidget()
        self.graphics_view.dropped.connect(self.load_data)

        # Create a left widget with a grid layout for control elements.
        self.left_widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        self.left_widget.setLayout(layout)

        # Use a splitter as the central widget to separate controls and plots.
        self.setCentralWidget(QtWidgets.QSplitter())
        self.centralWidget().addWidget(self.left_widget)
        self.centralWidget().addWidget(self.graphics_view)

        # Create the primary 2D plot for x-y data.
        self.xy_plot = self.graphics_view.addPlot()

        # Add a second row for the image plot.
        self.graphics_view.nextRow()
        self.image_plot = self.graphics_view.addPlot()
        self.image_plot.hide()  # Hide until image mode is active.
        self.image = pg.ImageItem()
        self.image_plot.addItem(self.image)

        # Create a rectangular ROI (Region Of Interest) for the image plot.
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
        # Remove extra handle from the ROI.
        self.image_ROI.removeHandle(self.image_ROI.getHandles()[-1])
        self.image_plot.addItem(self.image_ROI)
        self.image_ROI.sigRegionChanged.connect(self._image_roi_moved)

        # Create and configure label for displaying error information.
        self.image_info_text = pg.LabelItem()
        self.image_info_text.setParentItem(self.image_plot)
        self.image_info_text.anchor(itemPos=(0.4, 0.5), parentPos=(0.4, 0.5))

        # Create draggable infinite lines for intensity level boundaries.
        self.intensity_line_lo = pg.InfiniteLine(pos=1, pen="r", movable=True)
        self.intensity_line_hi = pg.InfiniteLine(pos=2, pen="r", movable=True)

        # Create a histogram LUT item for image intensity control.
        self.histogram = pg.HistogramLUTItem()
        self.graphics_view.addItem(self.histogram)
        self.histogram.autoHistogramRange()
        self.histogram.hide()
        self.histogram.setImageItem(self.image)

        # Add a row for the ROI intensity plot.
        self.graphics_view.nextRow()
        self.roi_intensity_plot = self.graphics_view.addPlot()
        self.roi_intensity_plot.hide()

        # Add an infinite line to the intensity plot for user interaction.
        self.pos_line_1d = pg.InfiniteLine(pos=0, pen="r", movable=True)
        self.pos_line_1d.sigPositionChanged.connect(self._pos_line_moved)
        self.roi_intensity_plot.addItem(self.pos_line_1d)

        # Button to load measurement files via a file dialog.
        self.load_measurement_button = QtWidgets.QPushButton("Load Measurement")
        self.load_measurement_button.clicked.connect(self.load_measurement)

        self.dark_mode_box = QtWidgets.QCheckBox("Dark Mode")
        self.dark_mode_box.setChecked(False)
        self.dark_mode_box.stateChanged.connect(self._dark_mode_toggle)

        # Create a table widget to list and manage multiple plots.
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

        # Widget for multi-selection options for images and filters.
        self.multi_selection_widget = QtWidgets.QWidget()

        # Labels to display image axis information.
        self.image_xlabel = QtWidgets.QLabel()
        self.image_ylabel = QtWidgets.QLabel()
        self.image_xlabel.setFont(bolder_font)
        self.image_ylabel.setFont(bolder_font)
        self.image_x_values = []
        self.image_y_values = []
        self.last_x = 0
        self.last_y = 0

        # Add widgets to the left-side layout.
        layout.addWidget(self.load_measurement_button, 0, 0)
        layout.addWidget(self.dark_mode_box, 0, 1)
        layout.addWidget(self.plot_table, 1, 0, 1, 2)
        layout.addWidget(self.image_xlabel, 2, 0)
        layout.addWidget(self.image_ylabel, 2, 1)
        layout.addWidget(self.multi_selection_widget, 10, 0, 1, 2)
        self.options_layout = layout

        # Data containers for loaded dataset and created plot items.
        self.data = {}
        self.plot_items = []
        self.image_data = None

        self.showMaximized()
        self._last_plot_type = None

        # Ensure that pandas is installed.
        if not PANDAS_INSTALLED:
            raise Exception(
                "Pandas is not installed. Please install pandas to use all functionality"
            )

    def _dark_mode_toggle(self, state):
        """
        Toggle the dark mode of the application.

        Parameters:
            state (bool): If True, enable dark mode. Otherwise, disable it.
        """
        if state:
            set_theme(True)
            bg = "k"
            fg = "w"
        else:
            set_theme(False)
            bg = "w"
            fg = "k"
        self.graphics_view.setBackground(bg)
        for plot in [
            self.xy_plot,
            self.image_plot,
            self.roi_intensity_plot,
        ]:
            for axis in ["left", "bottom", "right", "top"]:
                ax = plot.getAxis(axis)
                ax.setPen(pg.mkPen(fg))
                ax.setTextPen(pg.mkPen(fg))
        # change text color of histogram
        self.histogram.axis.setPen(pg.mkPen(fg))
        self.histogram.axis.setTextPen(pg.mkPen(fg))

    def check_change(self, index):
        """
        Handler for clicks on the plot table. Updates the plot based on changes.

        Parameters:
            index (QModelIndex): Index of the cell clicked.
        """
        c = index.column()
        if c == 0:  # If the checkbox column was clicked.
            self._add_or_change_plot_data(index.row())
        self.make_multi_selection_widget(index.row())

    def make_multi_selection_widget(self, number):
        """
        Create and display the multi-selection widget to set image axis selections
        and data filters.

        Parameters:
            number (int): Row index from the plot table for which to configure the widget.
        """
        x_selection = self.plot_table.cellWidget(number, 1).currentText()
        y_selection = self.plot_table.cellWidget(number, 2).currentText()
        data = self._get_current_data(number)
        widget = Multi_Selection_Widget(
            data, x_selection=x_selection, y_selection=y_selection
        )
        # Replace the existing multi-selection widget.
        self.options_layout.replaceWidget(self.multi_selection_widget, widget)
        self.multi_selection_widget.deleteLater()
        self.multi_selection_widget = widget
        # Connect signals to update the image when selections change.
        self.multi_selection_widget.filter_signal.connect(
            lambda filters=None: self.update_image(number)
        )
        self.multi_selection_widget.x_selection_signal.connect(
            lambda text=None: self.update_image(number)
        )
        self.multi_selection_widget.y_selection_signal.connect(
            lambda text=None: self.update_image(number)
        )

    def add_table_row(self, data, fname="", entry_name=""):
        """
        Add a new row to the plot table to represent a new dataset/plot.

        Parameters:
            data (dict): The loaded CAMELS data.
            fname (str): File name of the loaded data.
            entry_name (str): Specific entry name within the file.
        """
        row = self.plot_table.rowCount()
        self.plot_table.setRowCount(row + 1)

        # Create a checkable item to enable/disable plotting.
        self.plot_table.setItem(row, 0, QtWidgets.QTableWidgetItem())
        self.plot_table.item(row, 0).setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        self.plot_table.item(row, 0).setCheckState(Qt.Checked)

        # Create combo boxes for data-set and axis selections.
        data_sets = list(data.keys())
        box1 = QtWidgets.QComboBox()
        box1.addItems(data_sets)
        self.plot_table.setCellWidget(row, 3, box1)

        combo_x = Box_and_Line_Widget()
        self.plot_table.setCellWidget(row, 1, combo_x)

        combo_y = Box_and_Line_Widget()
        self.plot_table.setCellWidget(row, 2, combo_y)

        # Create combo boxes for color, symbol, and linestyle.
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

        # Display file name and entry information.
        self.plot_table.setItem(row, 7, QtWidgets.QTableWidgetItem(fname))
        self.plot_table.setItem(row, 8, QtWidgets.QTableWidgetItem(entry_name))

        self.plot_table.resizeColumnsToContents()

        self._update_x_y_comboboxes(row)
        # Connect changes in combo boxes to update the plot.
        box1.currentTextChanged.connect(
            lambda text=None, x=row: self._update_x_y_comboboxes(x)
        )
        combo_x.textChanged.connect(
            lambda text=None, x=row: self._add_or_change_plot_data(x)
        )
        combo_y.textChanged.connect(
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
        """
        Update the X and Y axis combo boxes based on the selected dataset.

        Parameters:
            row (int): The row in the plot table to update.
        """
        fname = self.plot_table.item(row, 7).text()
        entry_name = self.plot_table.item(row, 8).text()
        data = self.data[f"{fname}_{entry_name}"]
        data_set_name = self.plot_table.cellWidget(row, 3).currentText()
        data_set = data[data_set_name]
        data_keys = list(data_set.keys())
        # Update X axis combo box.
        widget = self.plot_table.cellWidget(row, 1)
        widget.clear()
        widget.addItems(data_keys)
        # Update Y axis combo box.
        widget = self.plot_table.cellWidget(row, 2)
        widget.clear()
        widget.addItems(data_keys)
        self._add_or_change_plot_data(row)

    def load_measurement(self):
        """
        Open a file dialog to select and load one or multiple measurement files.
        """
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            self.load_data(file_paths)

    def load_data(self, file_paths):
        """
        Load data from each given file path.

        Parameters:
            file_paths (list): List of paths to HDF5/NeXus files.
        """
        for file_path in file_paths:
            # Open the HDF5 file.
            with h5py.File(file_path, "r") as f:
                keys = list(f.keys())
                # Choose an appropriate key that does not start with "NeXus_".
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
            # Read the CAMELS file data.
            data = read_camels_file(
                file_path, entry_key=key, read_all_datasets=True, return_dataframe=False
            )
            self.data[f"{file_path}_{key}"] = data
            self.add_table_row(data=data, fname=file_path, entry_name=key)

    def update_plot(self):
        """
        Refresh the entire plot area by clearing and re-adding all plot items.
        """
        self.image_plot.clear()
        self.image_plot.addItem(self.image)
        self.image_plot.addItem(self.image_ROI)
        self.xy_plot.clear()
        self.roi_intensity_plot.clear()
        self.plot_items.clear()
        for row in range(self.plot_table.rowCount()):
            self._add_or_change_plot_data(row)

    def _get_current_data(self, number, as_dataframe=False):
        """
        Retrieve the current dataset for the specified table row.

        Parameters:
            number (int): Row number in the plot table.
            as_dataframe (bool): Return data as pandas DataFrame if True.

        Returns:
            dict or pandas.DataFrame: The current dataset.
        """
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
        """
        Add a new plot or update an existing one based on the current selections.

        Parameters:
            number (int): Row number in the plot table.
        """
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
        if x_data in data:
            x = data[x_data]
        else:
            try:
                x = evaluate_string(x_data, data)
            except Exception as e:
                print(f"Could not evaluate x data: {x_data}\n{e}")
                return
        if y_data in data:
            y = data[y_data]
        else:
            try:
                y = evaluate_string(y_data, data)
            except Exception as e:
                print(f"Could not evaluate y data: {y_data}\n{e}")
                return
        try:
            x = x.astype(float)
            y = y.astype(float)
        except ValueError:
            print("Could not convert data to float.")
            return
        except AttributeError:
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

        # Disconnect intensity line signals to prevent recursive updates.
        self.intensity_line_lo.sigPositionChanged.disconnect()
        self.intensity_line_hi.sigPositionChanged.disconnect()

        # Set the intensity lines to data range.
        self.intensity_line_lo.setValue(x.min())
        self.intensity_line_hi.setValue(x.max())

        if x.ndim == 1 and y.ndim == 1:
            if self._last_plot_type != "1D":
                self._last_plot_type = "1D"
                self.update_plot()
                return
            # 1D plot: create or update a plot data item.
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
            # Set the marker symbol and brush.
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
            self.histogram.hide()
            self.image_xlabel.setText("")
            self.image_ylabel.setText("")
            self.intensity_line_lo.hide()
            self.intensity_line_hi.hide()
            self.roi_intensity_plot.hide()
            self.multi_selection_widget.hide()
        elif x.ndim == 2 and y.ndim == 2:
            # 2D plot (integrated image) requires the multi-selection widget.
            self.make_multi_selection_widget(number)
            self.image_plot.show()
            self.image.show()
            self.image_plot.setTitle(f"integrated intensity {y_data}")
            self.intensity_line_lo.show()
            self.intensity_line_hi.show()
            self._last_plot_type = "2D"
        else:
            print("Could not plot data, please check the data shapes.")
            return

    def update_intensity_line(self):
        """
        Hides image-related plots.
        """
        self.image_plot.hide()
        self.histogram.hide()

    def update_image(self, number):
        """
        Update the image plot using filters and current selections from the multi-selection widget.

        Parameters:
            number (int): Row number in the plot table.
        """
        if not self._update_intensities(number):
            return
        # Reconnect intensity line signals for interactivity.
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

        # Show and configure image plot.
        self.image.show()
        self.image_plot.show()
        self.image_plot.enableAutoRange()
        self.histogram.show()
        self.image_plot.autoRange()
        self.image_ROI.setPos((0, 0))
        self.image_ROI.show()
        self._image_roi_moved()

    def _update_intensities(self, number):
        """
        Compute integrated intensity over the selected x-range and update the image.

        Parameters:
            number (int): Row number in the plot table.

        Returns:
            bool: True if update is successful, False otherwise.
        """
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
            self.image_plot.show()
            self.roi_intensity_plot.hide()
            return False
        try:
            if y_ax == "None":
                sorted_data = data.sort_values([x_ax])
            else:
                sorted_data = data.sort_values([x_ax, y_ax])
        except Exception as e:
            self.image_info_text.setText(
                f"Could not make an image of the axes,\nplease check the data and your selection.\n{e}"
            )
            self.image_info_text.show()
            self.image_plot.show()
            self.roi_intensity_plot.hide()
            return False
        # Apply filters from the multi-selection widget.
        filters = self.multi_selection_widget.get_filters()
        if filters:
            for key in filters:
                filter_val = filters[key]
                # Convert to the data type of the column.
                try:
                    filter_val = sorted_data[key].dtype.type(filter_val)
                except Exception as e:
                    print(e)
                    pass
                sorted_data = sorted_data[sorted_data[key] == filter_val]
        # Check if filtering left any data.
        if sorted_data.empty:
            self.image_info_text.setText(
                "No data left after filtering.\nCheck your filters!"
            )
            self.image_info_text.show()
            self.image_plot.show()
            self.roi_intensity_plot.hide()
            return False

        if x_name in sorted_data:
            x_data = sorted_data[x_name]
        else:
            try:
                x_data = evaluate_string(x_name, sorted_data)
                sorted_data[x_name] = x_data
            except Exception as e:
                print(f"Could not evaluate x data: {x_name}\n{e}")
                return False
        if y_name in sorted_data:
            y_data = sorted_data[y_name]
        else:
            try:
                y_data = evaluate_string(y_name, sorted_data)
                sorted_data[y_name] = y_data
            except Exception as e:
                print(f"Could not evaluate y data: {y_name}\n{e}")
                return False
        x_ax_data = sorted_data[x_ax]
        if y_ax != "None":
            y_ax_data = sorted_data[y_ax]
        else:
            y_ax_data = np.zeros((1))
        if x_ax_data.ndim != 1 or y_ax_data.ndim != 1:
            self.image_info_text.setText("Please select 1D data for x and y axes.")
            self.image_info_text.show()
            self.image_plot.show()
            self.roi_intensity_plot.hide()
            return False
        # Get the positions from the intensity lines.
        lo_pos = self.intensity_line_lo.value()
        hi_pos = self.intensity_line_hi.value()
        if lo_pos > hi_pos:
            self.intensity_line_lo.setValue(hi_pos)
        intensities = []
        # Integrate intensity for each data entry within the selected range.
        for i, y_val in y_data.items():
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
            # Determine unique x and y values for the image grid.
            self.image_x_values = sorted(list(set(x_ax_data)))
            self.image_y_values = sorted(list(set(y_ax_data)))
            xlen = len(self.image_x_values)
            ylen = len(self.image_y_values)
            self.image_data = np.reshape(intensities, (xlen, ylen))
        except Exception as e:
            self.image_info_text.setText(
                f"Error: incompatible data shapes.\nYou may need to select other axes for the image.\n{e}"
            )
            self.image_info_text.show()
            self.image_plot.show()
            self.roi_intensity_plot.hide()
            return False

        self.sorted_data = sorted_data
        self.image_info_text.hide()
        # If more than one y value, display image; otherwise, use a plot.
        if ylen > 1:
            self.image.setImage(
                self.image_data, levels=[np.min(intensities), np.max(intensities)]
            )
            self.image_plot.show()
            self.histogram.show()
            self.image_plot.autoRange()
            self.histogram.autoHistogramRange()
            self.roi_intensity_plot.hide()
        else:
            try:
                self.roi_intensity_plot.plot(
                    x_ax_data.to_numpy(), intensities, pen=pg.mkPen(width=2), symbol="o"
                )
                self.roi_intensity_plot.show()
                self.roi_intensity_plot.autoRange()
                self.roi_intensity_plot.addItem(self.pos_line_1d)
                self.image_plot.hide()
                self.histogram.hide()
            except Exception as e:
                print(e)
                self.image_info_text.setText(f"Error: {e}")
                self.image_info_text.show()
                self.image_plot.show()
                self.roi_intensity_plot.hide()
                return False
        return True

    def _pos_line_moved(self):
        """
        Handler for when the horizontal line in the 1D intensity plot is moved.

        It updates the x position and refreshes the x-y plot accordingly.
        """
        x = self.pos_line_1d.value()
        if min(self.image_x_values) <= x <= max(self.image_x_values):
            self.last_x = x
        else:
            if self.last_x < min(self.image_x_values):
                self.last_x = min(self.image_x_values)
            elif self.last_x > max(self.image_x_values):
                self.last_x = max(self.image_x_values)
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
        self.xy_plot.plot(x_data, y_data, pen=pg.mkPen(color=_get_color("black")))
        self.xy_plot.addItem(self.intensity_line_lo)
        self.xy_plot.addItem(self.intensity_line_hi)

    def _image_roi_moved(self):
        """
        Handler for when the image ROI is moved.

        It updates the x-y labels and the x-y plot based on the current ROI position.
        """
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
        self.xy_plot.plot(x_data, y_data, pen=pg.mkPen(color=_get_color("black")))
        self.xy_plot.addItem(self.intensity_line_lo)
        self.xy_plot.addItem(self.intensity_line_hi)


class Multi_Selection_Widget(QtWidgets.QWidget):
    """
    A widget for selecting image axes and applying filters to the dataset.

    Provides combo boxes to choose which columns to use as X and Y axes for
    generating integrated images and a set of checkboxes and combo boxes for filtering.
    """

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

        # Create combo boxes for selecting image axes.
        self.x_image_box = QtWidgets.QComboBox()
        self.y_image_box = QtWidgets.QComboBox()

        self.keys = list(data.keys())
        # Remove x-y plot keys.
        if self.x_selection in self.keys:
            self.keys.remove(self.x_selection)
        if self.y_selection in self.keys:
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
        # Update filters when axis selections change.
        self.x_image_box.currentTextChanged.connect(self._enable_filters)
        self.y_image_box.currentTextChanged.connect(self._enable_filters)

        layout.addWidget(QtWidgets.QLabel("image X:"), 0, 0)
        layout.addWidget(self.x_image_box, 0, 1)
        layout.addWidget(QtWidgets.QLabel("image Y:"), 1, 0)
        layout.addWidget(self.y_image_box, 1, 1)

        # Create check boxes and combo boxes for filtering data.
        self.filter_checks = {}
        self.filter_boxes = {}
        i = 2
        for key in self.keys:
            if key == self.x_selection or key == self.y_selection:
                continue
            # Only consider one-dimensional data with multiple unique values.
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
        """
        Retrieve current filters based on checked options.

        Returns:
            dict: A dictionary mapping data keys to selected filter values.
        """
        x = self.x_image_box.currentText()
        y = self.y_image_box.currentText()
        filters = {}
        for key in self.filter_checks:
            if key == x or key == y:
                continue
            if self.filter_checks[key].isChecked():
                filters[key] = self.filter_boxes[key].currentText()
        return filters

    def _enable_filters(self):
        """
        Enable or disable filter controls based on current axis selections.
        """
        x = self.x_image_box.currentText()
        y = self.y_image_box.currentText()
        for key in self.filter_checks:
            if key == x or key == y:
                self.filter_checks[key].setEnabled(False)
                self.filter_boxes[key].setEnabled(False)
            else:
                self.filter_checks[key].setEnabled(True)
                self.filter_boxes[key].setEnabled(True)

    def _update_filters(self):
        """
        Emit the current filters whenever a filter control changes.
        """
        filters = self.get_filters()
        self.filter_signal.emit(filters)


def ask_for_input_box(values):
    """
    Open a dialog for the user to select one option from a list.

    Parameters:
        values (list): List of possible values.

    Returns:
        str: The value selected by the user.
    """
    item, ok = QtWidgets.QInputDialog.getItem(
        None, "Select Entry", "Select one of the following:", values, editable=False
    )
    if ok and item:
        return item
    return values[0]


def run_viewer():
    """
    Initialize and run the CAMELS Viewer application.
    """
    app = QtWidgets.QApplication([])
    set_theme()
    sys.excepthook = exception_hook  # Set the exception hook for debugging.
    window = CAMELS_Viewer()
    window.show()
    app.exec_()


if __name__ == "__main__":
    run_viewer()
