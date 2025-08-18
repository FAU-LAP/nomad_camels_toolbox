# NOMAD CAMELS toolbox

This package provides easy access to data generated with [NOMAD CAMELS](https://github.com/FAU-LAP/NOMAD-CAMELS).

Simplify your data evaluation with the NOMAD CAMELS toolbox!

## Optional Dependencies
To install the NOMAD CAMELS toolbox, run
```
pip install nomad-camels-toolbox[all]
```
in the Python environment you use for your evaluation. This installs all optional dependencies to use the full functionality. The following options are all included when using `all`.

Single installation options can be installed by using `pip install nomad-camels-toolbox[option-name]` ([see pip install documentation](https://pip.pypa.io/en/stable/cli/pip_install/)). The options are:
- `pandas`: This installs `pandas` as a powerful package for data evaluation along with the toolbox, so the data can be read directly as a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
- `plotly`: Installs [plotly](https://pypi.org/project/plotly/), [lmfit](https://pypi.org/project/lmfit/) and [pandas](https://pypi.org/project/pandas/). It enables to automatically recreate the plots made in CAMELS, using plotly figures.
- `qt`: Installs [PySide6](https://pypi.org/project/PySide6/) and [pyqtgraph](https://pypi.org/project/pyqtgraph/). This is used to provide a GUI, to quickly investigate data from CAMELS.



# Changelog

### 0.2.7
- made selection of dataset in reader more intuitive

### 0.2.4
- plots now show legend over the plot, not anymore making the plot tiny
- fixed optional dependencies

### 0.2.3
- fixed plots not being made if two plots had the same name

### 0.2.2
Fixes:
- fixed import problems

### 0.2.1
Features:
- custom plots in qt-viewer

Fixes:
- installation fixed
- qt viewer more stable

## 0.2.0
Features:
- Recreating plots from CAMELS with a single function
- UI tool to view data in a simplified way

### 0.1.2
Fixes:
- fixed returning of dataframes for different shapes of the data
