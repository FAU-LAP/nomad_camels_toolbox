from .data_reader import read_camels_file

try:
    from .plotting import recreate_plots
except ImportError:
    pass

try:
    from .qt_viewer import run_viewer
except ImportError:
    pass


print(
    "Imported the nomad_camels_toolbox, for documentation see https://fau-lap.github.io/NOMAD-CAMELS/doc/nomad_camels_toolbox.html"
)
