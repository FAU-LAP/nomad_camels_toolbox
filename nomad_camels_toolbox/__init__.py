from .data_reader import read_camels_file

try:
    from .plotting import recreate_plots
except ImportError as e:
    print(f"Could not import nomad-camels-toolbox plotting module:\n{e}")

try:
    from .qt_viewer import run_viewer
except ImportError as e:
    print(f"Could not import nomad-camels-toolbox qt_viewer module:\n{e}")


print(
    "Imported the nomad_camels_toolbox, for documentation see https://fau-lap.github.io/NOMAD-CAMELS/doc/nomad_camels_toolbox.html"
)
