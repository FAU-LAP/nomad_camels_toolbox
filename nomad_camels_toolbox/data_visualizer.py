from data_reader import read_camels_file, _ask_for_selection
import h5py

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def recreate_all_plots(file_path, entry_key: str = ""):
    """
    Recreate all plots from a data set.

    Parameters
    ----------
    file_path : str
        The path to the data set.
    entry_key : str, optional
        The key of the entry in the data set.

    Returns
    -------
    None
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required to use this function.\nInstall it via 'pip install matplotlib'."
        )
    with h5py.File(file_path, "r") as f:
        keys = list(f.keys())
        if entry_key in keys:
            key = entry_key
        elif entry_key:
            raise ValueError(
                f'The key "{entry_key}" you specified was not found in the file.'
            )
        elif len(keys) > 1:
            remaining_keys = []
            for key in keys:
                if not key.startswith("NeXus_"):
                    remaining_keys.append(key)
            if len(remaining_keys) > 1:
                key = _ask_for_selection(remaining_keys)
            else:
                key = remaining_keys[0]
        else:
            key = keys[0]
        data = f[key]["data"]
        plot_infos = {}
        if "axes" in data.attrs and "signal" in data.attrs:
            plot_infos["primary_stream"] = {
                "x": data.attrs["axes"],
                "y": [data.attrs["signal"]],
            }
        if "auxiliary_signals" in data.attrs:
            plot_infos["primary_stream"]["y"] += data.attrs["auxiliary_signals"]
        for key in f[key]["data"]:
            pass
    data, fit_dict = read_camels_file(file_path, entry_key=entry_key, return_fits=True)
    for key in data:
        pass
