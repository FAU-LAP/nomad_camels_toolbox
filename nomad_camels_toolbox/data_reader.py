import h5py

try:
    import pandas as pd

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False


def read_camels_file(
    file_path,
    data_set_key: str = "",
    entry_key: str = "",
    return_dataframe: bool = PANDAS_INSTALLED,
    read_variables: bool = True,
    return_fits: bool = False,
):
    """
    Read data from a CAMELS file.

    Parameters
    ----------
    file_path : str
        Path to the CAMELS file.
    data_set_key : str, optional (default: "")
        Key of the data set to read. If not specified, the main data set is read.
    entry_key : str, optional (default: "")
        Key of the entry to read. If not specified and there is more than one entry, the user is asked to select one.
    return_dataframe : bool, optional (default: True)
        Whether to return the data as a pandas DataFrame. Requires pandas to be installed if pandas is not installed, this parameter is ignored.
    read_variables : bool, optional (default: True)
        Whether to read the variables from the data set.
    return_fits : bool, optional (default: False)
        Whether to return the fits of the data set.

    Returns
    -------
    data : dict or pd.DataFrame
        The data from the data set.
    fit_dict : dict
        The fits of the data set, only returned if return_fits is True.
    """
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
        if data_set_key:
            if data_set_key not in f[key]["data"]:
                print(
                    f'The data set "{data_set_key}" you specified was not found in the data.'
                )
                groups = ["main dataset"]
                for group in f[key]["data"]:
                    if isinstance(f[key]["data"][group], h5py.Group):
                        groups.append(group)
                if len(groups) > 1:
                    data_set_key = _ask_for_selection(groups)
                else:
                    data_set_key = groups[0]
            if data_set_key == "main dataset":
                data_set = f[key]["data"]
            else:
                data_set = f[key]["data"][data_set_key]
        else:
            data_set = f[key]["data"]
        data = {}
        for key in data_set:
            if (
                read_variables
                and isinstance(data_set[key], h5py.Group)
                and key.endswith("_variable_signal")
            ):
                for sub_key in data_set[key]:
                    data[sub_key] = data_set[key][sub_key][()]
                continue
            if not isinstance(data_set[key], h5py.Dataset):
                continue
            data[key] = data_set[key][()]
        fit_dict = {}
        if return_fits and "fits" in data_set:
            for fit_key in data_set["fits"]:
                fit_dict[fit_key] = {}
                for fit_val in data_set["fits"][fit_key]:
                    fit_dict[fit_key][fit_val] = data_set["fits"][fit_key][fit_val][()]
    if return_dataframe and PANDAS_INSTALLED:
        if fit_dict:
            return pd.DataFrame(data), fit_dict
        return pd.DataFrame(data)
    if fit_dict:
        return data, fit_dict
    return data


def _ask_for_selection(values):
    """Asks the user to select a value from a list of values.

    Parameters
    ----------
    values : list
        List of values to choose from.

    Returns
    -------
    str
        The selected value.
    """
    print("Select one of the following:")
    for i, value in enumerate(values):
        print(f"[{i}]: {value}")
    try:
        given_input = input("Enter the number of your selection: ")
        selection = int(given_input)
        val = values[selection]
    except (ValueError, IndexError):
        print(
            f"Invalid input. Please enter one of the displayed numbers. (Your input: {given_input})"
        )
        return _ask_for_selection(values)
    return val
