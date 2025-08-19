import h5py
import numpy as np

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
    read_all_datasets: bool = False,
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
        Entry-Key to read. If not specified and there is more than one entry, the user is asked to select one. Usually files only have one entry "CAMELS_entry", i.e. this is not needed.
    return_dataframe : bool, optional (default: True)
        Whether to return the data as a pandas DataFrame. Requires pandas to be installed if pandas is not installed, this parameter is ignored.
    read_variables : bool, optional (default: True)
        Whether to read the variables from the data set.
    return_fits : bool, optional (default: False)
        Whether to return the fits of the data set.
    read_all_datasets : bool, optional (default: False)
        Whether to read all datasets in the file. If True, the data_set_key parameter is ignored. If True, a dictionary with the data sets is returned.

    Returns
    -------
    data : dict or pd.DataFrame
        The data from the data set.
    fit_dict : dict
        The fits of the data set, only returned if return_fits is True.
    """
    with h5py.File(file_path, "r") as f:
        key = decide_entry_key(f, entry_key)
        if read_all_datasets:
            data = {}
            groups = ["primary"]
            for group in f[key]["data"]:
                if isinstance(f[key]["data"][group], h5py.Group) and group != "fits":
                    groups.append(group)
            for data_set_key in groups:
                data[data_set_key] = _read_dataset(
                    f[key]["data"],
                    data_set_key,
                    return_dataframe=return_dataframe,
                    read_variables=read_variables,
                    return_fits=return_fits,
                )
            return data
        if (
            data_set_key
            and data_set_key not in f[key]["data"]
            and data_set_key != "primary"
        ) or not data_set_key:
            print(
                f'The data set "{data_set_key}" you specified was not found in the data.'
            )
            groups = ["primary"]
            for group in f[key]["data"]:
                if isinstance(f[key]["data"][group], h5py.Group) and group != "fits":
                    groups.append(group)
            if len(groups) > 1:
                data_set_key = _ask_for_selection(groups)
            else:
                data_set_key = groups[0]
        return _read_dataset(
            f[key]["data"],
            data_set_key,
            return_dataframe=return_dataframe,
            read_variables=read_variables,
            return_fits=return_fits,
        )


def _read_dataset(
    data_group,
    dataset_name,
    return_dataframe: bool = PANDAS_INSTALLED,
    read_variables: bool = True,
    return_fits: bool = False,
):
    if dataset_name == "primary":
        data_set = data_group
    else:
        data_set = data_group[dataset_name]
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
        try:
            try:
                df = pd.DataFrame(data)
            except ValueError:
                data = _change_arrays_to_lists(data)
                df = pd.DataFrame(data)
                for col in df.columns:  # Convert lists back into arrays
                    if isinstance(df[col].iloc[0], list):
                        df[col] = df[col].apply(np.array)
            try:
                fit_df = pd.DataFrame(fit_dict)
            except ValueError:
                fit_dict = _change_arrays_to_lists(fit_dict)
                fit_df = pd.DataFrame(fit_dict)
                for col in fit_df.columns:  # Convert lists back into arrays
                    if isinstance(fit_df[col].iloc[0], list):
                        fit_df[col] = fit_df[col].apply(np.array)
            if return_fits:
                return df, fit_df
            return df
        except Exception as e:
            print(
                "An error occurred while trying to convert the data to a pandas DataFrame. Returning the data as a dictionary instead."
            )
            print(e)
    if return_fits:
        return data, fit_dict
    return data


def _change_arrays_to_lists(data):
    """Changes arrays in a dictionary to lists. This is necessary for creating a pandas DataFrame from the data if the arrays have different shapes.

    Parameters
    ----------
    data : dict
        The data to change.

    Returns
    -------
    dict
        The data with arrays changed to lists.
    """
    for key, value in data.items():
        if key.endswith("_variable_signal"):
            data[key] = _change_arrays_to_lists(value)
        elif isinstance(value, np.ndarray) and value.ndim > 1:
            data[key] = value.tolist()
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


def decide_entry_key(file_handle, entry_key: str = ""):
    keys = list(file_handle.keys())
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
    return key


def h5_group_to_dict(group):
    """

    Parameters
    ----------
    group :


    Returns
    -------

    """
    data = {}
    for k in group.keys():
        if k == "data":
            continue
        if isinstance(group[k], h5py.Group):
            data[k] = h5_group_to_dict(group[k])
        else:
            # read dataset and handle scalar data
            if len(group[k].shape) == 0:
                data[k] = group[k][()]
            else:
                data[k] = group[k][:]
                if data[k].dtype == "|S28":
                    data[k] = data[k].astype(str)
                data[k].tolist()
            # if data[k] is of type bytes, convert it to string
            if isinstance(data[k], bytes):
                data[k] = data[k].decode("utf-8")
    return data


f = r"C:\Users\od93yces\Downloads\rough_adjust_burster_force_sensor_scan.h5"
d = read_camels_file(f, read_all_datasets=True)
print(d)
1
