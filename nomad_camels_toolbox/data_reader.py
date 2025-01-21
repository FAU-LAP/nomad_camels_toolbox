import h5py

try:
    import pandas as pd

    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False


def read_camels_file(
    file_path,
    entry_key: str = "",
    data_set_key: str = "",
    return_dataframe: bool = PANDAS_INSTALLED,
):
    """Reads a CAMELS file and returns a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the file to be read.
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
                key = ask_for_selection(remaining_keys)
            else:
                key = remaining_keys[0]
        else:
            key = keys[0]
        if data_set_key:
            if data_set_key not in f[key]["data"]:
                print(
                    f'The data set "{data_set_key}" you specified was not found in data.'
                )
                groups = []
                for group in f[key]["data"]:
                    if isinstance(f[key]["data"][group], h5py.Group):
                        groups.append(group)
                data_set_key = ask_for_selection(groups)
            data_set = f[key]["data"][data_set_key]
        else:
            data_set = f[key]["data"]
        data = {}
        for key in data_set:
            if not isinstance(data_set[key], h5py.Dataset):
                continue
            data[key] = data_set[key][()]
    if return_dataframe:
        return pd.DataFrame(data)
    return data


def ask_for_selection(values):
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
        selection = int(input("Enter the number of your selection: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return ask_for_selection(values)
    return values[selection]


if __name__ == "__main__":
    file_path = r"C:\Users\od93yces\NOMAD_CAMELS_data\user1\sample1\data_entry_106.nxs"
    data = read_camels_file(file_path)
    print(data)
