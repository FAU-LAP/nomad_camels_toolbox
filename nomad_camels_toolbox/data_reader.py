import h5py


def read_camels_file(file_path):
    """Reads a CAMELS file and returns a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the file to be read.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the data read from the file.
    """
