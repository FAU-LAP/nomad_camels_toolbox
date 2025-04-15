import numpy as np
import scipy.constants as const
import pandas as pd

# Create a base namespace with common modules and constants
base_namespace = {"numpy": np, "np": np, "time": 0, "const": const}
# Add all numpy functions to the base namespace
base_namespace.update({name: getattr(np, name) for name in np.__all__})


def evaluate_string(string, df):
    """Evaluates a string expression using the variables in the DataFrame.

    Parameters
    ----------
    string : str
        The string to evaluate.
    df : pandas.DataFrame
        The DataFrame containing the variables.

    Returns
    -------
    float or str
        The evaluated value or an error message.
    """
    string = string.strip()
    # Create a namespace that includes common modules/constants and the DataFrame's series.
    namespace = dict(base_namespace)
    if isinstance(df, pd.DataFrame):
        # If df is a DataFrame, convert it to a dictionary of series.
        namespace.update(df.to_dict(orient="series"))
    elif isinstance(df, dict):
        namespace.update(df)
    elif isinstance(df, list):
        namespace.update({v: 1 for v in df})
    return eval(string, {}, namespace)
