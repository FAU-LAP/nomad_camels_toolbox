import h5py
import json
import lmfit
import numpy as np
import warnings
import sys
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from packaging import version

from data_reader import read_camels_file, decide_entry_key
from utils.fit_variable_renaming import replace_name
from utils.string_evaluation import evaluate_string


def _wrap_recursive(node: ast.expr, source_str: str) -> str:
    """
    Internal recursive helper function to traverse the AST.

    We split on low-precedence operators (+, -) and treat all
    other expressions (like 'abc**2' or '(a*b)') as "atoms".
    """

    # --- Recursive Case ---
    # Check if the node is a Binary Operation AND its operator is
    # low-precedence (Add or Subtract).
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        # This is a good place to split!
        # Recursively call the function on the left and right children.
        left_str = _wrap_recursive(node.left, source_str)
        right_str = _wrap_recursive(node.right, source_str)

        # Reconstruct the operator. We add spaces for readability.
        # This loses original spacing (e.g. " + " vs "+") but is
        # far more robust than trying to find the operator in the source.
        op_char = "+" if isinstance(node.op, ast.Add) else "-"

        # Return the joined string with the <br> tag
        return f"{left_str}<br> {op_char} {right_str}"

    # --- Base Case ---
    # If the node is NOT a low-precedence BinOp, we treat it as an "atom".
    # This includes:
    # 1. Names (e.g., 'abc')
    # 2. Constants (e.g., '2', '12')
    # 3. High-precedence BinOps (e.g., 'abc**2', '3/12')
    # 4. Expressions in parentheses (e.g., '(a+b)')
    else:
        # We return its *exact* original source segment.
        # This is the magic that prevents "strange cut-offs" and
        # correctly keeps 'abc**2' or '(a+b)' together.
        try:
            return ast.get_source_segment(source_str, node)
        except Exception as e:
            # This can happen on very old Python versions or complex/unsupported
            # AST nodes. We'll include a fallback.
            print(f"Warning: Could not get source segment (requires Python 3.8+). {e}")
            # ast.unparse is available in 3.9+ and is a decent fallback.
            if hasattr(ast, "unparse"):
                return ast.unparse(node)
            return "[parsing error]"


def wrap_arithmetic_string(source_str: str) -> str:
    """
    Wraps a long arithmetic string with <br> tags at logical
    break points (before + and -) using an Abstract Syntax Tree.

    This method correctly handles operator precedence.

    Args:
        source_str: The arithmetic string (e.g., "abc**2+xyz-3/12").

    Returns:
        The wrapped string with <br> tags, or the original
        string if parsing fails.

    Note:
        Requires Python 3.8+ for best results.
    """
    if not source_str:
        return ""
    if len(source_str) < 20:
        return source_str  # No need to wrap short strings
    try:
        # 'eval' mode is used for a single expression
        tree = ast.parse(source_str, mode="eval")

        # tree.body is the top-level expression node
        return _wrap_recursive(tree.body, source_str)
    except SyntaxError as e:
        print(f"Error: Invalid arithmetic string. {e}")
        return source_str  # Fallback to original string
    except Exception as e:
        print(f"An error occurred during wrapping: {e}")
        return source_str  # Fallback


def _recursive_plots_from_sub_protocol_dict(own_name, protocol_info):
    """Create a dictionary to accumulate plot information for the given protocol"""
    plot_info = {}
    primary_plots = protocol_info["plots"]
    if primary_plots:
        plot_info[own_name] = primary_plots
    # Iterate over each step in the protocol.
    for step, step_info in protocol_info["loop_step_dict"].items():
        name = (
            f"{own_name}/{step_info['name']}"
            if own_name != "primary"
            else step_info["name"]
        )
        if "plots" in step_info:
            # If plots exist for this step, add them to the dictionary keyed by the step name.
            plot_info[name] = step_info["plots"]
        elif "_sub_protocol_dict" in step_info:
            # Recurse into any subprotocol dictionaries and merge the result.
            plot_info.update(
                _recursive_plots_from_sub_protocol_dict(
                    name, step_info["_sub_protocol_dict"]
                )
            )
    return plot_info


def find_all_paths(data_structure, target_key):
    """
    Traverses a nested dictionary and list structure and returns
    all paths to entries with a specific key.

    Args:
        data_structure (dict or list): The nested structure to search.
        target_key (str): The key name to search for (e.g., "plots").

    Returns:
        list: A list of paths, where each path is a list of
              keys and list indices.
    """
    found_paths = []

    # We use a stack for an iterative Depth-First Search (DFS)
    # Each item on the stack is a tuple: (node, path_to_node)
    stack = [(data_structure, [])]

    while stack:
        current_node, current_path = stack.pop()

        # --- Case 1: The current node is a dictionary ---
        if isinstance(current_node, dict):
            for key, value in current_node.items():
                # The new path is the path *to this key*
                new_path = current_path + [key]

                if key == target_key:
                    # Found it! Add the path to our results.
                    found_paths.append(new_path)

                # Push the *value* (the child node) onto the stack
                # to be explored next.
                stack.append((value, new_path))

        # --- Case 2: The current node is a list ---
        elif isinstance(current_node, list):
            # Iterate through the list with its index
            for index, value in enumerate(current_node):
                # The new path uses the list index
                new_path = current_path + [index]

                # Push the *value* (the list item) onto the stack
                # to be explored next.
                stack.append((value, new_path))

        # --- Base Case: Node is not a dict or list (e.g., str, int) ---
        # We do nothing, as it can't contain more keys.

    return found_paths


def get_camels_suitcase_version(file_path):
    """Get the version of suitcase-nomad-camels-hdf5 used to create the CAMELS file.

    Parameters
    ----------
    file_path : str
        Path to the CAMELS file.

    Returns
    -------
    str
        The version string of suitcase-nomad-camels-hdf5, or "unknown" if not found.
    """
    suitcase_version = "unknown"
    try:
        with h5py.File(file_path, "r") as f:
            for key in f:
                if key.endswith("_entry"):
                    camels_group = f[key]
                    suitcase_version = camels_group[
                        "program/python_environment/suitcase-nomad-camels-hdf5"
                    ][()].decode("utf-8")
    except Exception as e:
        warnings.warn(f"Could not read suitcase version from file: {e}")
    return suitcase_version


def recreate_plots(
    file_path,
    entry_key: str = "",
    data_set_key: str = "",
    show_figures=True,
    force=None,
):
    """Recreate plots from a CAMELS file as Plotly figures.
    This is the new version that fully supports nested protocols. It only works with files created with suitcase-nomad-camels-hdf5 >= 1.0.0
    This is always used to try and read CAMELS HDF5 files unless the first sniffing detects an older version of suitcase-nomad-camels-hdf5.

    Parameters
    ----------
    file_path : str
        Path to the CAMELS file.
    entry_key : str, optional
        The entry key to use for reading the file. If not provided, the first entry will be used.
    data_set_key : str, optional
        --- DEPRECATED ---
    show_figures : bool, optional
        If True, the figures will be displayed. Default is True.
    force : str, optional
        If "legacy", forces the use of the legacy method. If "current", forces the use of the new method.


    Returns
    -------
    dict
        A dictionary containing the recreated figures, keyed by their names.
    """
    if force == "legacy":
        print("Using the legacy reading method as forced by the user.")
        return recreate_plots_legacy(
            file_path,
            entry_key=entry_key,
            data_set_key=data_set_key,
            show_figures=show_figures,
        )
    elif force == "current":
        pass  # continue with the current method
    else:
        suitcase_version = get_camels_suitcase_version(file_path)

        # If the suitcase version is older than 1.0.0, fall back to the legacy method
        if version.parse(suitcase_version) < version.parse("1.0.0"):
            warnings.warn(
                f"The CAMELS file was created with suitcase-nomad-camels-hdf5 version {suitcase_version}. "
                "Falling back to the legacy plot recreation method. "
                "For full support of nested protocols, please re-export the data using suitcase-nomad-camels-hdf5 >= 1.0.0",
                UserWarning,
                stacklevel=2,
            )
            return recreate_plots_legacy(
                file_path,
                entry_key=entry_key,
                data_set_key=data_set_key,
                show_figures=show_figures,
            )
    # Continue with the new method
    if data_set_key:
        warnings.warn(
            "'data_set_key' is deprecated for newer CAMELS files using the CAMELS suitcase (data export) > 1.0.0 and is ignored",
            DeprecationWarning,
            stacklevel=2,
        )
    with h5py.File(file_path, "r") as f:
        key = decide_entry_key(f, entry_key)
    list_of_plot_paths = find_plot_paths(file_path, key=key)
    # order the list so that it goes plot_1, plot_2, plot_3, ...
    list_of_plot_paths.sort(key=lambda x: int(x.split("plot_")[-1].split("/")[0]))
    built_plots = build_plots_from_paths(file_path, list_of_plot_paths)
    if show_figures:
        for fig in built_plots.values():
            fig.show()
    return built_plots


def build_plots_from_paths(file_path, list_of_plot_paths):
    """Build Plotly figures from a list of plot paths in a CAMELS file. Also builds the fits if they are present.

    Parameters
    ----------
    file_path : str
        Path to the CAMELS file.
    list_of_plot_paths : list
        A list of full string paths to plot entries in the HDF5 file.

    Returns
    -------
    dict
        A dictionary containing the recreated figures, keyed by their names.
    """
    figures = {}
    with h5py.File(file_path, "r") as f:
        for plot_path in list_of_plot_paths:
            plot_group = f[plot_path]
            # Check if its a 1D plot
            if "_plot_data_axes" in plot_group:  # This is a 1D plot
                # Check if any signal should be plotted on the secondary y-axis

                for entry in plot_group.values():
                    if (
                        "y_axes_index" in entry.attrs
                        and entry.attrs["y_axes_index"] == 2
                    ):
                        has_second = True
                        break
                    else:
                        has_second = False
                if has_second:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.update_layout(
                        title=f"Plot from {plot_path}",
                        showlegend=True,
                        xaxis_title=plot_group["_plot_data_axes"].attrs["long_name"],
                    )

                else:
                    fig = make_subplots()
                    fig.update_layout(
                        title=f"Plot from {plot_path}",
                        showlegend=True,
                        xaxis_title=plot_group["_plot_data_axes"].attrs["long_name"],
                    )

                # Get the x axis data
                x_data = plot_group["_plot_data_axes"][()]
                # Plot all signals in the plot group
                for entry in plot_group.values():
                    if "_plot_data_signal" in entry.name:
                        y_data = plot_group[entry.name.split("/")[-1]][()]
                        y_axis_index = entry.attrs["y_axes_index"]
                        if y_axis_index == 1:
                            secondary_y = False
                        else:
                            secondary_y = True
                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_data,
                                mode="markers",
                                name=wrap_arithmetic_string(entry.attrs["long_name"]),
                            ),
                            secondary_y=secondary_y,
                        )
                        # Add left y axis label
                        if y_axis_index == 1:
                            fig.update_yaxes(
                                title_text=entry.attrs["long_name"], secondary_y=False
                            )
                        # Add right y axis label
                        elif y_axis_index == 2:
                            fig.update_yaxes(
                                title_text=entry.attrs["long_name"], secondary_y=True
                            )
                if "fit" in plot_group:
                    fit_group = plot_group["fit"]
                    for fit_entry in fit_group.values():
                        plot_metadata = json.loads(
                            fit_entry.attrs.get("plot_metadata", "{}")
                        )
                        # Get the type of fit
                        if plot_metadata["use_custom_func"]:
                            func = plot_metadata["custom_func"]
                            model = lmfit.models.ExpressionModel(func)
                        else:
                            func = plot_metadata["predef_func"]
                            model = lmfit.models.lmfit_models[func]()
                        # Create the params of the model
                        params = model.make_params()
                        # Set the parameters from the fit entry
                        for param in params:
                            params[param].set(value=fit_entry[param][0])
                        y_fit_data = model.eval(params=params, x=x_data)
                        y_axis_index = fit_entry.attrs["y_axes_index"]
                        if y_axis_index == 1:
                            secondary_y = False
                        else:
                            secondary_y = True
                        fig.add_trace(
                            go.Scatter(
                                x=x_data,
                                y=y_fit_data,
                                mode="lines",
                                name=wrap_arithmetic_string("Fit" + plot_metadata["y"]),
                                line=dict(dash="dash"),
                            ),
                            secondary_y=secondary_y,
                        )

            elif "_plot_data_axes_0" in plot_group:  # This is a 2D plot
                x_data = plot_group["_plot_data_axes_0"][()]
                y_data = plot_group["_plot_data_axes_1"][()]
                z_data = plot_group["_plot_data_signal"][()]
                fig = go.Figure(
                    data=go.Heatmap(
                        x=x_data,
                        y=y_data,
                        z=z_data,
                        colorscale="Viridis",
                        colorbar=dict(
                            title=plot_group["_plot_data_signal"].attrs["long_name"]
                        ),
                        showscale=True,
                    )
                )
                # Add x and y axes labels
                fig.update_layout(
                    title=f"2D Plot from {plot_path}",
                    xaxis_title=plot_group["_plot_data_axes_0"].attrs["long_name"],
                    yaxis_title=plot_group["_plot_data_axes_1"].attrs["long_name"],
                )
            # Set small text fonts as the labels can be long
            fig.update_layout(
                font=dict(size=9),  # affects all text elements by default
                title_font=dict(size=9),
            )
            figures[plot_path] = fig
    return figures


def find_plot_paths(filepath, key=""):
    """
    Walks an entire HDF5 file and returns a list of full paths
    to entries that are of class 'NXdata' and whose names
    start with 'plot_'.

    Args:
        filepath (str): The path to the HDF5 file.

    Returns:
        list: A list of full string paths to matching entries.
    """
    found_paths = []

    def check_node(name, obj):
        """
        This is a callback function called by visititems for every object.
        'name' is the full path, 'obj' is the HDF5 object (Group or Dataset).
        """

        # 1. Check the name criteria
        # We get the base name (the last part of the path)
        basename = name.split("/")[-1]

        if basename.startswith("plot_"):
            # 2. Check the class criteria
            # NeXus classes are stored in an attribute named 'NX_class'
            if "NX_class" in obj.attrs:
                # Read the attribute
                nx_class = obj.attrs["NX_class"]

                # Attributes can be stored as bytes, so we decode if necessary
                if isinstance(nx_class, bytes):
                    nx_class = nx_class.decode("utf-8")

                if nx_class == "NXdata":
                    # If both criteria match, add the full path to our list
                    found_paths.append(obj.name)

    # --- Main execution ---
    try:
        with h5py.File(filepath, "r") as f:
            # .visititems() recursively visits every item in the file
            # and calls 'check_node' for each one.
            f[key].visititems(check_node)

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr)
    except OSError as e:
        print(
            f"Error: Could not read file. Is it a valid HDF5 file? \n{e}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

    return found_paths


def recreate_plots_legacy(
    file_path, entry_key: str = "", data_set_key: str = "", show_figures=True
):
    """Recreate plots from a CAMELS file as Plotly figures.

    Parameters
    ----------
    file_path : str
        Path to the CAMELS file.
    entry_key : str, optional
        The entry key to use for reading the file. If not provided, the first entry will be used.
    data_set_key : str, optional
        The dataset key to use for reading the file. If not provided, all datasets will be used.
    show_figures : bool, optional
        If True, the figures will be displayed. Default is True.


    Returns
    -------
    dict
        A dictionary containing the recreated figures, keyed by their names.
    """

    # Open the file and load the measurement protocol JSON.
    with h5py.File(file_path, "r") as f:
        key = decide_entry_key(f, entry_key)
        protocol_json = f[key]["measurement_details/protocol_json"][()].decode("utf-8")
    # Parse the protocol JSON into a Python dictionary.
    protocol_info = json.loads(protocol_json)
    # Retrieve all plot information from the protocol.
    plot_info = _recursive_plots_from_sub_protocol_dict("primary", protocol_info)
    if not plot_info:
        print(
            "No plot info found in the file.\n"
            "It might be that no plots were defined for the measurement.\n"
            "Caveat: Plots for subprotocols only work from CAMELS version 1.8.3 onwards."
        )
        return None
    # Load the data from the file using the data_reader.
    if not data_set_key:
        # Read all datasets if no specific one is provided.
        data = read_camels_file(
            file_path, entry_key=key, read_all_datasets=True, return_fits=True
        )
    else:
        data = {
            data_set_key: read_camels_file(
                file_path, entry_key=key, data_set_key=data_set_key, return_fits=True
            )
        }

    figures = {}
    # Iterate over each stream and its associated plots.
    for stream, plots in plot_info.items():
        if stream not in data:
            warnings.warn(
                f'The stream "{stream}" you specified was not found in the data.\n'
                "Check the available streams in the file."
            )
            continue
        df = data[stream][0]
        fit_data = data[stream][1]
        for plot in plots:
            if plot["plt_type"] == "X-Y plot":
                y_names = plot["y_axes"]["formula"]
                y_axes = plot["y_axes"]["axis"]
                x_name = plot["x_axis"]
                # Create a subplot with a secondary y-axis if necessary.
                if "right" in y_axes:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    if plot["ylabel2"]:
                        y2_name = plot["ylabel2"]
                    else:
                        # Use the corresponding y value if no label provided.
                        index = y_axes.index("right")
                        y2_name = y_names[index]
                    fig.update_layout(yaxis2_title=y2_name)
                else:
                    fig = make_subplots()
                # Update general layout properties.
                fig.update_layout(
                    title=plot["name"],
                    xaxis_title=plot["xlabel"] or x_name,
                    yaxis_title=plot["ylabel"] or y_names[0],
                )
                # Retrieve x data from the DataFrame or evaluate the expression.
                if x_name in df:
                    x_data = df[x_name]
                else:
                    x_data = evaluate_string(x_name, df)
                # Loop over each y value to add them as separate traces.
                for i, y_name in enumerate(y_names):
                    y_axis = y_axes[i]
                    if y_name in df:
                        y_data = df[y_name]
                    else:
                        y_data = evaluate_string(y_name, df)
                    fig.add_trace(
                        go.Scatter(x=x_data, y=y_data, mode="markers", name=y_name),
                        secondary_y=y_axis == "right",
                    )
                # Handle fits if defined.
                if plot["same_fit"] and plot["all_fit"]["do_fit"]:
                    fit = plot["all_fit"]
                    _make_fit(
                        fit,
                        fit_data,
                        df,
                        plot["y_axes"],
                        stream,
                        fig,
                        is_all_fit=True,
                    )
                else:
                    for fit in plot["fits"]:
                        if not fit["do_fit"]:
                            continue
                        _make_fit(
                            fit,
                            fit_data,
                            df,
                            plot["y_axes"],
                            stream,
                            fig,
                        )
                name = f"{stream}: {plot['name']}"
                fig.update_layout(
                    legend=dict(
                        orientation="h",  # or "v" depending on your preference
                        yanchor="bottom",
                        y=1.02,  # just above the plot area
                        xanchor="right",
                        x=1,
                    ),
                    margin=dict(l=40, r=40, t=40, b=40),  # adjust margins if necessary
                )
                figures[name] = fig
            elif plot["plt_type"] == "2D plot":
                # For 2D plots, prepare x, y and z data; evaluate strings if needed.
                if plot["x_axis"] in df:
                    x_data = df[plot["x_axis"]]
                else:
                    x_data = evaluate_string(plot["x_axis"], df)
                if plot["y_axes"]["formula"][0] in df:
                    y_data = df[plot["y_axes"]["formula"][0]]
                else:
                    y_data = evaluate_string(plot["y_axes"]["formula"][0], df)
                if plot["z_axis"] in df:
                    z_data = df[plot["z_axis"]]
                else:
                    z_data = evaluate_string(plot["z_axis"], df)
                # Create a colormesh (or a heatmap) from the x, y and z data.
                mesh = _make_colormesh(x_data, y_data, z_data)
                if mesh:
                    fig = go.Figure(
                        data=go.Heatmap(
                            x=mesh[0].flatten(),
                            y=mesh[1].flatten(),
                            z=mesh[2].flatten(),
                            colorscale="Viridis",
                            colorbar=dict(
                                title=plot["zlabel"]
                                or plot["z_axis"],  # Use z label or z axis name
                            ),
                            showscale=True,
                        )
                    )
                else:
                    # Fallback to a scatter plot if colormesh cannot be created.
                    fig = go.Figure(
                        data=go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode="markers",
                            marker=dict(
                                color=z_data,  # Use z values for color
                                colorscale="Viridis",  # Specify the colorscale
                                colorbar=dict(
                                    title=plot["zlabel"]
                                    or plot["z_axis"],  # Use z label or z axis name
                                ),  # Optionally add a colorbar
                                showscale=True,
                            ),
                        )
                    )
                # Update layout to include axis labels and title
                fig.update_layout(
                    title=plot["name"],
                    xaxis_title=plot["xlabel"] or plot["x_axis"],
                    yaxis_title=plot["ylabel"] or plot["y_axes"]["formula"][0],
                )
                name = f"{stream}: {plot['name']}"
                figures[name] = fig
    if show_figures:
        for fig in figures.values():
            fig.show()
    return figures


def _make_colormesh(x_data, y_data, z_data):
    """Create a colormesh (or a heatmap) from x, y and z data.

    Parameters
    ----------
    x_data : array-like
        The x data for the plot.
    y_data : array-like
        The y data for the plot.
    z_data : array-like
        The z data for the plot.


    Returns
    -------
    tuple or None
        A tuple containing the reshaped x, y, and z data if successful, otherwise None.
    """
    # Determine the shape of unique x and y data.
    x_shape = len(set(x_data))
    y_shape = len(set(y_data))
    # If both shapes are unavailable, return None indicating failure.
    if x_shape is None and y_shape is None:
        return None
    elif x_shape is not None and y_shape is None:
        y_shape = int(np.array(x_data).size / x_shape)
    elif x_shape is None and y_shape is not None:
        x_shape = int(np.array(y_data).size / y_shape)
    try:
        # Reshape the x, y and z arrays into 2D arrays for plotting.
        x = np.array(x_data).reshape((x_shape, y_shape))
        y = np.array(y_data).reshape((x_shape, y_shape))
        c = np.array(z_data).reshape((x_shape, y_shape))
        return x, y, c
    except Exception as e:
        # If reshaping fails, return None.
        return None


def _make_fit(fit_info, fit_data, df, y_axes, stream, figure, is_all_fit=False):
    if fit_info["use_custom_func"]:
        func = fit_info["custom_func"]
        model = lmfit.models.ExpressionModel(func)
    else:
        func = fit_info["predef_func"]
        model = lmfit.models.lmfit_models[func]()
    params = model.make_params()
    if is_all_fit:
        for i, y in enumerate(y_axes["formula"]):
            _make_single_fit(
                func,
                y,
                fit_info["x"],
                stream,
                params,
                model,
                df,
                fit_data,
                y_axes["axis"][i],
                figure,
            )
    else:
        y_axis = y_axes["axis"][y_axes["formula"].index(fit_info["y"])]
        _make_single_fit(
            func,
            fit_info["y"],
            fit_info["x"],
            stream,
            params,
            model,
            df,
            fit_data,
            y_axis,
            figure,
        )


def _make_single_fit(func, y, x, stream, params, model, df, fit_data, y_axis, figure):
    try:
        fit_name = "_".join((func, y, "v", x, stream))
        fit_name = fit_name.replace("/", "||sub_stream||")
        fit_name = replace_name(fit_name)
        for param in params:
            params[param].set(value=fit_data[fit_name][param])
        if x in df:
            x_data = df[x].values
        else:
            x_data = evaluate_string(x, df).values
        if len(x_data) < 100:
            x_data = np.linspace(x_data.min(), x_data.max(), 100)
        y_data = model.eval(params=params, x=x_data)
        figure.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                name=fit_name,
                line=dict(dash="dash"),
            ),
            secondary_y=y_axis == "right",
        )
    except Exception as e:
        warnings.warn(
            f'Could not plot the fit {func} for {y} vs {x} in the stream "{stream}".\n'
            f"Please check the fit parameters and the data.\n{e}"
        )
        
