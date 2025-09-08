import h5py
import json
import lmfit
import numpy as np
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data_reader import read_camels_file, decide_entry_key
from .utils.fit_variable_renaming import replace_name
from .utils.string_evaluation import evaluate_string


def _recursive_plots_from_sub_protocol_dict(own_name, protocol_info):
    """Create a dictionary to accumulate plot information for the given protocol"""
    plot_info = {}
    primary_plots = protocol_info["plots"]
    if primary_plots:
        plot_info[own_name] = primary_plots
    # Iterate over each step in the protocol.
    for step, step_info in protocol_info["loop_step_dict"].items():
        name = (
            f'{own_name}/{step_info["name"]}'
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


def recreate_plots(
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
                lines=dict(dash="dash"),
            ),
            secondary_y=y_axis == "right",
        )
    except Exception as e:
        warnings.warn(
            f'Could not plot the fit {func} for {y} vs {x} in the stream "{stream}".\n'
            f"Please check the fit parameters and the data.\n{e}"
        )
