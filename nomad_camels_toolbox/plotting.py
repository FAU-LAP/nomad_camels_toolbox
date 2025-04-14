from data_reader import read_camels_file, decide_entry_key
from utils.fit_variable_renaming import replace_name
import h5py
import json
import lmfit
import numpy as np
import scipy.constants as const
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _recursive_plots_from_sub_protocol_dict(own_name, protocol_info):
    plot_info = {}
    primary_plots = protocol_info["plots"]
    if primary_plots:
        plot_info[own_name] = primary_plots
    for step, step_info in protocol_info["loop_step_dict"].items():
        if "plots" in step_info:
            plot_info[step_info["name"]] = step_info["plots"]
        elif "_sub_protocol_dict" in step_info:
            plot_info.update(
                _recursive_plots_from_sub_protocol_dict(
                    step_info["name"], step_info["_sub_protocol_dict"]
                )
            )
    return plot_info


def recreate_plots(
    file_path, entry_key: str = "", data_set_key: str = "", show_figures=True
):
    with h5py.File(file_path, "r") as f:
        key = decide_entry_key(f, entry_key)
        protocol_json = f[key]["measurement_details/protocol_json"][()].decode("utf-8")
    protocol_info = json.loads(protocol_json)
    plot_info = _recursive_plots_from_sub_protocol_dict("primary", protocol_info)
    if not plot_info:
        print(
            "No plot info found in the file.\nIt might be that no plots were defined for the measurement.\nCaveat: Plots for subprotocol only work from CAMELS version 1.8.3 onwards."
        )
        return None
    if not data_set_key:
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
    for stream, plots in plot_info.items():
        if stream not in data:
            warnings.warn(
                f'The stream "{stream}" you specified was not found in the data.\n'
                "Check the available streams in the file."
            )
        df = data[stream][0]
        fit_data = data[stream][1]
        for plot in plots:
            if plot["plt_type"] == "X-Y plot":
                y_names = plot["y_axes"]["formula"]
                y_axes = plot["y_axes"]["axis"]
                x_name = plot["x_axis"]
                if "right" in y_axes:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    if plot["ylabel2"]:
                        y2_name = plot["ylabel2"]
                    else:
                        index = y_axes.index("right")
                        y2_name = y_names[index]
                    fig.update_layout(
                        yaxis2_title=y2_name,
                    )
                else:
                    fig = make_subplots()
                fig.update_layout(
                    title=plot["name"],
                    xaxis_title=plot["xlabel"] or x_name,
                    yaxis_title=plot["ylabel"] or y_names[0],
                )
                if x_name in df:
                    x_data = df[x_name]
                else:
                    x_data = _evaluate_string(x_name, df)
                for i, y_name in enumerate(y_names):
                    y_axis = y_axes[i]
                    if y_name in df:
                        y_data = df[y_name]
                    else:
                        y_data = _evaluate_string(y_name, df)
                    fig.add_trace(
                        go.Scatter(x=x_data, y=y_data, mode="lines", name=y_name),
                        secondary_y=y_axis == "right",
                    )
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
                figures[plot["name"]] = fig
            elif plot["plt_type"] == "2D plot":
                if plot["x_axis"] in df:
                    x_data = df[plot["x_axis"]]
                else:
                    x_data = _evaluate_string(plot["x_axis"], df)
                if plot["y_axes"]["formula"][0] in df:
                    y_data = df[plot["y_axes"]["formula"][0]]
                else:
                    y_data = _evaluate_string(plot["y_axes"]["formula"][0], df)
                if plot["z_axis"] in df:
                    z_data = df[plot["z_axis"]]
                else:
                    z_data = _evaluate_string(plot["z_axis"], df)
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
                figures[plot["name"]] = fig
    if show_figures:
        for fig in figures.values():
            fig.show()
    return figures


def _make_colormesh(x_data, y_data, z_data):
    x_shape = len(set(x_data))
    y_shape = len(set(y_data))
    if x_shape is None and y_shape is None:
        return None
    elif x_shape is not None and y_shape is None:
        y_shape = int(np.array(x_data).size / x_shape)
    elif x_shape is None and y_shape is not None:
        x_shape = int(np.array(y_data).size / y_shape)
    try:
        x = np.array(x_data).reshape((x_shape, y_shape))
        y = np.array(y_data).reshape((x_shape, y_shape))
        c = np.array(z_data).reshape((x_shape, y_shape))
        return x, y, c
    except Exception as e:
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
        fit_name = replace_name(fit_name)
        for param in params:
            params[param].set(value=fit_data[fit_name][param])
        if x in df:
            x_data = df[x].values
        else:
            x_data = _evaluate_string(x, df).values
        if len(x_data) < 100:
            x_data = np.linspace(x_data.min(), x_data.max(), 100)
        y_data = model.eval(params=params, x=x_data)
        figure.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                name=fit_name,
            ),
            secondary_y=y_axis == "right",
        )
    except Exception as e:
        warnings.warn(
            f"Could not plot the fit {func} for {y} vs {x}.\n"
            f"Please check the fit parameters and the data.\n{e}"
        )


# Create a base namespace with common modules and constants
base_namespace = {"numpy": np, "np": np, "time": 0, "const": const}
# Add all numpy functions to the base namespace
base_namespace.update({name: getattr(np, name) for name in np.__all__})


def _evaluate_string(string, df):
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
    namespace = dict(base_namespace)
    namespace.update(df.to_dict(orient="series"))
    return eval(string, {}, namespace)
