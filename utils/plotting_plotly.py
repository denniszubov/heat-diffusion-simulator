import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


def create_heat_plot(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    y_min: float,
    y_max: float,
    simulation_time: float,
    total_simulation_time: float,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                color=y,  # color by temperature
                colorscale=[[0, "blue"], [1, "red"]],  # blue->red
                cmin=y.min(),
                cmax=y.max(),
                size=6,
                opacity=0.9,
            ),
            name="Temperature",
            showlegend=False,
        )
    )

    title = f"Heat Diffusion - Time: {simulation_time:.1f}s / {total_simulation_time}s"

    fig.update_layout(
        title={"text": title, "font": {"size": 24}, "x": 0.5},
        xaxis={
            "title": {"text": "Position (m)", "font": {"size": 18}},
            "tickfont": {"size": 14},
            "range": [float(x.min()), float(x.max())],
        },
        yaxis={
            "title": {"text": "Temperature (°C)", "font": {"size": 18}},
            "tickfont": {"size": 14},
            "range": [y_min, y_max],
        },
        width=1200,
        height=600,
        margin={"l": 80, "r": 80, "t": 80, "b": 80},
    )

    return fig


def update_heat_plot_data(
    fig: go.Figure,
    y: NDArray[np.float64],
    simulation_time: float,
    total_simulation_time: float,
) -> None:
    fig.data[0].y = y
    fig.data[0].marker.color = y

    title = f"Heat Diffusion - Time: {simulation_time:.1f}s / {total_simulation_time}s"
    fig.update_layout(title=dict(text=title))
