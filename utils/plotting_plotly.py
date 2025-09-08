import plotly.graph_objects as go
import numpy as np
from numpy.typing import NDArray


def create_heat_plot(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    y_min: float,
    y_max: float,
    step: int,
    simulation_time: float = None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="red", width=3),
            name="Temperature",
            showlegend=False,
        )
    )

    title = (
        f"Heat Diffusion - Time: {simulation_time:.1f}s (Step {step})"
        if simulation_time is not None
        else f"Temperature profile at step {step}"
    )

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
    step: int,
    simulation_time: float = None,
) -> None:
    """Update only the data and title of existing Plotly figure."""

    # Update temperature data
    fig.data[0].y = y

    title = (
        f"Heat Diffusion - Time: {simulation_time:.1f}s (Step {step})"
        if simulation_time is not None
        else f"Temperature profile at step {step}"
    )
    fig.update_layout(title=dict(text=title))
