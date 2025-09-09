import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from numpy.typing import NDArray


def plot_colored_line(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    y_min: float,
    y_max: float,
    step: int,
    figsize: tuple[float, float],
    pad: float,
    cbar_fraction: float,
    cbar_pad_rel: float,
    simulation_time: float = None,
):
    title_fontsize = 32
    label_fontsize = 26
    tick_fontsize = 22

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(y_min, y_max)
    lc = LineCollection(segments, cmap="plasma", norm=norm)
    lc.set_array(y)
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(lc)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Position (m)", fontsize=label_fontsize)
    ax.set_ylabel("Temperature (°C)", fontsize=label_fontsize)

    # Show simulation time if provided, otherwise just step
    if simulation_time is not None:
        ax.set_title(f"Heat Diffusion - Time: {simulation_time:.1f}s (Step {step})", fontsize=title_fontsize)
    else:
        ax.set_title(f"Temperature profile at step {step}", fontsize=title_fontsize)

    # Increase tick label font sizes
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    cbar = fig.colorbar(lc, ax=ax, fraction=cbar_fraction, pad=cbar_pad_rel)
    cbar.set_label("Temperature (°C)", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    fig.tight_layout(pad=pad)
    return fig
