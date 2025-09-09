from typing import Callable

import numpy as np

from .types import Array1D, InitialCondition


def initial_condition_gaussian(center_fraction: float = 0.5, width_fraction: float = 0.1) -> InitialCondition:
    """
    Returns a Gaussian initial condition function for 1D heat diffusion.
    center_fraction: Fractional position of the mean (center) relative to the domain length.
    width_fraction: Fractional standard deviation (width) relative to the domain length.
    """

    def _f(x: Array1D) -> Array1D:
        domain_length = float(x[-1] - x[0])
        center = center_fraction * domain_length + x[0]
        width = width_fraction * domain_length
        return np.exp(-0.5 * ((x - center) / width) ** 2).astype(np.float64)

    return _f


def initial_condition_step() -> InitialCondition:
    """
    Step function: hot on the left half, cold on the right half.
    Classic discontinuous initial condition.
    """

    def _f(x: Array1D) -> Array1D:
        domain_length = float(x[-1] - x[0])
        midpoint = x[0] + 0.5 * domain_length
        return np.where(x < midpoint, 100.0, 0.0).astype(np.float64)

    return _f


def initial_condition_sine_wave(frequency: float = 2.0, amplitude: float = 50.0) -> InitialCondition:
    """
    Sinusoidal initial condition.
    frequency: Number of complete waves across the domain
    amplitude: Peak temperature above baseline
    """

    def _f(x: Array1D) -> Array1D:
        domain_length = float(x[-1] - x[0])
        x_normalized = (x - x[0]) / domain_length
        baseline = amplitude
        return (baseline + amplitude * np.sin(frequency * 2 * np.pi * x_normalized)).astype(np.float64)

    return _f


def initial_conditions_square_wave(steps: int = 5, amplitude: float = 50.0) -> InitialCondition:
    """
    Square wave initial condition.
    steps: Number of square wave steps across the domain
    amplitude: Peak temperature above baseline
    """

    def _f(x: Array1D) -> Array1D:
        domain_length = float(x[-1] - x[0])
        x_normalized = (x - x[0]) / domain_length

        wave = np.sign(np.sin(steps * 2 * np.pi * x_normalized))

        # Force exact zeros at boundaries (since sin(0)=0, sin(2πk)=0)
        u0 = amplitude * wave
        u0[0] = 0.0
        u0[-1] = 0.0

        return u0.astype(np.float64)

    return _f


INITIAL_CONDITIONS_FACTORY: dict[str, Callable[..., InitialCondition]] = {
    "Gaussian Bump": initial_condition_gaussian,
    "Step Function": initial_condition_step,
    "Sine Wave": initial_condition_sine_wave,
    "Square Wave": initial_conditions_square_wave,
}
