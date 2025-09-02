from typing import Callable

import numpy as np

from .types import Array1D
from .types import InitialCondition


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


INITIAL_CONDITIONS_FACTORY: dict[str, Callable[..., InitialCondition]] = {
    "Gaussian Bump": initial_condition_gaussian,
}
