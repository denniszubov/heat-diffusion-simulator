def calculate_stable_timestep(alpha: float, dx: float, safety_factor: float = 0.5) -> float:
    """
    Calculate the maximum stable time step for explicit finite difference methods.

    Args:
        alpha: Thermal diffusivity (m²/s)
        dx: Spatial grid spacing (m)
        safety_factor: Safety factor for stability (default 0.5 for FTCS)

    Returns:
        Maximum stable time step (s)
    """
    return safety_factor * dx**2 / alpha
