import streamlit as st
from typing import Tuple


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


def calculate_stability_parameter(alpha: float, dt: float, dx: float) -> float:
    """
    Calculate the stability parameter r = α·dt/dx².
    
    Args:
        alpha: Thermal diffusivity (m²/s)
        dt: Time step (s)
        dx: Spatial grid spacing (m)
    
    Returns:
        Stability parameter r
    """
    return alpha * dt / dx**2


def stability_control_ui(alpha: float, dx: float) -> Tuple[float, float]:
    """
    Create stability control UI in Streamlit sidebar.
    
    Args:
        alpha: Thermal diffusivity (m²/s)
        dx: Spatial grid spacing (m)
    
    Returns:
        Tuple of (dt, r)
    """
    # Auto-calculate stable time step
    dt = calculate_stable_timestep(alpha, dx)
    dt_ms = dt * 1000
    r = 0.5  # Always stable by design

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stability Control")

    # Show stability information
    st.sidebar.markdown(f"**Time step:** {dt_ms:.3f} ms")
    st.sidebar.markdown(f"**Grid spacing:** {dx*1000:.2f} mm")
    st.sidebar.info(f"Stable: r = {r:.1f}")

    return dt, r
