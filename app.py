import numpy as np
import time
import streamlit as st

from core.config import BoundaryConfig, BoundaryType, GridSpec, Method, PhysicalSpec, SolverSpec, TimeSpec
from core.factory import build_boundary
from core.factory import build_stepper
from core.initial_condition import INITIAL_CONDITIONS_FACTORY
from core.solver import Runner
from core.types import Array1D
from utils.plotting_plotly import create_heat_plot, update_heat_plot_data
from utils.stability import calculate_stable_timestep
from utils.initial_condition_ui import initial_condition_ui


# Physical constants
L = 0.1  # in m
Nx = 600  # Spatial points
total_time = 3.0  # in seconds
update_frequency = 20  # Animation update frequency for better performance

# Thermal diffusivity with realistic material options (in m²/s)
diffusivity_options = {
    "Aluminium": 8.4e-5,
    "Copper": 1.1e-4,
    "Steel": 1.2e-5,
}

# Streamlit UI setup
st.set_page_config(page_title="1D Heat Diffusion", layout="wide")
st.title("1D Heat Equation Simulator")

st.sidebar.markdown("### Physical Setup")
st.sidebar.markdown(f"**Rod length:** {L:.2f} m")
st.sidebar.markdown(f"**Simulation time:** {total_time:.0f} seconds")
st.sidebar.markdown(f"**Spatial resolution:** {Nx} points")

st.sidebar.markdown("---")

# User-controllable parameters
st.sidebar.markdown("### Simulation Parameters")

material_choice = st.sidebar.selectbox(
    "Material (thermal diffusivity α)", 
    list(diffusivity_options.keys()),
    index=0
)


def setup_simulation():
    alpha = diffusivity_options[material_choice]
    st.sidebar.markdown(f"α = {alpha:.2e} m²/s")

    ic_choice, ic_params = initial_condition_ui()

    # Calculate grid spacing and handle stability
    dx = L / (Nx - 1)
    dt = calculate_stable_timestep(alpha, dx)

    # Build simulation specs
    grid = GridSpec(L=L, Nx=Nx)
    time_spec = TimeSpec(T=total_time, dt=dt)
    phys = PhysicalSpec(alpha=alpha)

    # Initial condition
    x = np.linspace(0.0, grid.L, grid.Nx, dtype=np.float64)
    u0_fn = INITIAL_CONDITIONS_FACTORY[ic_choice](**ic_params)
    u0: Array1D = u0_fn(x)

    # Set boundary conditions to match initial condition endpoints -- only Dirichlet for now
    left_bc_val = float(u0[0])
    right_bc_val = float(u0[-1])

    # Set up simulation
    boundary_config = BoundaryConfig(
        boundary_type=BoundaryType.DIRICHLET,
        left_value=left_bc_val,
        right_value=right_bc_val
    )
    bc = build_boundary(boundary_config)

    method_choice = Method.FTCS_EXPLICIT.value
    spec = SolverSpec(method=Method(method_choice), boundary_type=BoundaryType.DIRICHLET)

    # Calculate plot limits
    y_min = float(min(u0.min(), left_bc_val, right_bc_val)) - 0.1
    y_max = float(max(u0.max(), left_bc_val, right_bc_val)) + 0.1

    stepper = build_stepper(spec.method, grid, time_spec, phys, bc)
    
    return x, u0, y_min, y_max, stepper, time_spec


def reset_simulation():
    """Reset the simulation state and reinitialize everything."""
    # Clear the sidebar to reset UI components
    # st.sidebar.empty()
    # Force rerun to reset all state
    st.rerun()


# Initial setup
x, u0, y_min, y_max, stepper, time_spec = setup_simulation()

placeholder = st.empty()
run = st.button("Run simulation")

fig = create_heat_plot(x, u0, y_min, y_max, 0, simulation_time=0.0, total_simulation_time=total_time)
placeholder.plotly_chart(fig, use_container_width=True)

if run:
    runner = Runner(stepper=stepper, u0=u0)
    total_steps = int(np.floor(time_spec.T / time_spec.dt))
    
    estimated_frames = total_steps // update_frequency + 1
    
    frame_count = 0
    for snap in runner.run():        
        if snap.step % update_frequency != 0 and snap.step != total_steps:
            frame_count += 1
            continue
        
        update_heat_plot_data(fig, snap.u, snap.step, snap.t, total_simulation_time=total_time)
        placeholder.plotly_chart(fig, use_container_width=True, key=f"heat_plot_{snap.step}")
        frame_count += 1
    
    st.success("Simulation complete! Will auto-reset in 5 seconds...")
    time.sleep(5)
    reset_simulation()
