import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from core.config import BoundaryConfig, BoundaryType, GridSpec, Method, PhysicalSpec, SolverSpec, TimeSpec
from core.factory import build_boundary
from core.factory import build_stepper
from core.initial_condition import INITIAL_CONDITIONS_FACTORY
from core.solver import Runner
from core.types import Array1D
from utils.plotting import plot_colored_line
from utils.stability import calculate_stable_timestep
from utils.initial_condition_ui import initial_condition_ui


# Physical constants
L = 0.1  # in m
Nx = 600  # Spatial points
total_time = 5.0  # in seconds
update_frequency = 100  # Animation update frequency

# Streamlit UI setup
st.set_page_config(page_title="1D Heat Diffusion", layout="wide")
st.title("1D Heat Equation Simulator")

# Display physical setup
st.sidebar.markdown("### Physical Setup")
st.sidebar.markdown(f"**Rod length:** {L:.2f} m")
st.sidebar.markdown(f"**Simulation time:** {total_time:.0f} seconds")
st.sidebar.markdown(f"**Spatial resolution:** {Nx} points")

st.sidebar.markdown("---")

# User-controllable parameters
st.sidebar.markdown("### Simulation Parameters")

# Thermal diffusivity with realistic material options (in m²/s)
diffusivity_options = {
    "Aluminium": 8.4e-5,
    "Copper": 1.1e-4,
    "Steel": 1.2e-5,
}

material_choice = st.sidebar.selectbox(
    "Material (thermal diffusivity α)", 
    list(diffusivity_options.keys()),
    index=0
)

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

placeholder = st.empty()
progress = st.progress(0)
run = st.button("Run simulation")

try:
    stepper = build_stepper(spec.method, grid, time_spec, phys, bc)
except NotImplementedError as e:
    st.error(f"{e}")
    stepper = None

figure_width, figure_height = 18, 8

if stepper is None:
    st.sidebar.error("Selected method not implemented yet")

fig = plot_colored_line(
    x, u0, y_min, y_max, 0,
    figsize=(figure_width, figure_height),
    pad=1.0,
    cbar_fraction=0.046,
    cbar_pad_rel=0.04,
    simulation_time=0.0,
)
placeholder.pyplot(fig, clear_figure=True, width='content')
plt.close(fig)

if run and stepper is not None:
    runner = Runner(stepper=stepper, u0=u0)
    total_steps = int(np.floor(time_spec.T / time_spec.dt))
    
    estimated_frames = total_steps // update_frequency + 1
    
    # Show simulation info
    st.info(f"Simulating {time_spec.T:.0f}s of heat diffusion in {material_choice.lower()} rod")
    st.info(f"Time step: {dt*1000:.3f} ms | Steps: {total_steps:,}")
    st.info(f"Will show ~{estimated_frames} frames (every {update_frequency} steps)")
    
    frame_count = 0
    for snap in runner.run():
        progress.progress(min(1.0, snap.step / max(1, total_steps)))
        
        # Update visualization based on user choice
        if snap.step % update_frequency == 0 or snap.step == total_steps:
            fig = plot_colored_line(
                x, snap.u, y_min, y_max, snap.step,
                figsize=(figure_width, figure_height),
                pad=1.0,
                cbar_fraction=0.046,
                cbar_pad_rel=0.04,
                simulation_time=snap.t,
            )
            placeholder.pyplot(fig, clear_figure=True, width='content')
            plt.close(fig)
            frame_count += 1
    
    st.success(f"Simulation completed! Showed {frame_count} frames.")
