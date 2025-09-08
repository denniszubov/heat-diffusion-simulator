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
from utils.stability import display_stability_education
from utils.stability import stability_control_ui


# Physical constants
L = 0.1  # in m
Nx = 201  # Spatial points
total_time = 60.0  # in seconds
left_bc_val = 0.0  # in °C
right_bc_val = 0.0  # in °C
update_frequency = 100  # Animation update frequency

# Streamlit UI setup
st.set_page_config(page_title="1D Heat Diffusion", layout="wide")
st.title("1D Heat Equation Simulator")

# Display physical setup
st.sidebar.markdown("### Physical Setup")
st.sidebar.markdown(f"**Rod length:** {L:.2f} m")
st.sidebar.markdown(f"**Simulation time:** {total_time:.0f} seconds")
st.sidebar.markdown(f"**Spatial resolution:** {Nx} points")
st.sidebar.markdown(f"**Boundary conditions:** {left_bc_val}°C (both ends)")

st.sidebar.markdown("---")

# User-controllable parameters
st.sidebar.markdown("### Simulation Parameters")

# Thermal diffusivity with realistic material options (in m²/s)
diffusivity_options = {
    "Aluminum": 8.4e-5,
    "Copper": 1.1e-4,
    "Steel": 1.2e-5,
    "Glass": 3.4e-7,
    "Custom": None
}

material_choice = st.sidebar.selectbox(
    "Material (thermal diffusivity α)", 
    list(diffusivity_options.keys()),
    index=0
)

if material_choice == "Custom":
    alpha = st.sidebar.number_input(
        "Custom α (m²/s)", 
        value=8.4e-5, 
        min_value=1e-8, 
        max_value=1e-3, 
        step=1e-6, 
        format="%.2e"
    )
else:
    alpha = diffusivity_options[material_choice]
    st.sidebar.markdown(f"α = {alpha:.2e} m²/s")

method_choice = st.sidebar.selectbox(
    "Numerical method", 
    [m.value for m in Method], 
    index=0
)

ic_choice = st.sidebar.selectbox(
    "Initial condition", 
    list(INITIAL_CONDITIONS_FACTORY.keys()), 
    index=0
)

# Calculate grid spacing and handle stability
dx = L / (Nx - 1)
dt, r = stability_control_ui(alpha, dx)

# Build simulation specs
grid = GridSpec(L=L, Nx=Nx)
time_spec = TimeSpec(T=total_time, dt=dt)
phys = PhysicalSpec(alpha=alpha)

# Initial condition
x = np.linspace(0.0, grid.L, grid.Nx, dtype=np.float64)
u0_fn = INITIAL_CONDITIONS_FACTORY[ic_choice]()
u0: Array1D = u0_fn(x)

# Set up simulation
boundary_config = BoundaryConfig(
    boundary_type=BoundaryType.DIRICHLET,
    left_value=left_bc_val,
    right_value=right_bc_val
)
bc = build_boundary(boundary_config)
spec = SolverSpec(method=Method(method_choice), boundary_type=BoundaryType.DIRICHLET)

# Calculate plot limits
y_min = float(min(u0.min(), left_bc_val, right_bc_val)) - 0.1
y_max = float(max(u0.max(), left_bc_val, right_bc_val)) + 0.1

left_col, mid_col, right_col = st.columns([1, 3, 1])
with mid_col:
    placeholder = st.empty()
progress = st.progress(0)
run = st.button("Run simulation")

try:
    stepper = build_stepper(spec.method, grid, time_spec, phys, bc)
except NotImplementedError as e:
    st.error(f"{e}")
    stepper = None

if stepper is not None:
    note = stepper.stability_note()
    if note:
        st.sidebar.info(note)

    display_stability_education()
else:
    st.sidebar.error("Selected method not implemented yet")

fig = plot_colored_line(
    x, u0, y_min, y_max, 0,
    figsize=(10, 4),
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
                figsize=(10, 4),
                pad=1.0,
                cbar_fraction=0.046,
                cbar_pad_rel=0.04,
                simulation_time=snap.t,
            )
            placeholder.pyplot(fig, clear_figure=True, width='content')
            plt.close(fig)
            frame_count += 1
    
    st.success(f"Simulation completed! Showed {frame_count} frames.")
