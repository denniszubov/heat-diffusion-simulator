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


# Streamlit UI setup
st.set_page_config(page_title="1D Heat Diffusion", layout="wide")
st.title("1D Heat Equation Simulator (Skeleton)")

# Sidebar inputs
L = st.sidebar.number_input("Rod length L", value=1.0, min_value=0.1, step=0.1, format="%.2f")
Nx = st.sidebar.slider("Spatial points Nx", min_value=50, max_value=800, value=200, step=10)
alpha = st.sidebar.number_input("Thermal diffusivity α", value=1.0, min_value=0.0001, step=0.1, format="%.4f")
total_time = st.sidebar.number_input("Total simulated time", value=0.2, min_value=0.01, step=0.05, format="%.3f")
dt = st.sidebar.number_input("Time step Δt", value=1e-5, min_value=1e-6, step=1e-5, format="%.6f")

method_choice = st.sidebar.selectbox("Numerical method", [m.value for m in Method], index=0)
boundary_type_choice = st.sidebar.selectbox("Boundary condition type", [b.value for b in BoundaryType], index=0)
ic_choice = st.sidebar.selectbox("Initial condition", list(INITIAL_CONDITIONS_FACTORY.keys()), index=0)

left_bc_val = st.sidebar.number_input("Left BC value", value=0.0, step=0.1, format="%.2f")
right_bc_val = st.sidebar.number_input("Right BC value", value=0.0, step=0.1, format="%.2f")

# Build simulation specs
grid = GridSpec(L=float(L), Nx=int(Nx))
time_spec = TimeSpec(T=float(total_time), dt=float(dt))
phys = PhysicalSpec(alpha=float(alpha))

# Initial condition
x = np.linspace(0.0, grid.L, grid.Nx, dtype=np.float64)
u0_fn = INITIAL_CONDITIONS_FACTORY[ic_choice]()
u0: Array1D = u0_fn(x)

# Set up simulation
boundary_config = BoundaryConfig(
    boundary_type=BoundaryType(boundary_type_choice),
    left_value=left_bc_val,
    right_value=right_bc_val
)
bc = build_boundary(boundary_config)
spec = SolverSpec(method=Method(method_choice), boundary_type=BoundaryType(boundary_type_choice))

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

fig = plot_colored_line(
    x, u0, y_min, y_max, 0,
    figsize=(10, 4),
    pad=1.0,
    cbar_fraction=0.046,
    cbar_pad_rel=0.04,
)
placeholder.pyplot(fig, clear_figure=True, use_container_width=False)

if run and stepper is not None:
    runner = Runner(stepper=stepper, u0=u0)
    total_steps = int(np.floor(time_spec.T / time_spec.dt))
    for snap in runner.run():
        progress.progress(min(1.0, snap.step / max(1, total_steps)))
        fig = plot_colored_line(
            x, snap.u, y_min, y_max, snap.step,
            figsize=(10, 4),
            pad=1.0,
            cbar_fraction=0.046,
            cbar_pad_rel=0.04,
        )
        placeholder.pyplot(fig, clear_figure=True, use_container_width=False)
        plt.close(fig)
        time.sleep(0.0001)

    st.success("Done. Implement your stepper to see dynamics.")
