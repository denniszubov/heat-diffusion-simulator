# 1D Heat Diffusion Simulator

A real-time interactive simulator for the 1D heat equation using finite difference methods. Watch heat flow through different materials and explore how boundary conditions affect thermal behavior.

## Overview

This application simulates heat diffusion through a 1D rod, demonstrating fundamental principles of thermal physics. Heat naturally flows from hot to cold regions, and this simulator lets you visualize that process in real-time with different materials and conditions.

**Key Physics Concepts:**
- **Thermal Diffusivity**: How quickly heat spreads through a material
- **Boundary Effects**: How rod endpoints influence heat flow (fixed temperatures vs. insulated boundaries)
- **Material Properties**: Real thermal properties of aluminum, copper, and steel
- **Temperature Evolution**: Watch initial heat distributions smooth out over time

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Navigate to the local URL to start experimenting with heat diffusion.

## What You'll Observe

- **Gaussian Bumps**: Hot spots that spread and cool down
- **Step Functions**: Sharp temperature jumps that smooth out
- **Material Differences**: Copper conducts heat faster than steel
- **Boundary Effects**: Fixed boundaries act as heat reservoirs; isolated boundaries prevent heat loss

---

## Mathematical Foundation

The simulator solves the 1D heat diffusion equation:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Where $u(x,t)$ is temperature and $\alpha$ is thermal diffusivity. This equation captures Fourier's law of heat conduction: heat flows proportional to temperature gradients.

**Numerical Method**: Forward-Time Central-Space (FTCS) finite differences with automatic stability control ($r = \alpha \frac{\Delta t}{\Delta x^2} \leq 0.5$).

**Boundary Conditions**:
- *Dirichlet*: $u(0,t) = u_L$, $u(L,t) = u_R$ (thermostats at endpoints)
- *Neumann*: $\frac{\partial u}{\partial x}\big|_{x=0,L} = 0$ (perfectly insulated ends)
