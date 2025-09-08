import numpy as np

from ..solver import Stepper
from ..types import Array1D


class FTCS(Stepper):
    """
    Forward-Time Central-Space (FTCS) explicit method for 1D heat diffusion.

    Discretizes the heat equation: ∂u/∂t = α ∂²u/∂x²
    Using:
    - Forward difference in time: (u[i]^(n+1) - u[i]^n) / dt
    - Central difference in space: (u[i+1]^n - 2u[i]^n + u[i-1]^n) / dx²

    Update formula: u[i]^(n+1) = u[i]^n + r * (u[i+1]^n - 2u[i]^n + u[i-1]^n)
    Where r = α * dt / dx² (mesh ratio)

    Stability condition: r ≤ 0.5 for 1D case
    """

    def step(self, u: Array1D, t: float) -> Array1D:
        """
        Perform one time step using FTCS scheme.
        
        Args:
            u: Current temperature profile at time t
            t: Current time

        Returns:
            Temperature profile at next time step (t + dt)
        """
        r = self.phys.alpha * self.time.dt / (self.dx * self.dx)
        
        u_next = u.copy()
        
        # Update interior points using FTCS scheme
        u_next[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        
        # Apply boundary conditions to the new solution at the new time t + dt
        new_t = t + self.time.dt
        self.bc.apply(u_next, new_t, self.dx)
        
        return u_next
