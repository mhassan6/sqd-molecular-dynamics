"""Time integrators for (ab initio) molecular dynamics.

The integrators here are agnostic to how the potential energy is computed.
The only coupling to the electronic-structure method is a callable
``energy_fn(x) -> float`` returning the (ground-state) energy at nuclear
configuration ``x``. That callable can wrap DFT, coupled cluster, VQE, SQD,
an analytic model potential, etc.

Units are whatever the caller uses consistently (e.g. atomic units for
molecules: bohr, hartree, electron masses, atomic time units).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

EnergyFn = Callable[[np.ndarray], float]
ForceFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class MDTrajectory:
    """Result of an MD run. Arrays have leading dimension ``n_steps + 1``."""

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    kinetic_energies: np.ndarray
    potential_energies: Optional[np.ndarray] = None

    @property
    def total_energy(self) -> np.ndarray:
        if self.potential_energies is None:
            raise ValueError("Potential energies were not recorded (record_energy=False).")
        return self.kinetic_energies + self.potential_energies


def finite_difference_force(energy_fn: EnergyFn, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Force F = -dE/dx by central finite differences.

    F_i = -[E(x + h e_i) - E(x - h e_i)] / (2h)

    Costs 2 energy evaluations per degree of freedom.
    """
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    force = np.empty_like(flat)
    for i in range(flat.size):
        xp = flat.copy()
        xm = flat.copy()
        xp[i] += h
        xm[i] -= h
        force[i] = -(energy_fn(xp.reshape(x.shape)) - energy_fn(xm.reshape(x.shape))) / (2.0 * h)
    return force.reshape(x.shape)


def velocity_verlet(
    energy_fn: EnergyFn,
    x0,
    v0,
    mass,
    dt: float,
    n_steps: int,
    force_fn: Optional[ForceFn] = None,
    fd_step: float = 1e-5,
    record_energy: bool = True,
    callback: Optional[Callable[[int, np.ndarray, np.ndarray, np.ndarray], None]] = None,
) -> MDTrajectory:
    """Integrate Newton's equations with the velocity Verlet algorithm.

    Each step:
        x(t+dt) = x(t) + v(t) dt + F(t) / (2m) dt^2
        F(t+dt) = F(x(t+dt))
        v(t+dt) = v(t) + [F(t) + F(t+dt)] / (2m) dt

    Parameters
    ----------
    energy_fn : callable
        ``energy_fn(x) -> float``; potential (ground-state) energy at
        configuration ``x`` (same shape as ``x0``). Any method may be used.
    x0, v0 : scalar or array_like
        Initial positions and velocities (same shape).
    mass : scalar or array_like
        Mass(es), broadcastable to the shape of ``x0``.
    dt : float
        Time step.
    n_steps : int
        Number of integration steps.
    force_fn : callable, optional
        ``force_fn(x) -> array`` returning -dE/dx. If omitted, forces are
        obtained from ``energy_fn`` by central finite differences.
    fd_step : float
        Displacement for finite-difference forces.
    record_energy : bool
        If True, evaluate and store ``energy_fn(x)`` at every step (one extra
        energy call per step). Set False to save cost for expensive methods.
    callback : callable, optional
        Called as ``callback(step, x, v, F)`` after every step (and at step 0).

    Returns
    -------
    MDTrajectory
    """
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    if x.shape != v.shape:
        raise ValueError(f"x0 shape {x.shape} and v0 shape {v.shape} differ")
    m = np.broadcast_to(np.asarray(mass, dtype=float), x.shape)

    if force_fn is None:
        def compute_force(pos):
            return finite_difference_force(energy_fn, pos, fd_step)
    else:
        def compute_force(pos):
            return np.asarray(force_fn(pos), dtype=float).reshape(pos.shape)

    shape = (n_steps + 1,) + x.shape
    positions = np.empty(shape)
    velocities = np.empty(shape)
    forces = np.empty(shape)
    kinetic = np.empty(n_steps + 1)
    potential = np.empty(n_steps + 1) if record_energy else None

    F = compute_force(x)

    def store(i):
        positions[i] = x
        velocities[i] = v
        forces[i] = F
        kinetic[i] = 0.5 * np.sum(m * v**2)
        if record_energy:
            potential[i] = energy_fn(x)
        if callback is not None:
            callback(i, x.copy(), v.copy(), F.copy())

    store(0)
    for i in range(1, n_steps + 1):
        x = x + v * dt + F / (2.0 * m) * dt**2
        F_new = compute_force(x)
        v = v + (F + F_new) / (2.0 * m) * dt
        F = F_new
        store(i)

    times = dt * np.arange(n_steps + 1)
    return MDTrajectory(times, positions, velocities, forces, kinetic, potential)
