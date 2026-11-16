"""Test case: 1D harmonic oscillator integrated with qmd.velocity_verlet.

V(x) = k x^2 / 2 with k = 4 pi^2 and m = 1, so omega = 2 pi and the period is 1.
Forces are obtained from the energy by finite differences, exercising the same
code path that will later be used with DFT / SQD energies.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qmd import velocity_verlet  # noqa: E402

k = 4 * np.pi**2
m = 1.0
omega = np.sqrt(k / m)
x0, v0 = 1.6, 0.0
dt = 0.0005
n_periods = 2
n_steps = int(round(n_periods * 2 * np.pi / omega / dt))


def harmonic_energy(x):
    return float(np.sum(0.5 * k * np.asarray(x) ** 2))


traj = velocity_verlet(harmonic_energy, x0, v0, m, dt, n_steps)

t = traj.times
x = traj.positions
v = traj.velocities
x_exact = x0 * np.cos(omega * t)
v_exact = -x0 * omega * np.sin(omega * t)

E = traj.total_energy
rel_dE = (E - E[0]) / E[0]
print(f"steps: {n_steps}, dt: {dt}")
print(f"max |relative energy error|: {np.max(np.abs(rel_dE)):.3e}")
print(f"max |x - x_exact|:           {np.max(np.abs(x - x_exact)):.3e}")
print(f"max |v - v_exact|:           {np.max(np.abs(v - v_exact)):.3e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

ax1.plot(x_exact, v_exact, color="0.6", lw=4, label="Analytic")
ax1.plot(x, v, color="tab:blue", lw=1.5, label="Velocity Verlet")
ax1.plot(x[0], v[0], "o", color="tab:red", label="Start")
ax1.set_xlabel("Position x")
ax1.set_ylabel("Velocity v")
ax1.set_title("Phase space: harmonic oscillator")
ax1.set_aspect("auto")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(t, rel_dE, color="tab:blue")
ax2.set_xlabel("Time")
ax2.set_ylabel(r"$(E(t) - E_0)/E_0$")
ax2.set_title("Relative total-energy error")
ax2.grid(alpha=0.3)

fig.tight_layout()
out = ROOT / "harmonic_oscillator_phase_space.png"
fig.savefig(out, dpi=200)
print(f"saved {out}")
