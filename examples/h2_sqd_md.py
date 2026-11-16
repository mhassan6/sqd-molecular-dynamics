"""H2 ab initio MD with SQD energies (LUCJ + StatevectorSampler + qiskit-addon-sqd).

Brute-force finite-difference forces as in Fedorov et al., J. Chem. Phys. 154,
164103 (2021): F = -[E(x+d) - E(x-d)] / (2d), each E from an independent SQD run.
Same MD setup as examples/h2_fci.py, whose trajectory is overlaid for reference.

Outputs: h2_sqd_phase_space.png, h2_sqd_trajectory.npz, h2_sqd_states.txt
(SQD subspace and ground state at each MD step's geometry x(t)).
All quantities in atomic units.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qmd import velocity_verlet  # noqa: E402
from qmd.sqd import BIT_CONVENTION, SQDEnergy, format_record  # noqa: E402

AU_TIME_FS = 0.02418884326585747
AMU_TO_ME = 1822.888486209
M_H = 1.00782503223 * AMU_TO_ME

x0, v0 = 2.0, 0.0
dt = 5.0
fd_step = 1e-4
t_total_fs = 10.5
n_steps = int(round(t_total_fs / AU_TIME_FS / dt))
MASS = M_H

SHOTS = 10_000
N_REPS = 1
SEED = 12345


def main():
    solver = SQDEnergy(shots=SHOTS, n_reps=N_REPS, seed=SEED)
    states_path = ROOT / "h2_sqd_states.txt"
    fh = open(states_path, "w")
    fh.write(
        "# H2/STO-3G ab initio MD with SQD energies: subspace and ground state at each MD step\n"
        f"# LUCJ n_reps={N_REPS} (CCSD-seeded), StatevectorSampler shots={SHOTS}, seed={SEED};\n"
        f"# x0={x0} bohr, v0={v0}, dt={dt} a.u., fd_step={fd_step} bohr, mass={MASS:.4f} m_e\n"
        + BIT_CONVENTION + "\n"
    )

    def callback(step, x, v, F):
        # velocity_verlet evaluates energy_fn(x) right before this callback,
        # so solver.last is the SQD result at the center geometry x(t).
        rec = solver.last
        assert abs(rec.r - float(np.ravel(x)[0])) < 1e-14
        fh.write(format_record(
            rec, f"# step {step}   t = {step * dt * AU_TIME_FS:.4f} fs   r = {rec.r:.8f} bohr   "
                 f"v = {float(np.ravel(v)[0]):+.6e}   F = {float(np.ravel(F)[0]):+.6e}"))

    tic = time.perf_counter()
    traj = velocity_verlet(solver, x0, v0, MASS, dt, n_steps, fd_step=fd_step, callback=callback)
    fh.close()
    print(f"{n_steps} steps, {solver.n_calls} SQD runs in {time.perf_counter() - tic:.1f} s")
    dims = np.array(solver.dims)
    print(f"subspace dimension over all SQD runs: min {dims.min()}, max {dims.max()}"
          f" ({np.sum(dims < 4)} runs below the full 4)")

    t_fs = traj.times * AU_TIME_FS
    x, v = traj.positions, traj.velocities
    dE_mHa = (traj.total_energy - traj.total_energy[0]) * 1e3
    i_min = np.argmin(x)
    print(f"bond length range: {x.min():.4f} - {x.max():.4f} bohr (min at t = {t_fs[i_min]:.2f} fs)")
    print(f"max E - E0: {dE_mHa.max():+.4f} mHa, min: {dE_mHa.min():+.4f} mHa")
    print(f"final x: {x[-1]:.4f} bohr (drift {x[-1] - x0:+.4f})")

    ref_path = ROOT / "h2_fci_trajectory.npz"
    ref = np.load(ref_path) if ref_path.exists() else None
    if ref is not None and len(ref["positions"]) == len(x):
        print(f"max |x_SQD - x_FCI|: {np.abs(x - ref['positions']).max():.2e} bohr, "
              f"max |v_SQD - v_FCI|: {np.abs(v - ref['velocities']).max():.2e} a.u.")
    else:
        ref = None

    np.savez(ROOT / "h2_sqd_trajectory.npz", times_au=traj.times, positions=x,
             velocities=v, forces=traj.forces, potential=traj.potential_energies,
             kinetic=traj.kinetic_energies, mass=MASS, dt=dt, shots=SHOTS, n_reps=N_REPS)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ref_kw = dict(color="0.65", lw=5, label="FCI/STO-3G")
    ax = axes[0]
    if ref is not None:
        ax.plot(ref["positions"], ref["velocities"], **ref_kw)
    ax.plot(x, v, color="tab:green", lw=1.8, label="SQD (LUCJ, statevector)")
    ax.plot(x[0], v[0], "o", color="tab:red", label="Start")
    ax.set_xlabel("Position (bond length), a.u.")
    ax.set_ylabel("Velocity, a.u.")
    ax.set_title(r"Phase space: H$_2$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if ref is not None:
        E_ref = ref["kinetic"] + ref["potential"]
        ax.plot(t_fs, (E_ref - E_ref[0]) * 1e3, **ref_kw)
    ax.plot(t_fs, dE_mHa, color="tab:green", lw=1.8, label="SQD")
    ax.set_xlabel("Time, fs")
    ax.set_ylabel(r"$E_{tot}(t) - E_{tot}(0)$, mHa")
    ax.set_title("Total-energy conservation")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    if ref is not None:
        ax.plot(t_fs, ref["potential"], **ref_kw)
    ax.plot(t_fs, traj.potential_energies, color="tab:green", lw=1.8, label="SQD")
    ax.set_xlabel("Time, fs")
    ax.set_ylabel("Potential energy, Ha")
    ax.set_title("Potential energy")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(rf"H$_2$ AIMD, SQD/STO-3G (LUCJ n_reps={N_REPS}, {SHOTS} shots), "
                 rf"$x_0$={x0} a$_0$, $\Delta t$={dt:g} a.u.")
    fig.tight_layout()
    out = ROOT / "h2_sqd_phase_space.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}\nsaved {states_path}")


if __name__ == "__main__":
    main()
