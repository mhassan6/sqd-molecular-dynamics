"""H2 ab initio MD with FCI/STO-3G energies and velocity Verlet.

Same setup as examples/h2_dft.py (Fedorov et al., J. Chem. Phys. 154, 164103
(2021)), but with FCI energies, which is the paper's exact reference curve in Fig. 6.

All quantities are in atomic units (bohr, hartree, electron masses, a.u. time).
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # tiny molecule: threading only adds overhead
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto, scf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qmd import velocity_verlet  # noqa: E402

AU_TIME_FS = 0.02418884326585747          # 1 a.u. of time in fs
AMU_TO_ME = 1822.888486209                # 1 amu in electron masses
M_H = 1.00782503223 * AMU_TO_ME           # hydrogen-atom mass (electron masses)

# --- MD parameters (from the paper) ---------------------------------------
x0 = 2.0              # initial bond length, bohr
v0 = 0.0              # initial bond-length velocity, a.u.
dt = 5.0              # time step, a.u.
fd_step = 1e-4        # finite-difference displacement d, bohr
t_total_fs = 10.5     # one vibrational period, as in Fig. 6
n_steps = int(round(t_total_fs / AU_TIME_FS / dt))
MASS = M_H            # m_H on the bond coordinate (reproduces the paper's ~10.5 fs period)


def fci_energy(x):
    """FCI/STO-3G total energy (hartree) of H2 at bond length x (bohr)."""
    r = float(np.ravel(x)[0])
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis="sto-3g", unit="bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge at r = {r} bohr")
    cis = fci.FCI(mf)
    cis.conv_tol = 1e-12
    e, _ = cis.kernel()
    return e


def main():
    tic = time.perf_counter()
    traj = velocity_verlet(fci_energy, x0, v0, MASS, dt, n_steps, fd_step=fd_step)
    print(f"{n_steps} steps in {time.perf_counter() - tic:.1f} s")

    t_fs = traj.times * AU_TIME_FS
    x, v = traj.positions, traj.velocities
    E = traj.total_energy
    dE_mHa = (E - E[0]) * 1e3
    i_min = np.argmin(x)
    print(f"bond length range: {x.min():.4f} - {x.max():.4f} bohr (min at t = {t_fs[i_min]:.2f} fs)")
    print(f"max |v|: {np.abs(v).max():.3e} a.u.")
    print(f"max E - E0: {dE_mHa.max():+.4f} mHa, min: {dE_mHa.min():+.4f} mHa")
    print(f"final x: {x[-1]:.4f} bohr (drift {x[-1] - x0:+.4f})")

    np.savez(ROOT / "h2_fci_trajectory.npz", times_au=traj.times, positions=x,
             velocities=v, forces=traj.forces, potential=traj.potential_energies,
             kinetic=traj.kinetic_energies, mass=MASS, dt=dt)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    ax.plot(x, v, color="tab:blue", lw=2, label="FCI/STO-3G")
    ax.plot(x[0], v[0], "o", color="tab:red", label="Start")
    ax.set_xlabel("Position (bond length), a.u.")
    ax.set_ylabel("Velocity, a.u.")
    ax.set_title(r"Phase space: H$_2$")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t_fs, dE_mHa, color="tab:blue")
    ax.set_xlabel("Time, fs")
    ax.set_ylabel(r"$E_{tot}(t) - E_{tot}(0)$, mHa")
    ax.set_title("Total-energy conservation")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t_fs, traj.potential_energies, color="tab:blue")
    ax.set_xlabel("Time, fs")
    ax.set_ylabel("Potential energy, Ha")
    ax.set_title("Potential energy")
    ax.grid(alpha=0.3)

    fig.suptitle(rf"H$_2$ AIMD, FCI/STO-3G, $x_0$={x0} a$_0$, $\Delta t$={dt:g} a.u.")
    fig.tight_layout()
    out = ROOT / "h2_fci_phase_space.png"
    fig.savefig(out, dpi=200)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
