"""Overlay the H2 phase-space trajectories (FCI, B3LYP, SQD) from saved .npz files.

No simulations are run; reads h2_fci_trajectory.npz, h2_dft_trajectory.npz and
h2_sqd_trajectory.npz. Axes follow Fedorov et al., JCP 154, 164103 (2021), Fig. 6a
(velocity ticks every 2.5e-3 up to |7.5e-3|, position ticks 1.0-2.0).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

RUNS = [  # file, label, plot style
    ("h2_fci_trajectory.npz", "FCI/STO-3G", dict(color="0.6", lw=5, zorder=1)),
    ("h2_dft_trajectory.npz", "B3LYP/STO-3G", dict(color="tab:orange", lw=1.8, zorder=2)),
    ("h2_sqd_trajectory.npz", "SQD/STO-3G (LUCJ, statevector)",
     dict(color="tab:green", lw=1.8, ls="--", zorder=3)),
]

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for fname, label, style in RUNS:
    d = np.load(ROOT / fname)
    ax.plot(d["positions"], d["velocities"], label=label, **style)

ax.plot(2.0, 0.0, "o", color="tab:red", zorder=4, label="Start")
ax.set_xlim(0.98, 2.04)
ax.set_xticks(np.arange(1.0, 2.01, 0.2))
ax.set_ylim(-8.0e-3, 8.0e-3)
ax.set_yticks(np.arange(-7.5e-3, 7.51e-3, 2.5e-3))
ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
ax.set_xlabel("Position, a.u.")
ax.set_ylabel("Velocity, a.u.")
ax.set_title(r"H$_2$ phase space, $x_0$=2.0 a$_0$, $\Delta t$=5 a.u.")
ax.legend(loc="center")
ax.grid(alpha=0.3)
fig.tight_layout()
out = ROOT / "h2_phase_space_comparison.png"
fig.savefig(out, dpi=200)
print(f"saved {out}")
