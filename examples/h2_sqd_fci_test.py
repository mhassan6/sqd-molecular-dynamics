"""Test: SQD (LUCJ + StatevectorSampler + qiskit-addon-sqd) vs FCI for H2/STO-3G.

At several bond lengths, check that the SQD ground-state energy and vector match
PySCF FCI. For H2 in STO-3G the full space is 4 determinants, so SQD should be
exact once both alpha and both beta strings appear in the samples.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qmd.sqd import BIT_CONVENTION, SQDEnergy, fci_on_sqd_basis, format_record  # noqa: E402

E_TOL = 1e-8
OVERLAP_TOL = 1e-8
BOND_LENGTHS = [1.0, 1.4, 2.0, 3.0]


def main():
    solver = SQDEnergy(shots=10_000, n_reps=1, seed=12345)
    out = ROOT / "h2_sqd_fci_test.txt"
    failures = []
    print(f"{'r (bohr)':>8}  {'E_SQD':>15}  {'E_FCI':>15}  {'dE':>9}  {'|<SQD|FCI>|':>13}  dim")
    with open(out, "w") as fh:
        fh.write("# H2/STO-3G: SQD ground states vs FCI\n" + BIT_CONVENTION + "\n")
        for r in BOND_LENGTHS:
            rec = solver.run(r)
            dE = rec.e_sqd - rec.e_fci
            overlap = abs(np.vdot(fci_on_sqd_basis(rec), rec.amplitudes))
            print(f"{r:8.3f}  {rec.e_sqd:15.10f}  {rec.e_fci:15.10f}  {dE:+9.1e}  {overlap:13.10f}  {rec.dim}")
            fh.write(format_record(rec, f"# r = {r:.4f} bohr"))
            if abs(dE) > E_TOL or overlap < 1 - OVERLAP_TOL:
                failures.append(r)
    print(f"wrote {out}")
    assert not failures, f"SQD does not match FCI at r = {failures}"
    print("PASS: SQD matches FCI (energy and ground-state vector) at all bond lengths")


if __name__ == "__main__":
    main()
