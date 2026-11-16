"""Sample-based quantum diagonalization (SQD) energies for H2 via qiskit-addon-sqd.

Pipeline at each bond length r (follows the SQD notebooks in this directory):
  1. PySCF RHF -> CASCI integrals (all orbitals active), CCSD t1/t2, FCI reference.
  2. LUCJ ansatz (ffsim) seeded from the CCSD amplitudes, on the Hartree-Fock state.
  3. Sample with Qiskit's StatevectorSampler (exact statevector, finite shots).
  4. Plain SQD: project the Hamiltonian onto the sampled determinants and
     diagonalize (qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian,
     one batch, one iteration, no trimming or other variants).

``SQDEnergy`` is a callable ``energy_fn(x) -> float`` usable by
``qmd.velocity_verlet``. Units: bohr, hartree.

Bit conventions: a CI string is an integer whose bit i is set when spatial
orbital i is occupied (same as PySCF). Printed strings have orbital 0 as the
rightmost character. Qiskit measurement bitstrings read beta|alpha
(alpha = qubits 0..norb-1 = rightmost norb characters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import ffsim
import numpy as np
from pyscf import ao2mo, cc, fci, gto, mcscf, scf
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch


@dataclass
class H2Problem:
    r: float
    norb: int
    nelec: tuple[int, int]
    hcore: np.ndarray
    eri: np.ndarray
    e_nuc: float
    t1: np.ndarray
    t2: np.ndarray
    e_fci: float
    fci_vec: np.ndarray           # shape (n_alpha_strings, n_beta_strings)
    fci_strs_a: np.ndarray
    fci_strs_b: np.ndarray


def h2_problem(r: float, basis: str = "sto-3g") -> H2Problem:
    """Integrals, CCSD amplitudes and FCI reference for H2 at bond length r (bohr)."""
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=basis, unit="bohr", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF did not converge at r = {r} bohr")

    norb = mol.nao_nr()
    nelec = (mol.nelectron // 2, mol.nelectron // 2)
    cas = mcscf.CASCI(mf, norb, nelec)
    hcore, e_nuc = cas.get_h1cas()
    eri = ao2mo.restore(1, cas.get_h2cas(), norb)

    ccsd = cc.CCSD(mf)
    ccsd.conv_tol = 1e-10
    ccsd.kernel()
    if not ccsd.converged:
        raise RuntimeError(f"CCSD did not converge at r = {r} bohr")

    solver = fci.FCI(mf)
    solver.conv_tol = 1e-12
    e_fci, fci_vec = solver.kernel()

    return H2Problem(
        r=r, norb=norb, nelec=nelec, hcore=hcore, eri=eri, e_nuc=float(e_nuc),
        t1=ccsd.t1, t2=ccsd.t2, e_fci=float(e_fci), fci_vec=np.asarray(fci_vec),
        fci_strs_a=fci.cistring.make_strings(range(norb), nelec[0]),
        fci_strs_b=fci.cistring.make_strings(range(norb), nelec[1]),
    )


def build_lucj_circuit(norb: int, nelec: tuple[int, int], t1, t2, n_reps: int = 1) -> QuantumCircuit:
    """Hartree-Fock + spin-balanced LUCJ (seeded from CCSD amplitudes), with measurements."""
    pairs_aa = [(p, p + 1) for p in range(norb - 1)]
    pairs_ab = [(p, p) for p in range(norb)]
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2, t1=t1, n_reps=n_reps, interaction_pairs=(pairs_aa, pairs_ab)
    )
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    circuit.measure_all()
    return circuit


_PASS_MANAGER = None


def _pass_manager():
    global _PASS_MANAGER
    if _PASS_MANAGER is None:
        pm = generate_preset_pass_manager(
            optimization_level=1, basis_gates=["x", "sx", "rz", "cx"]
        )
        pm.pre_init = ffsim.qiskit.PRE_INIT
        _PASS_MANAGER = pm
    return _PASS_MANAGER


@dataclass
class SQDRecord:
    r: float
    e_sqd: float
    e_fci: float
    counts: dict
    ci_strs_a: np.ndarray
    ci_strs_b: np.ndarray
    amplitudes: np.ndarray        # shape (len(ci_strs_a), len(ci_strs_b))
    norb: int
    problem: H2Problem = field(repr=False)

    @property
    def dim(self) -> int:
        return self.amplitudes.size


@dataclass
class SQDEnergy:
    """Callable SQD energy for H2: ``SQDEnergy()(r) -> E_SQD`` (hartree)."""

    shots: int = 10_000
    n_reps: int = 1
    seed: int = 12345
    basis: str = "sto-3g"
    last: SQDRecord | None = None
    n_calls: int = 0
    dims: list = field(default_factory=list)   # subspace dimension of every call

    def run(self, r: float) -> SQDRecord:
        p = h2_problem(r, self.basis)
        circuit = build_lucj_circuit(p.norb, p.nelec, p.t1, p.t2, self.n_reps)
        isa = _pass_manager().run(circuit)

        sampler = StatevectorSampler(seed=self.seed)
        bit_array = sampler.run([isa], shots=self.shots).result()[0].data.meas

        result = diagonalize_fermionic_hamiltonian(
            p.hcore, p.eri, bit_array,
            samples_per_batch=self.shots,
            norb=p.norb, nelec=p.nelec,
            num_batches=1,
            max_iterations=1,
            symmetrize_spin=True,
            sci_solver=partial(solve_sci_batch, spin_sq=0.0),
            seed=self.seed,
        )
        st = result.sci_state
        rec = SQDRecord(
            r=r, e_sqd=float(result.energy + p.e_nuc), e_fci=p.e_fci,
            counts=dict(sorted(bit_array.get_counts().items(), key=lambda kv: -kv[1])),
            ci_strs_a=np.asarray(st.ci_strs_a), ci_strs_b=np.asarray(st.ci_strs_b),
            amplitudes=np.asarray(st.amplitudes), norb=p.norb, problem=p,
        )
        self.last = rec
        self.n_calls += 1
        self.dims.append(rec.dim)
        return rec

    def __call__(self, x) -> float:
        return self.run(float(np.ravel(x)[0])).e_sqd


def fci_on_sqd_basis(rec: SQDRecord) -> np.ndarray:
    """FCI vector rearranged onto rec's (alpha string, beta string) determinant grid.

    Determinants missing from the FCI string lists get 0; FCI weight outside the
    SQD subspace is ignored (the overlap then falls below 1).
    """
    p = rec.problem
    ia = {int(s): i for i, s in enumerate(p.fci_strs_a)}
    ib = {int(s): i for i, s in enumerate(p.fci_strs_b)}
    out = np.zeros_like(rec.amplitudes)
    for i, sa in enumerate(rec.ci_strs_a):
        for j, sb in enumerate(rec.ci_strs_b):
            if int(sa) in ia and int(sb) in ib:
                out[i, j] = p.fci_vec[ia[int(sa)], ib[int(sb)]]
    return out


BIT_CONVENTION = (
    "# Bit conventions: in alpha/beta strings, orbital 0 is the RIGHTMOST bit\n"
    "# (bit i set = spatial orbital i occupied). Sampled qiskit bitstrings read\n"
    "# beta|alpha (rightmost norb characters = alpha). Energies in hartree,\n"
    "# distances in bohr. Amplitudes are the SQD ground-state coefficients; the\n"
    "# FCI column is the PySCF FCI vector on the same determinants (sign aligned).\n"
)


def format_record(rec: SQDRecord, header: str) -> str:
    n = rec.norb
    fci_amp = fci_on_sqd_basis(rec)
    overlap = float(np.vdot(fci_amp, rec.amplitudes))
    sign = 1.0 if overlap >= 0 else -1.0
    lines = [
        header,
        f"E_SQD = {rec.e_sqd:.10f}   E_FCI = {rec.e_fci:.10f}   "
        f"E_SQD - E_FCI = {rec.e_sqd - rec.e_fci:+.2e}   |<SQD|FCI>| = {abs(overlap):.10f}",
        f"subspace: {len(rec.ci_strs_a)} alpha strings x {len(rec.ci_strs_b)} beta strings "
        f"= {rec.dim} determinants",
        "sampled bitstrings (beta|alpha): "
        + "  ".join(f"{b}: {c}" for b, c in rec.counts.items()),
        f"  {'alpha':<{max(n, 5)}}  {'beta':<{max(n, 5)}}  {'SQD amplitude':>14}  {'FCI amplitude':>14}",
    ]
    for i, sa in enumerate(rec.ci_strs_a):
        for j, sb in enumerate(rec.ci_strs_b):
            lines.append(
                f"  {format(int(sa), f'0{n}b'):<{max(n, 5)}}  {format(int(sb), f'0{n}b'):<{max(n, 5)}}"
                f"  {rec.amplitudes[i, j]:+14.10f}  {sign * fci_amp[i, j]:+14.10f}"
            )
    return "\n".join(lines) + "\n\n"
