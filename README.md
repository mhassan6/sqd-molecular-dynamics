# Ab initio molecular dynamics with sample-based quantum diagonalization

A small, modular Python package (`qmd`) for ab initio molecular dynamics (AIMD) in which
the potential energy can come from any electronic-structure method, including
**sample-based quantum diagonalization (SQD)** on quantum-circuit samples.

The H₂ examples reproduce the bond-vibration dynamics of
D. A. Fedorov, M. J. Otten, S. K. Gray, Y. Alexeev,
*Ab initio molecular dynamics on quantum computers*, J. Chem. Phys. **154**, 164103 (2021)
([doi:10.1063/5.0046930](https://doi.org/10.1063/5.0046930)), with SQD in place of VQE.

![H₂ phase space: FCI, B3LYP and SQD](h2_phase_space_comparison.png)

## How it works

The integrator only needs a callable `energy_fn(x) -> float`. Forces come from central
finite differences, `F = -[E(x+d) - E(x-d)] / 2d`, and positions are advanced with
velocity Verlet (`qmd/integrators.py`). The same integrator is driven by:

| Backend | Example | Energy source |
|---|---|---|
| Analytic | `examples/harmonic_oscillator.py` | `V(x) = k x²/2` (integrator test, known period) |
| DFT | `examples/h2_dft.py` | PySCF B3LYP/STO-3G |
| FCI | `examples/h2_fci.py` | PySCF FCI/STO-3G (exact reference) |
| **SQD** | `examples/h2_sqd_md.py` | quantum-circuit samples + subspace diagonalization |

The SQD energy at each geometry (`qmd/sqd.py`) is computed by:

1. **PySCF**: RHF, active-space integrals, and CCSD amplitudes.
2. **ffsim**: a local unitary cluster Jastrow (LUCJ) circuit seeded from the CCSD amplitudes.
3. **Qiskit**: sample bitstrings from the circuit (`StatevectorSampler`, 10⁴ shots).
4. **qiskit-addon-sqd**: project the Hamiltonian onto the sampled determinants and diagonalize.

## Results

* SQD reproduces the FCI trajectory: energies and ground-state vectors match FCI to
  ~10⁻¹⁵ Ha at every MD step (`h2_sqd_states.txt`, `h2_sqd_fci_test.txt`).
* B3LYP gives a slightly wider orbit, as in Fig. 6 of Fedorov et al.

**Scope.** H₂ in STO-3G has only four determinants and the sampler here is noiseless,
so SQD recovers the exact answer by construction. This is a validated end-to-end
pipeline, not a demonstration of quantum advantage. Natural next steps are larger
molecules/active spaces (where the sampled subspace is truly a subset of the full
space), shot noise and hardware noise models, and analytic (Hellmann–Feynman) forces.

The bond coordinate uses the hydrogen-atom mass rather than the reduced mass, to match
the reference paper's setup (~10.5 fs period).

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python examples/harmonic_oscillator.py   # integrator sanity check
python examples/h2_dft.py                # DFT trajectory
python examples/h2_fci.py                # FCI reference trajectory
python examples/h2_sqd_fci_test.py       # SQD vs FCI at fixed bond lengths
python examples/h2_sqd_md.py             # SQD-driven MD (overlays FCI if present)
python examples/plot_phase_spaces.py     # comparison figure from saved .npz files
```

All quantities are in atomic units (bohr, hartree, electron masses, a.u. of time).
Outputs (`*.png`, `*_trajectory.npz`, `h2_sqd_states.txt`) are written to the repository root.

`MolecularDynamics_Tutorial*.ipynb` are introductory notebooks on DFT potential-energy
curves and classical MD for H₂.

## Development

The code was written with [Claude Code](https://claude.com/claude-code) as an AI pair
programmer. I set the physics, units, and conventions and reviewed the code it wrote.
Results were accepted only after passing independent checks: an analytic oscillator,
DFT and FCI baselines, and an SQD-versus-FCI comparison at every step.

## License

MIT; see [LICENSE](LICENSE).
