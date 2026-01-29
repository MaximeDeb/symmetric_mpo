# Symmetric MPO Library

A Python library for efficient manipulation of **symmetric Matrix Product Operators (MPOs)** with U(1) symmetry conservation. Designed for quantum many-body physics applications including:

- Time evolution using TEBD (Time-Evolving Block Decimation)
- Natural orbital rotations via Givens decomposition
- Correlation function computations
- Support for Heisenberg and IRLM models

## Features

- **Block-sparse storage**: Only non-zero symmetry sectors are stored
- **Flexible symmetry**: Supports particle number (α = L+1) and particle-hole (α = -1) conservation
- **High-order Trotter**: 1st, 2nd, and 4th order decompositions
- **Efficient contractions**: Optimized tensor operations preserving sparsity
- **Natural orbital basis**: Rotation to expose factorized structure

## Installation

```bash
git clone https://github.com/username/lib-symmetric_mpo.git
cd lib-symmetric_mpo
pip install -e .
```

## Quick Start

```python
from symmetric_mpo import (
    SymmetricMPO,
    build_trotter_sequence,
    apply_gate,
    compute_R_matrix
)

# Create identity MPO for L=10 spin-1/2 chain
L, d = 10, 2
params = {'L': L, 'd': d, 'J': 1.0, 'Jz': 1.0, 'phys_dims': 1}

mpo = SymmetricMPO(
    L, d, q_alpha=L, phys_dims=1,
    chi_max=256, alpha=-1, initial="Id"
)

# Build 2nd order Trotter sequence for Heisenberg model
steps, U, U_dag, _ = build_trotter_sequence(
    order=2, n_steps=100, dt=0.01, n_parts=2,
    obs_interval=10, params=params, model="Heis_nn"
)

# Define gate layers (even/odd bonds)
gate_layers = {
    "H0": [(i, i+1) for i in range(0, L-1, 2)],
    "H1": [(i, i+1) for i in range(1, L-1, 2)]
}

# Time evolve
for step in steps:
    for (l1, l2) in gate_layers[step.layer]:
        mpo, _ = apply_gate(
            mpo, U[step.layer, step.dt],
            U_dag[step.layer, step.dt], l1, l2
        )
    
    if step.compute_obs:
        R = compute_R_matrix(mpo, unitary=True)
        print(f"t={step.dt}: max eigenvalue = {R.max():.4f}")
```

## Module Overview

### Core Classes

- **`SymmetricMPO`**: Main MPO class with Vidal representation
- **`SymmetricTensor`**: Block-sparse tensor with symmetry
- **`SymmetricLambda`**: Diagonal singular value matrices
- **`SymmetricGate`**: Two-site gates for TEBD

### Key Functions

| Function | Description |
|----------|-------------|
| `tensor_contract(A, B, indices)` | Contract symmetric tensors |
| `apply_gate(mpo, U, U_dag, l1, l2)` | Apply TEBD gate |
| `build_trotter_sequence(...)` | Build Trotter decomposition |
| `compute_R_matrix(mpo)` | Compute correlation matrix |
| `givens_rotations(mat, loc, sect)` | Givens rotation decomposition |

## Supported Models

### Heisenberg XXZ Chain
```python
H = J/2 Σ(S+_i S-_{i+1} + h.c.) + Jz Σ Sz_i Sz_{i+1}
```

### Interacting Resonant Level Model (IRLM)
```python
H = V(c†_0 c_1 + h.c.) + U n_0 n_1 + γ Σ(c†_i c_{i+1} + h.c.)
```

## Algorithm Details

### Symmetry Conservation

The library exploits U(1) symmetry to reduce computational cost. For a system with particle number conservation, the Hilbert space decomposes as:

```
H = ⊕_N H_N
```

Only blocks with consistent quantum numbers are stored and contracted.

### Truncation Strategies

- **`global`**: Keep top χ singular values across all sectors
- **`block`**: Keep top χ per sector
- **`block_threshold`**: Use largest sector to set threshold

### Natural Orbital Rotation

After time evolution, the MPO can be rotated to the natural orbital basis where the correlation matrix is diagonal. This is done through a sequence of Givens rotations applied as local gates.

## Performance

1. **Use `data_as_tensors=False`** for 1D systems (matrix storage is faster)
2. **Choose appropriate `chi_max`** based on entanglement growth
3. **Use 4th order Trotter** for high-accuracy long-time evolution
4. **Enable `optimized=True`** in `compute_R_matrix` for O(L²) scaling

## File Formats

MPOs can be saved/loaded using HDF5:

```python
mpo.export("my_mpo.h5")
loaded = SymmetricMPO.load("my_mpo.h5")
```

## Acknowledgments

Key references:

- Vidal, "Efficient classical simulation of slightly entangled quantum computations" (2003)
- Hastings, "An area law for one-dimensional quantum systems" (2007)
- Schollwöck, "The density-matrix renormalization group in the age of matrix product states" (2011)
- Gisti, Luitz, Debertolis, "Symmetry resolved out-of-time-order correlators of Heisenberg spin chains using projected matrix product operators" (2025)