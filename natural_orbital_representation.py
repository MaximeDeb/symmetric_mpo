"""
Example: Natural Orbital Representation of MPOs.

This script demonstrates time evolution of a symmetric MPO using TEBD
and subsequent rotation to the natural orbital basis.

Models supported:
- Heisenberg XXZ chain ("Heis_nn")
- Interacting Resonant Level Model ("IRLM")
"""

import numpy as np
import time
from pathlib import Path

# Import the library
from symmetric_mpo import (
    SymmetricMPO,
    SymmetricGate,
    build_trotter_sequence,
    apply_gate,
    compute_R_matrix,
    givens_rotations,
    rotation_from_givens,
)


def main():
    # =========================================================================
    # Model Parameters
    # =========================================================================
    L = 10                # System size
    d = 2                 # Local dimension (spin-1/2)
    model = "IRLM"        # "Heis_nn" or "IRLM"
    save_data = False     # Save results to file
    
    # =========================================================================
    # Tensor Network Parameters
    # =========================================================================
    phys_dims = 1               # Physical dimensions per tensor
    alpha = -1                  # Super-charge: -1 (particle-hole) or L+1 (particle number)
    truncation_type = "global"  # "global", "block", or "block_threshold"
    chi_max = 1024              # Maximum bond dimension
    th_sing_vals = 1e-12        # Singular value threshold
    data_as_tensors = False     # False for matrix storage (faster in 1D)
    
    # Symmetry sector
    s = L // 2 if alpha == L + 1 else L
    
    # =========================================================================
    # Time Evolution Parameters
    # =========================================================================
    trotter_order = 4     # Trotter order (1, 2, or 4)
    T_final = 0.5         # Final time
    dt = 0.01             # Time step
    obs_interval = 20000  # Compute observables every N steps
    
    n_steps = int(T_final / dt)
    n_obs = (n_steps // obs_interval) + 1
    
    # =========================================================================
    # Model-Specific Parameters
    # =========================================================================
    if model == "Heis_nn":
        J = 1.0
        Jz = 1.0
        params = {
            'L': L, 'd': d, 'J': J, 'Jz': Jz,
            'phys_dims': phys_dims,
            'periodT': 0, 'hmean': 0, 'hdrive': 0
        }
        # Even/odd bond decomposition
        gate_layers = {
            "H0": [np.array([i, i + 1]) for i in range(0, L - 1, 2)],
            "H1": [np.array([i, i + 1]) for i in range(1, L - 1, 2)]
        }
        
    elif model == "IRLM":
        U_int = 0.2   # Impurity interaction
        V = 0.2       # Impurity-bath coupling
        gamma = 0.5   # Bath hopping
        ed = 0        # Impurity energy
        params = {
            'L': L, 'd': d,
            'Uint': U_int, 'V': V, 'gamma': gamma, 'ed': ed,
            'phys_dims': phys_dims
        }
        # Three-part decomposition for IRLM
        gate_layers = {
            "H0": [np.array([0, 1])],                                    # Impurity
            "H1": [np.array([i, i + 1]) for i in range(2, L - 1, 2)],   # Even bath
            "H2": [np.array([i, i + 1]) for i in range(1, L - 1, 2)]    # Odd bath
        }
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Print configuration
    print("=" * 60)
    print(f"Model: {model}")
    print("-" * 60)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"\nTensor Network:")
    print(f"  chi_max: {chi_max}")
    print(f"  alpha: {alpha}")
    print(f"  truncation: {truncation_type}")
    print(f"\nTime Evolution:")
    print(f"  T_final: {T_final}")
    print(f"  dt: {dt}")
    print(f"  Trotter order: {trotter_order}")
    print("=" * 60 + "\n")
    
    # =========================================================================
    # Build Trotter Sequence
    # =========================================================================
    n_parts = len(gate_layers)
    layer_names = list(gate_layers.keys())
    
    trotter_steps, U, U_dag, n_layers = build_trotter_sequence(
        order=trotter_order,
        n_steps=n_steps,
        dt=dt,
        n_parts=n_parts,
        obs_interval=obs_interval,
        params=params,
        alpha=alpha,
        data_as_tensors=data_as_tensors,
        model=model,
        layer_names=layer_names
    )
    
    print(f"Trotter sequence: {len(trotter_steps)} steps")
    print(f"Layers per time step: {n_layers}")
    
    # =========================================================================
    # Initialize MPO
    # =========================================================================
    mpo = SymmetricMPO(
        L, d, s, phys_dims,
        chi_max=chi_max,
        alpha=alpha,
        initial="Id",
        data_as_tensors=data_as_tensors,
        th_sing_vals=th_sing_vals,
        truncation_type=truncation_type
    )
    
    print("\nInitial bond dimensions:")
    mpo.print_bond_dimensions()
    
    # =========================================================================
    # Storage for Results
    # =========================================================================
    R_eigenvalues = np.zeros((2 * L, n_obs), dtype=float)
    cumulative_error = np.zeros((L, n_obs), dtype=float)
    times = np.zeros(n_obs, dtype=float)
    
    # =========================================================================
    # Time Evolution
    # =========================================================================
    t = 0.0
    obs_idx = 0
    start_time = time.time()
    step_start = time.time()
    
    for step in trotter_steps:
        # Apply gates for this layer
        for (l1, l2) in gate_layers[step.layer]:
            mpo, s_disc = apply_gate(
                mpo, U[step.layer, step.dt], U_dag[step.layer, step.dt],
                l1, l2, both_sides=False
            )
            cumulative_error[l1, obs_idx] += s_disc
        
        # Check for completed time step
        if step.new_time:
            t += dt
            elapsed = time.time() - step_start
            print(f"t = {t:.4f}, computation time: {elapsed:.2f}s")
            step_start = time.time()
            
            # Compute observables
            if step.compute_obs:
                R = compute_R_matrix(mpo, unitary=True, optimized=True)
                R_ev, R_evecs = np.linalg.eigh(R)
                
                R_eigenvalues[:, obs_idx] = R_ev
                times[obs_idx] = t
                
                print(f"  Bond dimensions:")
                mpo.print_bond_dimensions()
                
                obs_idx += 1
    
    total_time = time.time() - start_time
    print(f"\nTotal evolution time: {total_time:.2f}s")
    
    # =========================================================================
    # Rotate to Natural Orbital Basis
    # =========================================================================
    print("\nRotating to natural orbital basis...")
    
    # Compute final R matrix
    R = compute_R_matrix(mpo, unitary=True)
    A = R[:L, L:].copy()
    
    # Get eigenvalues of the off-diagonal block
    A_ev, A_evecs = np.linalg.eigh(A @ A.conj().T)
    A_ev = np.sqrt(np.maximum(A_ev, 0))
    
    print(f"\nOff-diagonal singular values (deviation from 0.5):")
    print(f"  {0.5 - np.abs(A_ev)}")
    
    # Rotate MPO using Givens rotations
    mpo_rotated = mpo.copy()
    _rotate_mpo_to_natural_basis(mpo_rotated, A)
    
    print("\nBond dimensions after rotation:")
    mpo_rotated.print_bond_dimensions()
    
    # =========================================================================
    # Save Results
    # =========================================================================
    if save_data:
        import h5py
        
        filename = f"../data/MPO/{model}/U_{model}"
        for key, val in params.items():
            filename += f"_{key}{val}"
        filename += ".h5"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(filename, "w") as f:
            f.create_dataset("R_eigenvalues", data=R_eigenvalues)
            f.create_dataset("cumulative_error", data=cumulative_error)
            f.create_dataset("times", data=times)
            for key, val in params.items():
                f.attrs[key] = val
        
        print(f"\nData saved to {filename}")


def _rotate_mpo_to_natural_basis(mpo, A, method="Hermitian"):
    """
    Rotate an MPO to the natural orbital basis using Givens rotations.
    
    Parameters
    ----------
    mpo : SymmetricMPO
        The MPO to rotate (modified in-place).
    A : ndarray
        The off-diagonal block of the R matrix.
    method : str
        "Hermitian" for eigenvalue decomposition, "Unitary" for SVD.
    """
    L = mpo.L
    params = {'L': L, 'd': mpo.d}
    
    active = np.arange(L)
    
    for i in range(L):
        l = L - i
        sect = active[:l]
        A_sub = A[:l, :l]
        
        if method == "Hermitian":
            ev, evecs = np.linalg.eigh(A_sub)
            order = np.argsort(0.5 - np.abs(ev))[::-1]
            ev, evecs = ev[order], evecs[:, order]
            
            indices, givens = givens_rotations(evecs, [l - 1], sect, direction="right")
            L_givens = R_givens = givens
        else:
            U, S, Vd = np.linalg.svd(A_sub)
            order = np.argsort(0.5 - np.abs(S))[::-1]
            S = S[order]
            L_ev, R_ev = U[:, order], Vd.conj().T[:, order]
            
            indices, L_givens = givens_rotations(L_ev, [l - 1], sect, direction="right")
            _, R_givens = givens_rotations(R_ev.conj(), [l - 1], sect, direction="right")
        
        # Apply Givens rotations
        for m, ind in enumerate(indices):
            L_giv = L_givens[m].T
            R_giv = R_givens[m].T if method != "Hermitian" else L_giv
            
            U_gate = SymmetricGate(
                2 * mpo.phys_dims, 0, params,
                gate_type="Hamiltonian", model="givens",
                alpha=mpo.alpha, data_as_tensors=mpo.data_as_tensors,
                rot=L_giv
            )
            U_dag_gate = SymmetricGate(
                2 * mpo.phys_dims, 0, params,
                gate_type="Hamiltonian", model="givens",
                dag=True, alpha=mpo.alpha, data_as_tensors=mpo.data_as_tensors,
                rot=R_giv
            )
            
            mpo, _ = apply_gate(mpo, U_gate, U_dag_gate, ind[0], ind[1], both_sides=True)
        
        # Update A matrix
        if method == "Hermitian":
            rot = rotation_from_givens(indices, givens, sect)
            A[:l, :l] = rot @ A[:l, :l] @ rot.conj().T
        else:
            R = compute_R_matrix(mpo, unitary=True)
            A = R[:L, L:].copy()
        
        print(f"  Rotation step {i + 1}/{L}")


if __name__ == "__main__":
    main()
