"""
Linear algebra operations for symmetric tensor networks.

This module provides tensor contraction, trace operations, and
correlation matrix computations for symmetric MPOs.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import TYPE_CHECKING

from .tensor import SymmetricTensor, swap_indices, mask_coordinates
from . import sparse_ops as so

if TYPE_CHECKING:
    from .mpo import SymmetricMPO


def tensor_contract(
    A: SymmetricTensor,
    B: SymmetricTensor,
    indices: tuple[list[int], list[int]]
) -> SymmetricTensor:
    """
    Contract two symmetric tensors along specified indices.
    
    Performs the tensor product of A and B, summing over the legs
    specified in indices while preserving the block-sparse structure.
    
    Parameters
    ----------
    A : SymmetricTensor
        First tensor.
    B : SymmetricTensor
        Second tensor.
    indices : tuple of (list, list)
        (indices_A, indices_B) specifying which legs to contract.
        
    Returns
    -------
    SymmetricTensor
        The contracted tensor.
    """
    alpha = A.alpha
    x_ind = np.array(indices[0])
    y_ind = np.array(indices[1])
    
    n_A = np.arange(A.n_legs)
    n_B = np.arange(B.n_legs)
    
    # Partition legs
    left_A = np.setdiff1d(n_A, x_ind)
    right_A = x_ind
    left_B = y_ind
    right_B = np.setdiff1d(n_B, y_ind)
    
    # Reshape tensor blocks as matrices
    A_blocks, A_shapes = so.reshape_data_tensors(A, left_A, right_A)
    B_blocks, B_shapes = so.reshape_data_tensors(B, left_B, right_B)
    
    # Determine symmetry sectors for contraction
    R_arrow_A = A.arrows[right_A] == 'i'
    R_sigma_A = A.leg_type[right_A] == 's'
    L_arrow_B = B.arrows[left_B] == 'i'
    L_sigma_B = B.leg_type[left_B] == 's'
    
    sectors_A = np.sum(
        A.coordinates[right_A] * 
        (-1 * R_arrow_A[:, None] + (~R_arrow_A)[:, None]) * 
        (alpha * R_sigma_A[:, None] + (~R_sigma_A[:, None])),
        axis=0
    )
    sectors_B = np.sum(
        B.coordinates[left_B] * 
        (L_arrow_B[:, None] - 1 * (~L_arrow_B)[:, None]) * 
        (alpha * L_sigma_B[:, None] + (~L_sigma_B[:, None])),
        axis=0
    )
    
    unique_A = np.unique(sectors_A)
    unique_B = np.unique(sectors_B)
    sectors = np.intersect1d(unique_A, unique_B)
    n_sectors = len(sectors)
    
    A_sector_mask = sectors_A[None, :] == sectors[:, None]
    B_sector_mask = sectors_B[None, :] == sectors[:, None]
    
    L_coords, A_shapes_sect, n_L_A, n_R_A, id_A = so.construct_subblock_sectors(
        A, left_A, right_A, A_shapes, A_sector_mask, n_sectors
    )
    R_coords, B_shapes_sect, n_L_B, n_R_B, id_B = so.construct_subblock_sectors(
        B, left_B, right_B, B_shapes, B_sector_mask, n_sectors
    )
    
    L_coords, R_coords, A_shapes_sect, B_shapes_sect, empty = so.check_subblock_sectors(
        L_coords, R_coords, A_shapes_sect, B_shapes_sect, id_A, id_B
    )
    
    n_blocks_per_sector = n_L_A * n_R_B
    phys_dims_L = int(np.sum(A.leg_type[left_A] != 'v') / 2)
    phys_dims_R = int(np.sum(B.leg_type[right_B] != 'v') / 2)
    
    C = SymmetricTensor(
        A.L, A.d, phys_dims_L + phys_dims_R,
        alpha=alpha,
        data_as_tensors=A.data_as_tensors,
        n_legs=len(left_A) + len(right_B),
        n_sectors=np.sum(n_blocks_per_sector)
    )
    C.arrows = np.concatenate([A.arrows[left_A], B.arrows[right_B]])
    C.leg_type = np.concatenate([A.leg_type[left_A], B.leg_type[right_B]])
    
    if not A.data_as_tensors and C.leg_type.size > 0:
        virtual_mask = C.leg_type == "v"
    else:
        virtual_mask = ...
    
    n_coord = 0
    valid_blocks = np.ones(np.sum(n_blocks_per_sector), dtype=bool)
    
    for s in range(n_sectors):
        n_out = n_blocks_per_sector[s]
        if empty[s]:
            valid_blocks[n_coord:n_coord + n_out] = False
        else:
            mat_A, list_L_A, list_R_A, shape_L_A, shape_R_A = so.construct_matrix_from_subblocks(
                A_blocks, A_shapes_sect[s], L_coords[s], A_sector_mask[s]
            )
            mat_B, list_L_B, list_R_B, shape_L_B, shape_R_B = so.construct_matrix_from_subblocks(
                B_blocks, B_shapes_sect[s], R_coords[s], B_sector_mask[s]
            )
            
            mat_C = mat_A @ mat_B
            
            # Store coordinates
            A_coords_out = np.repeat(
                A.coordinates[left_A][:, A_sector_mask[s]][:, list_L_A],
                n_R_B[s], axis=1
            )
            B_coords_out = np.tile(
                B.coordinates[right_B][:, B_sector_mask[s]][:, list_R_B],
                (1, n_L_A[s])
            )
            C.coordinates[:, n_coord:n_coord + n_out] = np.concatenate(
                [A_coords_out, B_coords_out], axis=0
            )
            
            A_shapes_out = np.repeat(
                A.shapes[left_A][:, A_sector_mask[s]][:, list_L_A],
                n_R_B[s], axis=1
            )
            B_shapes_out = np.tile(
                B.shapes[right_B][:, B_sector_mask[s]][:, list_R_B],
                (1, n_L_A[s])
            )
            C.shapes[:, n_coord:n_coord + n_out] = np.concatenate(
                [A_shapes_out, B_shapes_out], axis=0
            )
            
            if A.data_as_tensors:
                reshape_dims = C.shapes[:, n_coord:n_coord + n_out]
            else:
                reshape_dims = C.shapes[virtual_mask, n_coord:n_coord + n_out]
            
            cuts_rows = np.cumsum(shape_L_A)[:-1]
            cuts_cols = np.cumsum(shape_R_B)[:-1]
            split_C = so.block_split(mat_C, cuts_rows, cuts_cols)
            
            C.data[n_coord:n_coord + n_out] = [
                split_C[i].astype(complex).reshape(reshape_dims[:, i])
                for i in range(n_out)
            ]
        
        n_coord += n_out
    
    C = mask_coordinates(C, valid_blocks)
    return C


def trace_mpo_product(
    A: 'SymmetricMPO',
    B: 'SymmetricMPO',
    conj_A: bool = False,
    conj_B: bool = False
) -> complex:
    """
    Compute Tr(A^dag B) using the PacMan method.
    
    Contracts the MPO product efficiently without forming the full matrix.
    
    Parameters
    ----------
    A : SymmetricMPO
        First MPO.
    B : SymmetricMPO
        Second MPO.
    conj_A : bool
        Take complex conjugate of A.
    conj_B : bool
        Take complex conjugate of B.
        
    Returns
    -------
    complex
        The trace Tr(A^dag B) or Tr(A B^dag).
    """
    A_copy = A.copy()
    B_copy = B.copy()
    
    # Initialize PacMan tensor
    pac = SymmetricTensor(
        A.L, A.d, 0,
        alpha=A.alpha,
        data_as_tensors=A.data_as_tensors,
        n_legs=2, n_sectors=1
    )
    
    if A.alpha == -1:
        pac.coordinates[:, 0] = (A.L, A.L)
        pac.leg_sectors = np.array([np.array([A.L]), np.array([A.L])], dtype=object)
    elif A.alpha == A.L + 1:
        pac.coordinates[:, 0] = (0, 0)
        pac.leg_sectors = np.array([np.array([0]), np.array([0])], dtype=object)
    
    pac.shapes = np.ones((2, 1), dtype=np.intp)
    pac.data[0] = np.ones((1, 1), dtype=complex)
    pac.arrows = np.array(['o', 'o'])
    pac.leg_type = np.array(['v', 'v'])
    
    for i in range(A.L):
        B_A = A_copy.TN[f"B{i}"]
        B_B = B_copy.TN[f"B{i}"]
        
        if conj_A:
            for b in range(B_A.n_sectors):
                B_A.data[b] = B_A.data[b].conj()
            dims_A = [0, 1, 2]
            dims_B = [0, 1, 2]
        elif conj_B:
            for b in range(B_B.n_sectors):
                B_B.data[b] = B_B.data[b].conj()
            dims_A = [0, 1, 2]
            dims_B = [0, 1, 2]
        else:
            B_A.leg_type[[1, 2]] = B_A.leg_type[[2, 1]]
            dims_A = [0, 1, 2]
            dims_B = [0, 2, 1]
        
        pac_A = tensor_contract(pac, B_A, ([0], [0]))
        pac_A.arrows[[1, 2]] = 'o'
        pac = tensor_contract(pac_A, B_B, (dims_A, dims_B))
    
    if len(pac.data) == 0:
        return 0.0
    return np.sum(pac.data[0])


def site_pacman(
    O1: 'SymmetricMPO',
    O2: 'SymmetricMPO',
    conj_A: bool = False,
    conj_B: bool = False,
    left: bool = True,
    right: bool = True
) -> tuple[dict, dict]:
    """
    Compute PacMan environments at each site.
    
    Stores the left and right partial contractions, useful for
    computing local observables efficiently.
    
    Parameters
    ----------
    O1 : SymmetricMPO
        First MPO.
    O2 : SymmetricMPO
        Second MPO.
    conj_A : bool
        Conjugate first MPO.
    conj_B : bool
        Conjugate second MPO.
    left : bool
        Compute left environments.
    right : bool
        Compute right environments.
        
    Returns
    -------
    L_PM : dict
        Left PacMan at each site.
    R_PM : dict
        Right PacMan at each site.
    """
    A = O1.copy()
    B = O2.copy()
    L = A.L
    
    # Initialize left PacMan
    L_pac = SymmetricTensor(
        A.L, A.d, 0,
        alpha=A.alpha,
        n_legs=2, n_sectors=1
    )
    R_pac = SymmetricTensor(
        A.L, A.d, 0,
        alpha=A.alpha,
        n_legs=2, n_sectors=1
    )
    
    A_sect = A.TN[f"B{L-1}"].coordinates[-1, 0]
    B_sect = B.TN[f"B{L-1}"].coordinates[-1, 0]
    
    if A.alpha == -1:
        L_pac.coordinates[:, 0] = (L, L)
        L_pac.leg_sectors = np.array([np.array([L]), np.array([L])], dtype=object)
        R_pac.coordinates[:, 0] = (A_sect, B_sect)
        R_pac.leg_sectors = np.array([np.array([A_sect]), np.array([B_sect])], dtype=object)
    elif A.alpha == L + 1:
        L_pac.coordinates[:, 0] = (0, 0)
        L_pac.leg_sectors = np.array([np.array([0]), np.array([0])], dtype=object)
        R_pac.coordinates[:, 0] = (A_sect, B_sect)
        R_pac.leg_sectors = np.array([np.array([A_sect]), np.array([B_sect])], dtype=object)
    
    L_pac.shapes = np.ones((2, 1), dtype=np.intp)
    L_pac.data[0] = np.ones((1, 1), dtype=complex)
    L_pac.arrows = np.array(['o', 'o'])
    L_pac.leg_type = np.array(['v', 'v'])
    
    R_pac.shapes = np.ones((2, 1), dtype=np.intp)
    R_pac.data[0] = np.ones((1, 1), dtype=complex)
    R_pac.arrows = np.array(['i', 'i'])
    R_pac.leg_type = np.array(['v', 'v'])
    
    L_PM = {}
    R_PM = {}
    
    for i in range(L):
        L_A = A.TN[f"B{i}"].copy()
        L_B = B.TN[f"B{i}"].copy()
        R_A = A.TN[f"B{L-1-i}"].copy()
        R_B = B.TN[f"B{L-1-i}"].copy()
        
        if conj_A:
            for b in range(L_A.n_sectors):
                L_A.data[b] = L_A.data[b].conj()
            for b in range(R_A.n_sectors):
                R_A.data[b] = R_A.data[b].conj()
            dims_L_A, dims_L_B = [0, 1, 2], [0, 1, 2]
            dims_R_A, dims_R_B = [1, 2, 3], [1, 2, 3]
        elif conj_B:
            for b in range(L_B.n_sectors):
                L_B.data[b] = L_B.data[b].conj()
            for b in range(R_B.n_sectors):
                R_B.data[b] = R_B.data[b].conj()
            dims_L_A, dims_L_B = [0, 1, 2], [0, 1, 2]
            dims_R_A, dims_R_B = [1, 2, 3], [1, 2, 3]
        else:
            dims_L_A, dims_L_B = [0, 1, 2], [0, 2, 1]
            dims_R_A, dims_R_B = [1, 2, 3], [2, 1, 3]
            L_A.leg_type[[1, 2]] = L_A.leg_type[[2, 1]]
            R_A.leg_type[[1, 2]] = R_A.leg_type[[2, 1]]
        
        if left:
            L_PM[i] = L_pac.copy()
            pac_A = tensor_contract(L_pac, L_A, ([0], [0]))
            pac_A.arrows[[1, 2]] = 'o'
            L_pac = tensor_contract(pac_A, L_B, (dims_L_A, dims_L_B))
        
        if right:
            R_PM[L - 1 - i] = R_pac.copy()
            pac_A = tensor_contract(R_A, R_pac, ([3], [0]))
            pac_A.arrows[[1, 2]] = 'o'
            R_pac = tensor_contract(R_B, pac_A, (dims_R_B, dims_R_A))
            R_pac = swap_indices(R_pac, [1, 0], [0, 1])
        
        if not conj_A and not conj_B:
            L_A.leg_type[[1, 2]] = L_A.leg_type[[2, 1]]
            R_A.leg_type[[1, 2]] = R_A.leg_type[[2, 1]]
    
    return L_PM, R_PM


def compute_R_matrix(
    mpo: 'SymmetricMPO',
    unitary: bool = False,
    optimized: bool = True
) -> NDArray[np.complexfloating]:
    """
    Compute the R matrix (correlation matrix) for an MPO.
    
    The R matrix is defined as R_ij = <c_i^dag c_j> where the expectation
    is taken with respect to the operator interpreted as a density matrix.
    
    Parameters
    ----------
    mpo : SymmetricMPO
        The operator.
    unitary : bool
        Whether the operator is unitary (enables optimizations).
    optimized : bool
        Use optimized O(L^2) algorithm vs naive O(L^3).
        
    Returns
    -------
    R : ndarray, shape (2L, 2L), complex
        The correlation matrix.
    """
    from .mpo import apply_fermionic_op, apply_spin_op, mask_coordinates
    
    O = mpo.copy()
    norm_O = np.abs(O.norm())
    L = O.L
    
    R = np.zeros((2 * L, 2 * L), dtype=complex)
    
    if optimized:
        _, R_PM_O = site_pacman(O, O, left=False, conj_A=True)
        
        O_sgn_L = apply_spin_op("sz", O, np.arange(L), side="L")
        O_sgn_R = apply_spin_op("sz", O, np.arange(L), side="R")
        
        for i in range(2 * L):
            ind_i = i // 2
            side_i = "L" if i % 2 == 0 else "R"
            O_s = O_sgn_R if side_i == "L" else O_sgn_L
            
            if not unitary:
                O1_R = apply_fermionic_op("c", O, ind_i, side_i, sign_side="R")
                L_PM_between, _ = site_pacman(O1_R, O, right=False, conj_A=True)
            
            O1_L = apply_fermionic_op("c", O, ind_i, side_i, sign_side="L")
            L_PM_start, _ = site_pacman(O1_L, O_s, right=False, conj_A=True)
            
            for j in range(i, 2 * L):
                if unitary and (i % 2 == j % 2):
                    continue
                
                ind_j = j // 2
                side_j = "L" if j % 2 == 0 else "R"
                
                O1_loc = O1_L.TN[f"B{ind_j}"].copy()
                for b in range(O1_loc.n_sectors):
                    O1_loc.data[b] = O1_loc.data[b].conj()
                
                O2_loc = O.TN[f"B{ind_j}"].copy()
                
                # Apply c_j locally
                leg_nb = 2 if side_j == "L" else 1
                inc_st = 1 if side_j == "L" else 0
                mask = O2_loc.coordinates[leg_nb, :] == inc_st
                O2_loc = mask_coordinates(O2_loc, mask)
                
                O2_loc.coordinates[leg_nb, :] = (1 + inc_st) % 2
                O2_loc.coordinates[-1, :] = (
                    O2_loc.coordinates[0, :] +
                    O.alpha * O2_loc.coordinates[1, :] +
                    O2_loc.coordinates[2, :]
                )
                O2_loc.leg_sectors = np.array([
                    np.unique(O2_loc.coordinates[l, :])
                    for l in range(O2_loc.n_legs)
                ], dtype=object)
                
                # Adapt right PacMan sectors
                R_pac = R_PM_O[ind_j].copy()
                R_pac.coordinates[0, :] += -1 if side_i == "L" else O.alpha
                R_pac.coordinates[1, :] += -1 if side_j == "L" else O.alpha
                R_pac.leg_sectors = np.array([
                    np.unique(R_pac.coordinates[l, :])
                    for l in range(R_pac.n_legs)
                ], dtype=object)
                
                # Contract
                L_PM = L_PM_between[ind_j] if side_i == side_j else L_PM_start[ind_j]
                pac_A = tensor_contract(L_PM, O1_loc, ([0], [0]))
                pac_A.arrows[[1, 2]] = 'o'
                B_out = tensor_contract(pac_A, O2_loc, ([0, 1, 2], [0, 1, 2]))
                
                result = tensor_contract(B_out, R_pac, ([0, 1], [0, 1]))
                R[L * (i % 2) + ind_i, L * (j % 2) + ind_j] = np.sum(result.data) / norm_O
        
        if unitary:
            diag_val = 0.5 * 2 ** float(L) / norm_O
            np.fill_diagonal(R[:L, :L], diag_val)
            np.fill_diagonal(R[L:, L:], 1 - diag_val)
    else:
        # Naive O(L^3) implementation
        for i in range(2 * L):
            ind_i = i // 2
            side_i = "L" if i % 2 == 0 else "R"
            O1 = apply_fermionic_op("c", O, ind_i, side_i)
            
            for j in range(i, 2 * L):
                ind_j = j // 2
                side_j = "L" if j % 2 == 0 else "R"
                O2 = apply_fermionic_op("c", O, ind_j, side_j)
                
                R[L * (i % 2) + ind_i, L * (j % 2) + ind_j] = (
                    trace_mpo_product(O1, O2, conj_A=True) / norm_O
                )
    
    # Make Hermitian
    R = R + R.conj().T - np.diag(R.diagonal())
    return R


def compute_otoc(
    mpo: 'SymmetricMPO',
    sites: NDArray | None = None
) -> NDArray:
    """
    Compute the out-of-time-order correlator.
    
    For a local operator O(t), computes Tr(O(t) Sz_i O(t) Sz_i) / Tr(O^dag O).
    
    Parameters
    ----------
    mpo : SymmetricMPO
        The time-evolved operator.
    sites : ndarray, optional
        Sites at which to compute OTOC. Default: all sites.
        
    Returns
    -------
    otoc : ndarray
        OTOC values at each site.
    """
    from .mpo import apply_spin_op
    
    O = mpo.copy()
    L = O.L
    otoc = np.zeros(L)
    
    if sites is None:
        sites = np.arange(L)
    
    L_PM, R_PM = site_pacman(O, O)
    
    # Compute normalization
    norm_tensor = tensor_contract(L_PM[1], R_PM[0], ([0, 1], [0, 1]))
    norm = np.sum(norm_tensor.data).real
    
    for i in sites:
        O_i = apply_spin_op("sz", O, i).TN[f"B{i}"].copy()
        
        O_i.leg_type[[1, 2]] = O_i.leg_type[[2, 1]]
        A_tensor = tensor_contract(L_PM[i], O_i, ([0], [0]))
        A_tensor.arrows[[1, 2]] = 'o'
        O_i.leg_type[[1, 2]] = O_i.leg_type[[2, 1]]
        B_tensor = tensor_contract(A_tensor, O_i, ([0, 1, 2], [0, 2, 1]))
        
        result = tensor_contract(B_tensor, R_PM[i], ([0, 1], [0, 1]))
        otoc[i] = np.sum(result.data).real / norm
    
    return otoc
