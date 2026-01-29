"""
Givens rotations for orbital localization in tensor networks.

This module implements Givens rotation decompositions used to transform
between orbital bases while maintaining the MPS/MPO structure through
local two-site operations.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Literal


def givens_rotations(
    mat: NDArray[np.complexfloating],
    loc: list[int] | NDArray[np.intp],
    sect: NDArray[np.intp] | None = None,
    direction: Literal["left", "right"] = "left"
) -> tuple[NDArray[np.intp], NDArray[np.complexfloating]]:
    """
    Compute Givens rotations to localize orbitals on the MPS.
    
    Given a transfer matrix between current and desired orbital bases,
    computes a sequence of 2x2 Givens rotations that localizes specified
    orbitals to the edge of a region.
    
    Parameters
    ----------
    mat : ndarray, complex
        Transfer matrix where rows are current orbitals and columns
        are desired orbitals.
    loc : array-like of int
        Column indices in `mat` of orbitals to localize.
    sect : ndarray of int, optional
        Site indices that will be affected by the rotation.
        If None, uses all sites.
    direction : {"left", "right"}, default "left"
        Where to localize the orbitals:
        - "left": Stack localized orbitals at the left of sect (+----)
        - "right": Stack localized orbitals at the right of sect (----+)
        
    Returns
    -------
    indices : ndarray, shape (n_gates, 2)
        Sites on which each local Givens rotation is applied.
    givens : ndarray, shape (n_gates, 2, 2), complex
        The 2x2 Givens rotation matrices to apply.
        
    Notes
    -----
    The Givens rotations eliminate vector components one at a time,
    "pushing" the weight to the target edge. For direction="left",
    we sweep from right to left; for direction="right", from left to right.
    
    Examples
    --------
    >>> mat = np.eye(4, dtype=complex)
    >>> indices, givens = givens_rotations(mat, [3], np.arange(4), "right")
    """
    loc = np.atleast_1d(loc)
    
    if sect is None:
        sect = np.arange(mat.shape[0])
    
    # Pre-allocate with estimated capacity (can grow if needed)
    max_gates = len(loc) * (len(sect) - 1)
    givens_list = np.zeros((max_gates, 2, 2), dtype=complex)
    indices_list = np.zeros((max_gates, 2), dtype=np.intp)
    n_gates = 0
    
    M = mat.copy()
    
    if direction == "right":
        loc = loc[::-1]
        
    for n, k in enumerate(loc):
        # Determine sweep direction
        if direction == "right":
            reduction = np.arange(sect[0], sect[-1] - n)
        else:
            reduction = np.arange(sect[-1] - 1, sect[0] + n - 1, -1)
            
        layer_indices = []
        layer_givens = []
        v = M[:, k].copy()
        
        for j in reduction:
            q, p = v[j], v[j + 1]
            norm = np.sqrt(np.abs(p)**2 + np.abs(q)**2)
            
            if norm < 1e-14:
                g = np.eye(2, dtype=complex)
            else:
                if direction == "left":
                    # Eliminate component at j (move weight down)
                    g = np.array([
                        [q.conj(), p.conj()],
                        [-p, q]
                    ], dtype=complex) / norm
                    v[j], v[j + 1] = norm, 0
                else:
                    # Eliminate component at j (move weight up)
                    g = np.array([
                        [p, -q.conj()],
                        [q, p.conj()]
                    ], dtype=complex) / norm
                    v[j], v[j + 1] = 0, norm
                    
            layer_indices.append([j, j + 1])
            layer_givens.append(g)
        
        # Update basis for accumulated Givens rotations
        if layer_indices:
            rot = rotation_from_givens(layer_indices, layer_givens, sect)
            M = rot @ M
            
            # Add to output arrays
            n_new = len(layer_indices)
            if n_gates + n_new > len(givens_list):
                # Expand arrays if needed
                givens_list = np.concatenate([
                    givens_list, 
                    np.zeros((max_gates, 2, 2), dtype=complex)
                ])
                indices_list = np.concatenate([
                    indices_list,
                    np.zeros((max_gates, 2), dtype=np.intp)
                ])
                
            indices_list[n_gates:n_gates + n_new] = layer_indices
            givens_list[n_gates:n_gates + n_new] = layer_givens
            n_gates += n_new
    
    return indices_list[:n_gates], givens_list[:n_gates]


def rotation_from_givens(
    indices: list[list[int]] | NDArray[np.intp],
    givens: list[NDArray] | NDArray[np.complexfloating],
    sect: NDArray[np.intp]
) -> NDArray[np.complexfloating]:
    """
    Compute the full rotation matrix from a sequence of Givens rotations.
    
    After applying Givens rotations to localize one orbital, the other
    orbitals in the subspace are also affected. This function computes
    the combined unitary transformation on the subspace.
    
    Parameters
    ----------
    indices : array-like, shape (n_gates, 2)
        Site pairs (i, i+1) where each gate is applied.
    givens : array-like, shape (n_gates, 2, 2)
        The 2x2 Givens rotation matrices.
    sect : ndarray
        Site indices defining the subspace. The dimension of the
        rotation matrix equals len(sect).
        
    Returns
    -------
    rot : ndarray, shape (len(sect), len(sect)), complex
        The combined rotation matrix on the subspace.
        
    Notes
    -----
    The rotation is computed by applying the Givens rotations in reverse
    order compared to their application on the MPS, as we're computing
    the transformation on the orbital coefficients rather than the state.
    """
    dim = len(sect)
    rot = np.eye(dim, dtype=complex)
    
    # Apply in reverse order
    for ind, g in zip(reversed(indices), reversed(givens)):
        rot[:, ind] = rot[:, ind] @ g
    
    return rot


def decompose_unitary_givens(
    U: NDArray[np.complexfloating],
    direction: Literal["left", "right"] = "left"
) -> tuple[NDArray[np.intp], NDArray[np.complexfloating]]:
    """
    Decompose an arbitrary unitary into Givens rotations.
    
    This is useful for implementing general orbital rotations as
    sequences of nearest-neighbor gates on the tensor network.
    
    Parameters
    ----------
    U : ndarray, shape (n, n), complex
        Unitary matrix to decompose.
    direction : {"left", "right"}, default "left"
        Direction of the Givens sweep.
        
    Returns
    -------
    indices : ndarray, shape (n_gates, 2)
        Site pairs for each Givens rotation.
    givens : ndarray, shape (n_gates, 2, 2), complex
        The 2x2 Givens rotation matrices.
        
    Notes
    -----
    Uses the column-by-column elimination approach, reducing U to
    the identity through Givens rotations.
    """
    n = U.shape[0]
    loc = list(range(n - 1, -1, -1)) if direction == "left" else list(range(n))
    sect = np.arange(n)
    
    return givens_rotations(U, loc, sect, direction)
