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

def givens_rotations(mat: np.ndarray, loc: list[int], 
                    sect: NDArray[np.intp] | None = None, 
                    direction: str = "left") -> tuple[NDArray[np.intp], NDArray[np.complexfloating]]:
    """
    Compute Givens rotations to localize orbitals.
    
    Args:
        mat: Transfer matrix
        loc: Columns corresponding to orbitals to localize
        sect: Orbitals affected by rotation
        direction: "left" or "right" for localization direction
        
    Returns:
        indices: Site pairs for Givens rotations
        givens: Givens rotation matrices (2x2)
    """
    givens_list = []
    indices_list = []
    
    M = mat.copy()
    if direction == "right":
        loc = loc[::-1]
    
    for n, k in enumerate(loc):
        if direction == "right":
            reduction = np.arange(sect[0], sect[-1] - n)
        else:
            reduction = np.arange(sect[-1] - 1, sect[0] + n - 1, -1)
        
        i_n, g_n = [], []
        v = M[:, k].copy()
        
        for j in reduction:
            q, p = v[j], v[j + 1]
            norm = np.sqrt(np.abs(p)**2 + np.abs(q)**2)
            
            if norm < 1e-14:
                g = np.eye(2, dtype=complex)
            else:
                if direction == "left":
                    g = np.array([[q.conj(), p.conj()], [-p, q]], dtype=complex) / norm
                    v[j], v[j + 1] = norm, 0
                else:
                    g = np.array([[p, -q.conj()], [q, p.conj()]], dtype=complex) / norm ## Careful where is the minus with .T or not on the MPO ! 
                    v[j], v[j + 1] = 0, norm
            
            i_n.append([j, j + 1])
            g_n.append(g)
        
        if i_n:
            indices_list.extend(i_n)
            givens_list.extend(g_n)
    
    return np.array(indices_list), np.array(givens_list)



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
