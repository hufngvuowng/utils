"""Utilities for working with molecular Hamiltonian from ph-AFQMC.
"""

import numpy as np
from scipy.io import FortranFile
from scipy.linalg import svd
from numpy import ndarray

def idx2d_to_utrp(col_idx : int, row_idx : int, nrows : int):
    """Convert the (row_idx, col_idx) of an elt in a 2D symmetric matrix to its index in
    the 1D, upper triangular packed form of that matrix. All indices are 0-based.

    Parameters
    ---------
    col_idx : int
        The column index.
    row_idx : int
        The row index.
    nrows : number of rows.

    Returns
    -------
    packed_idx : int
        The index of the corresponding elt in the upper triangular form.
    """
    flat_dim = int(nrows*(nrows + 1)/2)
    packed_dim = flat_dim - int((nrows - row_idx)*(nrows - row_idx + 1)/2) - row_idx + col_idx
    return packed_dim

def utptsym(packed_mat, dim : int):
    """Convert a (real) symmetric matrix of size (dim, dim) stored in the upper triangular 
    packed format to the original symmetric matrix.

    Parameters
    ----------
    packed_mat : numpy.ndarray
        Matrix of size (1, packed_dim) storing the symmetric matrix in the upper, triangular
        packed format.
    dim : int
        The dimension of the original square symmetric matrix.

    Returns
    -------
    symm_mat : numpy.ndarray
        The original symmetric matrix.    
    """
    assert(dim > 0)
    symm_mat = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i, dim):
            if i >= j:
                packed_idx = idx2d_to_utrp(j, i, dim)
                symm_mat[i, j] = packed_mat[packed_idx]
            else:
                packed_idx = idx2d_to_utrp(i, j, packed_idx)
                symm_mat[i, j] = packed_mat[packed_idx]
    
    return symm_mat

def load_cholesky_file(chol_file : str):
    """Parse the Cholesky matrices, stored in symmetric real format to the fulle symmetric
    matrices.

    Parameters
    ----------
    chol_file : str
        The path to where the Cholesky file is stored.

    Returns
    -------
    Lmat_all : numpy.ndarray
        Array of size (mat_dim, mat_dim, n_chol), where mat_dim is the size of the
        individual matrix and n_chol is the number of Cholesky matrices.    
    """
    chol_mat = FortranFile(chol_file) 
    _ = chol_mat.read_ints(dtype=np.int32)[0]
    packed_dim, n_chol = chol_mat.read_record(dtype=np.int32)
    mat_dim = int(np.sqrt(packed_dim))
    Lmat_all = np.zeros((mat_dim, mat_dim, n_chol))
    for chol_idx in np.arange(n_chol):
        Lmat = chol_mat.read_record(dtype=float)
        Lmat_all[:, :, chol_idx] = utptsym(Lmat, mat_dim)
    return Lmat_all

def compress_eris_tensor(Lmat_list : ndarray, cmat : ndarray, ncas : int, svd_thresh=1e-5):
    """Compute the 4-index tensor Y_{ijrs} = sum_{gamma} L^{gamma}_{ir}L^{gamma}_js and
    and compute singular value decomposition, and write the compressed matrices out to 
    files. Note that all these system parameters have to have the number of frozen
    orbitals subtracted from them

    Parameters
    ----------
    Lmat_list : numpy.ndarray
        Arrays of size [M, M, X], where M is the size of basis set and X is the number of
        Cholesky matrices, in AO basis.
    cmat : numpy.ndarray
        Matrix of size [M, M] containing the MO matrix coefficients.
    ncas : int
        N_cas in SVD paper. This is the number of inactive and the active orbitals.
    svd_thresh : float, optional.
        The threshold to truncate the SVD decomposition. For each pair [i, j], we only
        keep the first K_{ij} SVD larger than `svd_thresh`. Default 1e-5.

    Returns
    -------
    0
    """
    nbas = Lmat_list.shape[0]
    nchol = Lmat_list.shape[-1]
    assert(cmat.shape[0] == nbas)

    hr_list = None
    Y_tensor = np.zeros(nbas, nbas, int(ncas*(ncas+1)/2))
    svd_dim_list = []

    # Compute the Y tenstor 
    # Lbar^{gamma}_{ir} = Cmat x L^{gamma} (ncas, nbas)
    # Y^{[ij]}_{rs} = sum_{gamma}L^{\gamma}_{ir}L^{\gamma}_{js}
    for i in np.arange(0, ncas):
        for j in np.arange(i, ncas):
            comp_idx = idx2d_to_utrp(j, i, ncas)
            for chol_idx in np.arange(nchol):
                Lmat = Lmat_list
                # (M x 1) x (1 x M)
                Li_vec = Lmat[i, :, chol_idx].reshape((nbas, 1)) # M x 1
                Lj_vec = Lmat[j, :, chol_idx].reshape((1, nbas)) # 1 x M
            Y_tensor[:, :, comp_idx] += np.matmul(Li_vec, Lj_vec)
        U, s, VT = svd(Y_tensor[:, :, i, j], lapack_driver="gesvd")
        # s should have been sorted in decreasing order
        # so sig_svd_idx = [0, 1, ..., K], where K is the number of signicant singular
        # values
        sig_svd_idx = np.where(s >= svd_thresh)[0]
        svd_dim = len(sig_svd_idx)
        svd_dim_list.append(svd_dim)
        #s_sqrt = np.sqrt(np.diag(s)[:svd_dim, :svd_dim])
        #Ubar = np.matmul(U[:, :svd_dim], s_sqrt)
        #Vbar = np.transpose(np.matmul(s_sqrt, VT[:svd_dim, :]))

        # Write to file
        #write_fortran_file(f"ERI_L_{comp_idx}.mat", Ubar, 0, 1)
        #write_fortran_file(f"ERI_R_{comp_idx}.mat", Vbar, 0, 1)
    # Done, now write the dimension
    #dim_file = FortranFile("Dimensions", "w")
    #npairs = int(ncas*(ncas+1)/2)
    #dim_file.write_record(npairs)
    #dim_file.write_record(svd_dim_list)
      
    avg_svd = np.average(svd_dim_list)
    dim_saved = avg_svd/nbas*100
    print(f"Average SVD dim <K> = {avg_svd:.2f}, saving {dim_saved:.2f}%.")
    np.savetxt("svd_dim.csv", svd_dim_list)

    return svd_dim_list