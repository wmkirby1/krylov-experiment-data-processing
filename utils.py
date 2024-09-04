"""
(C) Copyright 2024 IBM

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.

This code is associated to the paper "Diagonalization of large many-body Hamiltonians on a quantum processor" https://arxiv.org/abs/2407.14431.
"""


import numpy as np
import scipy as sp


def get_partial_matrix(mat, args):
    return np.copy(mat)[:, args][args, :]


####################
### REGULARIZERS ###
####################

def solve_regularized_gen_eig(h, s, k=1, threshold=1e-15, return_dimn=False, return_vecs=False):

    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = [vec for val, vec in zip(s_vals, s_vecs) if val > threshold]
    if not good_vecs:
        raise AssertionError('WHOLE SUBSPACE ILL-CONDITIONED')
    good_dimn = len(good_vecs)
    good_vecs = np.array(good_vecs).T

    h_reg = good_vecs.conj().T @ h @ good_vecs
    s_reg = good_vecs.conj().T @ s @ good_vecs
    vals, vecs = sp.linalg.eigh(h_reg, s_reg)

    if k==1:
        if return_dimn and return_vecs:
            return vals[0], good_vecs @ vecs[:,0], good_dimn
        elif return_dimn:
            return vals[0], good_dimn
        elif return_vecs:
            return vals[0], good_vecs @ vecs[:,0]
        else:
            return vals[0]
    else:
        if return_dimn and return_vecs:
            return vals[0:k], good_vecs @ vecs[:,0:k], good_dimn
        elif return_dimn:
            return vals[0:k], good_dimn
        elif return_vecs:
            return vals[0:k], good_vecs @ vecs[:,0:k]
        else:
            return vals[0:k]
    

'''
optimize_threshold starts from a small initial threshold guess and performs a logarithmic search for the optimal threshold,
with the criterion for success being a sufficiently good fit of the energy curve to an exponential decay (expected from theory).

Parameters:
D = max Krylov dimension
H_est = Krylov H matrix
S_est = Krylov S matrix
energy_lower = lower bound on ground state energy (need not be tight)
H_norm = upper bound on full Hamiltonian spectral norm
init_threshold = initial guess at threshold (just needs to be a guaranteed underestimate)
fit_tol = required exponential decay fit accuracy
threshold_tol_ratio = search grid for optimal threshold, as ratio of threshold
skip_D = number of low dimensions to skip (dimension 1 always included, so e.g. skip_D=1 means dimension 2 will be skipped)
dimn_scaling = scaling of threshold with dimension, given as a power of the dimension (should be between 0 and 1 most likely)

'''

def optimize_threshold( D, H_est, S_est, energy_lower, init_threshold=1e-8, fit_tol=0.5, threshold_tol_ratio=0.1, skip_D=0, dimn_scaling=1, return_vecs=False):

    init_energy = np.real(H_est[0,0])
    f = lambda x, a, b: (init_energy-a)*np.exp(-b*(x-1)) + a

    going_up = True
    converged = False
    threshold = init_threshold
    while not converged:
        gs_en_estimates = []
        for _d in range(1, D+1):
            h = get_partial_matrix(H_est, range(_d))
            s = get_partial_matrix(S_est, range(_d))
            
            gs_en_estimates.append(solve_regularized_gen_eig(h, s, k=1, threshold=(_d**dimn_scaling)*threshold))

        copt, ccov = sp.optimize.curve_fit(f, [1]+list(range(skip_D+2,D+1)), [gs_en_estimates[0]]+list(gs_en_estimates[skip_D+1:]), bounds=[[energy_lower, 0], [init_energy, np.infty]])

        fit_std_err = np.sqrt(sum((f(i,*copt)-gs_en_estimates[i-1])**2 for i in [1]+list(range(skip_D+2,D+1)))/(D-skip_D))
        
        if fit_std_err > fit_tol and going_up:
            threshold *= 1.2
        elif fit_std_err <= fit_tol and going_up:
            going_up = False
            threshold *= 1-threshold_tol_ratio
        elif fit_std_err <= fit_tol and not going_up:
            if threshold > init_threshold:
                threshold *= 1-threshold_tol_ratio
            else:
                print('init_threshold too high')
                threshold /= 1-threshold_tol_ratio
                converged = True
        elif fit_std_err > fit_tol and not going_up:
            threshold /= 1-threshold_tol_ratio
            converged = True

    gs_en_estimates = []
    gs_vec_estimates = []
    for _d in range(1, D+1):
        h = get_partial_matrix(H_est, range(_d))
        s = get_partial_matrix(S_est, range(_d))

        if return_vecs:
            val, vec = solve_regularized_gen_eig(h, s, k=1, threshold=(_d**dimn_scaling)*threshold, return_vecs=True)
            gs_vec_estimates.append(vec)
        else:
            val = solve_regularized_gen_eig(h, s, k=1, threshold=(_d**dimn_scaling)*threshold)

        gs_en_estimates.append(val)

    copt, ccov = sp.optimize.curve_fit(f, list(range(skip_D+1,D+1)), gs_en_estimates[skip_D:], bounds=[[energy_lower, 0], [init_energy, np.infty]])

    if return_vecs:
        return threshold, gs_en_estimates, copt, gs_vec_estimates
    else:
        return threshold, gs_en_estimates, copt