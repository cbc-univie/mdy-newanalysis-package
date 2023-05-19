# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport fabs, sqrt, floor, pow

@cython.boundscheck(False)
def atomicIndDip(double [:,:] coors, double [:] charges, double [:] alpha,
                 double [:,:] coors_ind, double [:,:] prev_atomic_dipoles,
                 double [:,:] atomic_dipoles, double boxl):

    """
    Calculate the atomic induced dipoles of a selection in an SCF approach
    TESTING version

    coors          ... (N_all, 3) # whole selection, e.g. all water atoms
    charges        ... (N_all)
    alpha          ... (N_ind)
    coors_ind      ... (N_ind, 3) # induced dip selection, e.g. oxygens
    atomic_dipoles ... (N_ind, 3)
    """

    cdef const int var = 3
    cdef int n_particles_all = coors.shape[0]
    cdef int n_particles_ind = atomic_dipoles.shape[0]
    cdef double dist,dist_sq,dist_cub
    cdef double boxl_2 = boxl_2/2.0
    cdef double dist_vec[3]
    cdef double e_0[var][3] 
    cdef double dip_ten[3][3]

#    # compute distances between selections and the resulting electric field
#    for atom1 in prange(n_particles_ind,nogil=True):
#        for atom2 in range(n_particles_all):
#            dist_sq = 0.0
#            for dim in range(3):
#                dist = coors_ind[atom1,dim] - coors[atom2,dim]
#                dist_sq += dist*dist
#
#            #if dist_sq == 0: # NOTE: if e_0 is not initialized with np.zeros,
#            #    continue     # you can't skip them even if the dist is 0
#
#            dist_cub = dist_sq**(-3/2)
#            for dim in range(3):
#                e_0[atom1,dim] = charges[atom1] * charges[atom2] * (
#                coors_ind[atom2,dim] - coors[atom1,dim]) * dist_cub
#
#    # for the case that no initial previous induced dipoles are present
#    if (atomic_dipoles[0,0] == 0 and atomic_dipoles[0,1] == 0 and
#    atomic_dipole[0,2] == 0):
#        for atom1 in prange(n_particles_ind,nogil=True):
#            for dim in range(3):
#                atomic_dipoles[atom1,dim] = alpha[atom1] * e_0[atom1,dim]
#
#    # Compute the induced electric field iteratively
#    # 4 cycles are usually enough
#    for it in range(4):
#        for atom1 in prange(n_particles_ind,nogil=True):
#            for dim in range(3):
#                atomic_dipoles[atom1,dim] = alpha[atom1] * e_0[atom1,dim]
#
#            #for atom2 in range(atom1+1, n_particles_ind):
#            for atom2 in range(n_particles_ind):
#                if atom1 == atom2:
#                    continue
#
#                dist_sq = 0.0
#                for dim in range(3):
#                    dist_vec[dim] = coors_ind[atom2,dim] - coors_ind[atom1,dim]
#                    if dist_vec[dim] > boxl_2:
#                        dist_vec[dim] -= boxl
#                    if dist_vec[dim] < -boxl_2:
#                        dist_vec[dim] += boxl
#                    dist_sq  += dist_vec[dim]**2
#                dist_cub = dist_sq**(-3/2)
#
#                for i in range(3):
#                    for j in range(3):
#                        dip_ten[i, j] = dist_vec[i] * dist_vec[j] / dist_sq
#                        if i == j:
#                            dip_ten[i, j] -= 1
#                        dip_ten[i, j] *= dist_cub
#
#                for dim in range(3):
#                    atomic_dipoles[atom1, dim] += alpha[atom1] * (
#                    dip_ten[dim, 0] * prev_atomic_dipoles[atom2, dim] +\
#                    dip_ten[dim, 1] * prev_atomic_dipoles[atom2, dim] +\
#                    dip_ten[dim, 2] * prev_atomic_dipoles[atom2, dim])
#
#        #TODO: Check convergence; if so, break, otherwise, continue
#        for atom1 in range(n_particles_ind):
#            for dim in range(3):
#                prev_atomic_dipoles[atom1, dim] = atomic_dipoles[atom1, dim]
            