# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#TODO: Maybe sort these functions into thematically fitting function libraries

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport sqrt, floor, pow

cdef extern from "miscellaneous_implementation.h":
    void test_parallelism(int iterations)
    void atomic_ind_dip(double * coors, double * charges, double * alpha, double * coors_ind, double * prev_atomic_dipoles, double * atomic_dipoles, int n_particles_all, int n_particles_ind, double boxlength)
    void atomic_ind_dip_per_atom(int atom1, double * coors, double * charges, double * alpha, double * coors_ind, double * prev_atomic_dipoles, double * atomic_dipoles, int n_particles_all, int n_particles_ind, double boxlength)
    void derive_ind_dip(double * coors_ind, double * vel_ind, double * atomic_dipoles, double * derived_atomic_dipoles, int n_particles_ind, double boxlength)
    void derive_ind_dip_per_atom(int atom1, double * coors_ind, double * vel_ind, double * atomic_dipoles, double * derived_atomic_dipoles, int n_particles_ind, double boxlength)

    void calc_accumulate_shellwise_g_function(double * aufpunkte, double * coor, char * dataset, double * histogram, double * norm, int n_aufpunkte, int n_particles, int number_of_datapoints, int maxshell, double histo_min, double histo_max, double boxlength)
    int get_best_index(double * aufpunkt, double * coor, int natoms)
    void write_Mr_diagram(double * coor, double * dip, int n_particles, double * aufpunkt, double * antagonist, double * histogram, double max_distance, int segments_per_angstroem, int order)
    void write_Kirkwood_diagram(double * aufpunkt, double * dip_aufpunkt, double * coor, double * dip, int natom, double * max_dist, double, int segs, int order)
    void write_Kirkwood_diagram_shellwise(double * coor_aufpunkt, double * dip_aufpunkt, double * coor, double * dip, int n_particles, double * histogram, char * dataset, int maxshell, double max_distance, int segments_per_angstroem, int order)
    void write_Kirkwood_diagram_2D(double * aufpunkt, double * dip_aufpunkt, double * coor, double * dip, int natom, double * max_dist, double, int segs, int cossegs, int order)

    void calc_donor_grid(double * coor, int * bond_table, int donor_count, double cutoff)

    void separateCollectiveDipolesSpherically(double * coor, double * dipoles, int n_particles, double * aufpunkt, double * dip_inside, double * dip_outside, double cutoff)
    void calc_sum_Venn_MD_Cage_Single(double * mdcage_timeseries, double * dipoles, int n_particles, char * dataset1, char * dataset2, int maxshell1, int maxshell2)

    void correlateSingleVectorTS(double * timeseries, double * result, int correlation_length, int order)
    void crossCorrelateSingleVectorTS(double * timeseries1, double * timeseries2, double * result, int correlation_length1, int correlation_length2, int both_directions, int order)
    void correlateMultiVectorTS(double * timeseries, double * result, int number_of_particles, int correlation_length, int order)
    void correlateMultiVectorShellwiseTS(double * timeseries, double * dataset, double * result, int number_of_particles, int correlation_length, int maxshell, int order)
    void correlateMultiVectorVennShellwiseTS(double * timeseries, double * dataset1,  double * dataset2, double * result, int number_of_particles, int correlation_length, int maxshell1, int maxshell2, int order)
    void crossCorrelateMultiVectorTS(double * timeseries1, double * timeseries2, double * result, int number_of_particles, int correlation_length1, int correlation_length2, int both_directions, int order)

    void calcVanHoveSingleVector(double * timeseries, double * histogram, int correlation_length, int cos_segs)
    void calcVanHoveMultiVector(double * timeseries, double * histogram, int n_particles, int correlation_length, int cos_segs)

    void calc_distance_delauny_mindist(int * delauny_matrix, double * coor, int n_particles, double boxlength, int number_of_shells, double bin_width)
    void calc_distance_delauny(int * delauny_matrix, double * coor, int n_particles, int number_of_shells, double bin_width)
    void calc_min_dist_tesselation(double * dataset, double * coor_core, double * coor_surround, int n_core, int n_surround, double binwidth)

    void sort_collective_dip_NN_shells(char * ds, double * dip_wat, double * dip_shell, int n_particles)
    void sort_collective_dip_NN_shells_int(int * ds, double * dip_wat, double * dip_shell, int n_particles)

    void calc_dip_ten_collective(double * coor, int n_particles, double * results)
    void calc_dip_ten_collective_per_atom(int i, double * coor, int n_particles, double * results)
    void calc_dip_ten_collective_cross(double * coor1, int n_particles1, double * coor2, int n_particles2, double * results)
    void calc_dip_ten_collective_NNshellwise(double * coor, int n_particles, char * ds, int maxshell, double * results)
    void calc_dip_ten_collective_NNshellwise_self(double * coor, int * f2c, int n_particles, int n_particles_tot, char * ds, int ds_idx, int maxshell, double * results)
    void calc_dip_ten_collective_NNshellwise_cross(double * coor1, double * coor2, int * f2c, int n_particles_1, int n_particles_2, int n_particles_tot, char * ds, int ds1_idx, int ds2_idx, int maxshell, double * results)
    void calc_dip_ten_collective_1Nshellwise_self(double * coor, int n_particles, char * ds, int maxshell, double * results)
    void calc_dip_ten_collective_1Nshellwise_cross(double * coor1, int n_particles1, double * coor2, int n_particles2, char * ds, int maxshell, double * results)
    void calc_dip_ten_collective_shellwise(double * coor, int n_particles, char * ds, int maxshell, double * results)
    void calc_dip_ten_collective_vennshellwise(double * coor, int n_particles, char * ds1, char * ds2, int maxshell1, int maxshell2, double * results)

    void construct_relay_matrix(double * coor, double * inv_atom_polarizabilities, double * matrix, int n_atoms)

    void pairiter_loop_(int pairlist_len, int * pairlist_p1, int * pairlist_p2, double * correlation_list, int n_pairs_h, int apr_emim_h, int * emim_h, int emim_h_len, double * run, double * dipt_0, double * dipt_t, int * bins, int max_distance)
    void dipten_double_loop_(double * coor_1, double * coor_2, double * dipt_t, int n_particles_1, int n_particles_2, int only_different_nuclei)
    void dipten_double_loop2_(double * coor_1, double * coor_2, double * dipt_t, int n_particles_1, int n_particles_2, int only_different_nuclei)

@cython.boundscheck(False)
def testParallelism(int n_iterations):
    test_parallelism(n_iterations)
    return

@cython.boundscheck(False)
def countHBonds(double [:,:] coor_surr, double [:,:] coor_oh2, int nres_surr, double maxdist, double cos_angle=-0.95):
    # this function is designed only for water hydrogen bonds!
    cdef int sites_per_res = coor_surr.shape[0] / nres_surr
    cdef int nsurr = coor_surr.shape[0] / sites_per_res
    cdef int nwat = coor_oh2.shape[0] / 3
    cdef int i, j, k, l, idx, idx2, idx3
    
    cdef int [:] hbond = np.zeros(nwat, dtype=np.int32)

    cdef double dx, dy, dz, dx2, dy2, dz2, dot, dot2, dist, dist2, cosine

    # loop over water molecules
    for i in prange(nwat, nogil=True):
        # loop over h atoms
        for j in range(2):
            idx = i*3+1+j
            idx3 = i*3
            # loop over surrounding molecules
            for k in range(nsurr):
                # loop over oxygen atoms
                for l in range(sites_per_res):
                    idx2 = k*sites_per_res+l
                    dx = coor_surr[idx2,0] - coor_oh2[idx,0]
                    dy = coor_surr[idx2,1] - coor_oh2[idx,1]
                    dz = coor_surr[idx2,2] - coor_oh2[idx,2]
                    dot = dx*dx + dy*dy + dz*dz
                    dist = sqrt(dot)
                    if dist < maxdist:
                        dx2 = coor_oh2[idx3,0] - coor_oh2[idx,0]
                        dy2 = coor_oh2[idx3,1] - coor_oh2[idx,1]
                        dz2 = coor_oh2[idx3,2] - coor_oh2[idx,2]
                        dot2 = dx2*dx2 + dy2*dy2 + dz2*dz2
                        dist2 = sqrt(dot2)
                        cosine = (dx * dx2 + dy * dy2 + dz * dz2) / (dist * dist2)
                        if cosine < cos_angle:
                            hbond[i] += 1

    return np.asarray(hbond)

@cython.boundscheck(False)
def structureFactorDipTen(double [:,:] coors, double [:,:] histogram, int bin_dist, int segs, double boxlength):
    cdef int n_particles = coors.shape[0]
    cdef int pair_1_par_1 = 0
    cdef int pair_1_par_2 = 0
    cdef int pair_2_par_1 = 0
    cdef int pair_2_par_2 = 0
    cdef double D_x, D_y, D_z
    cdef double dist1, dist2
    cdef int bin1, bin2
    cdef double blen2 = boxlength / 2.0

    for pair_1_par_1 in range(0, n_particles-1):
        for pair_1_par_2 in range(pair_1_par_1+1, n_particles):
            D_x = coors[pair_1_par_1, 0] - coors[pair_1_par_2, 0]
            if(D_x > blen2)   : D_x -= boxlength
            elif(D_x < -blen2) : D_x += boxlength

            D_y = coors[pair_1_par_1, 1] - coors[pair_1_par_2, 1]
            if(D_y > blen2)   : D_y -= boxlength
            elif(D_y < -blen2) : D_y += boxlength

            D_z = coors[pair_1_par_1, 2] - coors[pair_1_par_2, 2]
            if(D_z > blen2)   : D_z -= boxlength
            elif(D_z < -blen2) : D_z += boxlength

            dist1 = np.sqrt(D_x*D_x + D_y*D_y + D_z*D_z)

            for pair_2_par_1 in range(0, n_particles-1):
                for pair_2_par_2 in range(pair_2_par_1+1, n_particles):
                    if (pair_1_par_1 == pair_2_par_1) and (pair_1_par_2 == pair_2_par_2): continue

                    D_x = coors[pair_2_par_1, 0] - coors[pair_2_par_2, 0]
                    if(D_x > blen2)   : D_x -= boxlength
                    elif(D_x < -blen2) : D_x += boxlength

                    D_y = coors[pair_2_par_1, 1] - coors[pair_2_par_2, 1]
                    if(D_y > blen2)   : D_y -= boxlength
                    elif(D_y < -blen2) : D_y += boxlength

                    D_z = coors[pair_2_par_1, 2] - coors[pair_2_par_2, 2]
                    if(D_z > blen2)   : D_z -= boxlength
                    elif(D_z < -blen2) : D_z += boxlength

                    dist2 = np.sqrt(D_x*D_x + D_y*D_y + D_z*D_z)

                    bin1 = int(segs * dist1)
                    bin2 = int(segs * dist2)

                    if (bin1 < bin_dist) and (bin2 < bin_dist):
                        histogram[bin1, bin2] += (pow(dist1, -3) * pow(dist2, -3))
                        histogram[bin2, bin2] += histogram[bin1, bin2]


@cython.boundscheck(False)
def structureFactorCharge(double [:,:] coors, double [:] charges, double [:,:] histogram, int bin_dist, int segs, double boxlength):
    cdef int n_particles = coors.shape[0]
    cdef int pair_1_par_1 = 0
    cdef int pair_1_par_2 = 0
    cdef int pair_2_par_1 = 0
    cdef int pair_2_par_2 = 0
    cdef double D_x, D_y, D_z
    cdef double dist1, dist2
    cdef double energy1, energy2
    cdef int bin1, bin2
    cdef double blen2 = boxlength / 2.0

    for pair_1_par_1 in range(0, n_particles-1):
        for pair_1_par_2 in range(pair_1_par_1+1, n_particles):
            D_x = coors[pair_1_par_1, 0] - coors[pair_1_par_2, 0]
            if(D_x > blen2)   : D_x -= boxlength
            elif(D_x < -blen2) : D_x += boxlength

            D_y = coors[pair_1_par_1, 1] - coors[pair_1_par_2, 1]
            if(D_y > blen2)   : D_y -= boxlength
            elif(D_y < -blen2) : D_y += boxlength

            D_z = coors[pair_1_par_1, 2] - coors[pair_1_par_2, 2]
            if(D_z > blen2)   : D_z -= boxlength
            elif(D_z < -blen2) : D_z += boxlength

            dist1 = np.sqrt(D_x*D_x + D_y*D_y + D_z*D_z)
            energy1 = charges[pair_1_par_1] * charges[pair_1_par_2] / dist1

            for pair_2_par_1 in range(0, n_particles-1):
                for pair_2_par_2 in range(pair_2_par_1+1, n_particles):
                    if (pair_1_par_1 == pair_2_par_1) and (pair_1_par_2 == pair_2_par_2): continue

                    D_x = coors[pair_2_par_1, 0] - coors[pair_2_par_2, 0]
                    if(D_x > blen2)   : D_x -= boxlength
                    elif(D_x < -blen2) : D_x += boxlength

                    D_y = coors[pair_2_par_1, 1] - coors[pair_2_par_2, 1]
                    if(D_y > blen2)   : D_y -= boxlength
                    elif(D_y < -blen2) : D_y += boxlength

                    D_z = coors[pair_2_par_1, 2] - coors[pair_2_par_2, 2]
                    if(D_z > blen2)   : D_z -= boxlength
                    elif(D_z < -blen2) : D_z += boxlength

                    dist2 = np.sqrt(D_x*D_x + D_y*D_y + D_z*D_z)
                    energy2 = charges[pair_2_par_1] * charges[pair_2_par_2] / dist1

                    bin1 = int(segs * dist1)
                    bin2 = int(segs * dist2)

                    if (bin1 < bin_dist) and (bin2 < bin_dist):
                        histogram[bin1, bin2] += energy1 * energy2
                        histogram[bin2, bin2] += histogram[bin1, bin2]


#Version1: Just one distance per targeted (water) molecule
@cython.boundscheck(False)
def accumulateShellwiseGFunction(double [:,:] aufpunkte, double [:,:] coor, char [:] dataset, double[:,:] histogram, double[:] norm, int maxshell, double histo_min, double histo_max, double boxlength):
    '''
    aufpunkte   ... (n_aufpunkte, 3)
    coor        ... (n_particles, 3)
    dataset     ... (n_particles)
    histogram   ... (number_of_shells, number_of_datapoints)
    norm        ... (number_of_shells)
    '''

    cdef int n_aufpunkte = aufpunkte.shape[0]
    cdef int n_particles = coor.shape[0]
    cdef int number_of_datapoints = histogram.shape[1]

    calc_accumulate_shellwise_g_function(&aufpunkte[0,0], &coor[0,0], &dataset[0], &histogram[0,0], &norm[0], n_aufpunkte, n_particles, number_of_datapoints, maxshell, histo_min, histo_max, boxlength)

@cython.boundscheck(False)
def accumulateNMShellwiseGFunction(double [:,:] coor1, double [:,:] coor2, int[:,:] dataset, double[:,:] histogram, int particle_first, int n_particles, int max_distance, int segments_per_angstroem):
    cdef int particle1, particle2, point1, point2, shell, bin_dist
    cdef int maxshell = histogram.shape[1]
    cdef double dist

    for particle1 in range(n_particles):
        for particle2 in range(particle1+1, n_particles):
            point1 = particle_first + particle1
            point2 = particle_first + particle2

            shell = dataset[point1, point2]
            if(shell >= maxshell): shell = maxshell-1
            dist = 0
            bin_dist = int(dist*segments_per_angstroem)
            if(bin_dist < max_distance*segments_per_angstroem):
                histogram[shell, bin_dist] += 1


def sumUpByShell(double [:,:,:] timeseries, double [:,:,:] timeseries_shellwise, char[:,:] ds):
    """
    timeseries           ... (n_frames, n_particles, 3)
    timeseries_shellwise ... (n_shells, n_frames,    3)
    ds                   ... (n_frames, n_particles)
    """
    #BeyondShellPolicy = drop_particle

    cdef int n_frames    = timeseries.shape[0]
    cdef int n_particles = timeseries.shape[1]
    cdef int n_shells    = timeseries_shellwise.shape[0]
    cdef int particle, shell, frame

    for frame in range(n_frames):
        for particle in range(n_particles):
            shell = ds[frame, particle] - 1
            if shell < n_shells:
                timeseries_shellwise[shell, frame, 0] += timeseries[frame, particle, 0]
                timeseries_shellwise[shell, frame, 1] += timeseries[frame, particle, 1]
                timeseries_shellwise[shell, frame, 2] += timeseries[frame, particle, 2]

def countVennShellOccupation(double[:,:] data, int n_particles, char[:] ds1, char[:] ds2):
    '''
    data    ... (n_shells1, n_shells2)
    '''

    cdef int n_shells1 = data.shape[0]
    cdef int n_shells2 = data.shape[1]
    cdef int wat, shell1, shell2

    for wat in range(n_particles):
        shell1 = ds1[wat] - 1
        shell2 = ds2[wat] - 1

        if shell1 >= n_shells1 : shell1 = n_shells1 - 1
        if shell2 >= n_shells2 : shell2 = n_shells2 - 1

        data[shell1, shell2] += 1

def countMultiShell(double[:,:] data, char[:] ds1, char[:] ds2, char[:] ds3, char[:] ds4, char[:] ds5):
    '''
    data    ... (nshells, max_cardinality {==5})
    '''

    cdef int n_shells = data.shape[0]
    cdef int n_particles = ds1.shape[0]
    cdef int shell_01, shell_02, shell_03, shell_04, shell_05, closest_shell, cardinality
    cdef int wat

    for wat in range(n_particles):
        shell_01 = ds1[wat] - 1
        shell_02 = ds2[wat] - 1
        shell_03 = ds3[wat] - 1
        shell_04 = ds4[wat] - 1
        shell_05 = ds5[wat] - 1

        if shell_01 >= n_shells: shell_01 = n_shells-1
        if shell_02 >= n_shells: shell_02 = n_shells-1
        if shell_03 >= n_shells: shell_03 = n_shells-1
        if shell_04 >= n_shells: shell_04 = n_shells-1
        if shell_05 >= n_shells: shell_05 = n_shells-1

        closest_shell = min(shell_01, shell_02, shell_03, shell_04, shell_05)

        cardinality = 0
        if(shell_01 == closest_shell): cardinality += 1
        if(shell_02 == closest_shell): cardinality += 1
        if(shell_03 == closest_shell): cardinality += 1
        if(shell_04 == closest_shell): cardinality += 1
        if(shell_05 == closest_shell): cardinality += 1

        data[closest_shell, cardinality-1] += 1

'''
@cython.boundscheck(False)
def atomicDipoleTS(double [:,:] atomic_dip, double [:,:] vels, double[:,:] timeseries):
    #FIXME: Add extra dimension for time, call this function only once for entire time correlation function?
    """
    atomic_dip ... (N*3) Coordinates of the particles at time t (necessary?)
    vels       ... (N*3) Velocities of the particles at time t
    timeseries ... (N*3) Atomic dipoles of the particles at time t; this is practically the return value
    """

    cdef int n_particles = coors.shape[0]
    cdef int i, j

    for i in range(n_particles):
        pass #TODO: Put programming logic here

    return
'''

@cython.boundscheck(False)
def atomicIndDip(double [:,:] coors, double [:] charges, double [:] alpha, double [:,:] coors_ind, double [:,:] prev_atomic_dipoles, double [:,:] atomic_dipoles, double boxlength):
    """
    coors          ... (N_all, 3)
    charges        ... (N_all)
    alpha          ... (N_ind)
    coors_ind      ... (N_ind, 3)
    atomic_dipoles ... (N_ind, 3)
    """

    cdef int n_particles_all = coors.shape[0]
    cdef int n_particles_ind = atomic_dipoles.shape[0]
#    cdef int atom, dim, atom1, atom2, iteration, i, j
#    cdef double [:,:] e_0 = np.zeros((n_particles_ind, 3), dtype='float64') 
#    cdef double dist_sq, dist_cub
#    cdef double [:] dist_vec = np.zeros(3, dtype='float64') 
#    cdef double [:,:] dip_ten = np.zeros((3,3), dtype='float64') 

    atomic_ind_dip(&coors[0,0], &charges[0], &alpha[0], &coors_ind[0,0], &prev_atomic_dipoles[0,0], &atomic_dipoles[0,0], n_particles_all, n_particles_ind, boxlength)

"""
    else:
        #for atom1 in range(n_particles_all-1):
        #    for atom2 in range(atom1+1, n_particles_all):
        for atom1 in range(n_particles_ind):
            for atom2 in range(n_particles_all):
                #check for dist == 0
                dist_sq  = 0
                for dim in range(3):
                    dist_sq += (coors[atom2, dim] - coors[atom1, dim])**2
                if dist_sq == 0 : continue
                dist_cub = dist_sq**(-1.5)

                for dim in range(3):
                    e_0[atom1, dim] += charges[atom1] * charges[atom2] * (coors[atom2, dim] - coors[atom1, dim]) * dist_cub

        #Case: No previous initial induced dipoles available
        if((atomic_dipoles[0,0] == 0) and (atomic_dipoles[0,1] == 0) and (atomic_dipoles[0,2] == 0)):
            for atom in range(n_particles_ind):
                for dim in range(3):
                    prev_atomic_dipoles[atom, dim] = e_0[atom, dim] #prev_atomic_dipoles[atom, dim] = e_0[atom1, dim]

        for iteration in range(4): #TODO: Put convergence criteria here
        #    for atom1 in range(n_particles_ind-1):
        #        for dim in range(3):
            for atom1 in range(n_particles_ind):
                for dim in range(3):
                    atomic_dipoles[atom1, dim] = alpha[atom1] * e_0[atom1, dim]

                #for atom2 in range(atom1+1, n_particles_ind):
                for atom2 in range(n_particles_ind):
                    if atom1 == atom2: continue

                    dist_sq = 0
                    for dim in range(3):
                        dist_vec[dim] = coors_ind[atom2, dim] - coors_ind[atom1, dim]
                        dist_sq  += dist_vec[dim]**2
                    dist_cub = dist_sq**(-1.5)

                    for i in range(3):
                        for j in range(3):
                            dip_ten[i, j] = dist_vec[i] * dist_vec[j] / dist_sq
                            if(i == j): dip_ten[i, j] -= 1
                            dip_ten[i, j] *= dist_cub

                    for dim in range(3):
                        atomic_dipoles[atom1, dim] += alpha[atom1] * (dip_ten[dim, 0] * prev_atomic_dipoles[atom2, dim] + dip_ten[dim, 1] * prev_atomic_dipoles[atom2, dim] + dip_ten[dim, 2] * prev_atomic_dipoles[atom2, dim])

            #TODO: Check convergence; if so, break, otherwise, continue
            for atom1 in range(n_particles_ind):
                for dim in range(3):
                    prev_atomic_dipoles[atom1, dim] = atomic_dipoles[atom1, dim]
"""

@cython.boundscheck(False)
def atomicIndDipPerAtom(int idx, double [:,:] coors, double [:] charges, double [:] alpha, double [:,:] coors_ind, double [:,:] prev_atomic_dipoles, double [:,:] atomic_dipoles, double boxlength):
    """
    coors          ... (N_all, 3)
    charges        ... (N_all)
    alpha          ... (N_ind)
    coors_ind      ... (N_ind, 3)
    atomic_dipoles ... (N_ind, 3)
    """

    cdef int n_particles_all = coors.shape[0]
    cdef int n_particles_ind = atomic_dipoles.shape[0]
 
    atomic_ind_dip_per_atom(idx, &coors[0,0], &charges[0], &alpha[0], &coors_ind[0,0], &prev_atomic_dipoles[0,0], &atomic_dipoles[0,0], n_particles_all, n_particles_ind, boxlength)

def kronecker(int a, int b) -> int:
    if (a == b): return 1
    return 0


@cython.boundscheck(False)
def deriveIndDip(double [:,:] coors_ind, double [:,:] vel_ind, double [:,:] atomic_dipoles, double [:,:] derived_atomic_dipoles, double boxlength, int new = 0):
    """
    coors_ind              ... (N_ind, 3)
    vel_ind                ... (N_ind, 3)
    atomic_dipoles         ... (N_ind, 3)
    derived_atomic_dipoles ... (N_ind, 3)
    """

    cdef int n_ind_particles = coors_ind.shape[0]
    #cdef double [:,:,:] dip_ten = np.zeros((3,3,3), dtype = 'float64')
    #cdef int atom1, atom2, dim1, dim2, dim3
    #cdef double dist_sq
    #cdef double dist_5, dist_7
    #cdef double [:] dist_vec = np.zeros(3, dtype = 'float64')
    #cdef double [:] t_mu = np.zeros(3, dtype = 'float64')

    derive_ind_dip(&coors_ind[0,0], &vel_ind[0,0], &atomic_dipoles[0,0], &derived_atomic_dipoles[0,0], n_ind_particles, boxlength)
"""
    else:
        for atom1 in range(n_ind_particles):
            for atom2 in range(n_ind_particles):
                if atom1 == atom2: continue

                for dim1 in range(3):
                    dist_vec[dim1] = coors_ind[atom2, dim1] - coors_ind[atom1, dim1]
                dist_sq = dist_vec[0]**2 + dist_vec[1]**2 + dist_vec[2]**2
                dist_5 = dist_sq**(-2.5)
                dist_7 = dist_sq**(-3.5)

                for dim1 in range(3): #alpha
                    for dim2 in range(3): #beta
                        for dim3 in range(3): #gamma
                            dip_ten[dim1, dim2, dim3] = 3 * dist_5 * (kronecker(dim2, dim3)*dist_vec[dim1] + kronecker(dim3, dim1)*dist_vec[dim2] + kronecker(dim1, dim2)*dist_vec[dim3]) - 15 * dist_7 * dist_vec[dim1]*dist_vec[dim2]*dist_vec[dim3]

                for dim3 in range(3): #gamma
                    for dim2 in range(3): #beta
                        t_mu[dim2] = dip_ten[0, dim2, dim3] * atomic_dipoles[atom1, dim2] + dip_ten[1, dim2, dim3] * atomic_dipoles[atom1, dim2] + dip_ten[2, dim2, dim3] * atomic_dipoles[atom1, dim2]
                    for dim1 in range(3): #alpha
                        derived_atomic_dipoles[atom1, dim3] += vel_ind[atom1, dim1] * t_mu[dim1]
"""

'''
@cython.boundscheck(False)
def atomicDipIterateViaDipTen(double [:,:] atomic_dip_init, double [:,:] atomic_dip_final, double [:,:] coor_1, double [:,:] coor_2):
    """
    atomic_dip_init  ... (N,3)
    atomic_dip_final ... (N,3)
    coor_1           ... (N,3)
    coor_2           ... (N,3)
    """
    cdef double [:,:,:] dipTen = np.zeros((n_particles,3,3), dtype='float64')
    cdef int n_particles = coor_1.shape[0]
    cdef int dim1 = 0, dim2 = 0, particle = 0
    cdef double norm

    for particle in range(n_particles):
        norm_sq = ()

        for dim1 in range(3):
            for dim2 in range(3):
                dipTen[particle, dim1, dim2] = 
'''

@cython.boundscheck(False)
def deriveIndDipPerAtom(int idx, double [:,:] coors_ind, double [:,:] vel_ind, double [:,:] atomic_dipoles, double [:,:] derived_atomic_dipoles, double boxlength, int new = 0):
    """
    coors_ind              ... (N_ind, 3)
    vel_ind                ... (N_ind, 3)
    atomic_dipoles         ... (N_ind, 3)
    derived_atomic_dipoles ... (N_ind, 3)
    """

    cdef int n_ind_particles = coors_ind.shape[0]

    derive_ind_dip_per_atom(idx, &coors_ind[0,0], &vel_ind[0,0], &atomic_dipoles[0,0], &derived_atomic_dipoles[0,0], n_ind_particles, boxlength)

###BEGIN DANIEL NOE HELPERS
@cython.boundscheck(False)
def norm(double[:] x):
    cdef double res = np.dot(x,x)**.5
    return res

@cython.boundscheck(False)
def unfold(double[:] ref, double[:] new, double[:] run, double boxl):
    cdef int i
    cdef double D

    for i in range(3):
        D=(new[i]-ref[i])
        if abs(D)>(boxl/2):
            D-=np.sign(D)*boxl
        run[i]+=D
    return run


@cython.boundscheck(False)
def min_dist(double[:] res1, double[:] res2, double boxl):
    cdef int i
    cdef double D
    cdef double [:] vec = np.zeros(3) 

    for i in range(3):
        D=(res2[i]-res1[i])
        if abs(D)>(boxl/2):
            res2[i]-=np.sign(D)*boxl

    vec[0] = res2[0] - res1[0]
    vec[1] = res2[1] - res1[1]
    vec[2] = res2[2] - res1[2]
    return norm(vec)


@cython.boundscheck(False)
def dipTen(double[:] coo1, double [:] coo2): #returns dipole dipole tensor for a distance vector between coo1 and coo2
    cdef double [:] rv = np.zeros(3) #distance vector
    rv[0] = coo2[0] - coo1[0]
    rv[1] = coo2[1] - coo1[1]
    rv[2] = coo2[2] - coo1[2]
    cdef double r2 = np.dot(rv,rv) #norm squared of rv
    cdef double f1 = r2**-1.5      #prefactor 1: 1/r**3
    cdef double f2 = 3/r2          #prefactor 2: 3*1/r**2
    cdef double[:] dipt = np.empty(6,dtype=np.float64) #initialize dipole dipole tensor (xx,yy,zz,xy,xz,yz)

    dipt[0]=f1*(rv[0]*rv[0]*f2-1)     #calculate elements
    dipt[1]=f1*(rv[1]*rv[1]*f2-1)     #minus 1 in definition of dipole dipole tensor
    dipt[2]=f1*(rv[2]*rv[2]*f2-1)
    dipt[3]=f1*(rv[0]*rv[1]*f2)*2     #times 2 because off diag elements appear twice
    dipt[4]=f1*(rv[0]*rv[2]*f2)*2
    dipt[5]=f1*(rv[1]*rv[2]*f2)*2
    return dipt

@cython.boundscheck(False)
class selfobject:
    def __init__(self, int p, double [:,:] p_coor, int apr_emim_h, int n_pairs_h, int [:] emim_h): #TODO: Bin-handling for intra-molecular?
        self.p = p
        self.apr_emim_h = apr_emim_h
        self.emim_h     = emim_h
        self.n_pairs_h  = n_pairs_h

        cdef double [:,:] pl = np.zeros((self.apr_emim_h, 3), dtype='float64')
        cdef i, j, index = 0

        cdef double [:] rv = np.zeros(3)
        cdef double r2
        cdef double f1
        cdef double f2

        for i in self.emim_h:
            pl[index, 0] = np.copy(p_coor[i,0])
            pl[index, 1] = np.copy(p_coor[i,1])
            pl[index, 2] = np.copy(p_coor[i,2])
            index += 1

        cdef double [:]   r_0    = np.zeros(self.n_pairs_h, dtype='float64')
        #cdef int    [:]   bins   = np.zeros(self.n_pairs_h, dtype='int32')
        cdef double [:,:] dipt_0 = np.zeros((self.n_pairs_h, 6), dtype='float64')
        self.r_0 = r_0
        #self.bin = bins

        index = 0
        for i in range(self.apr_emim_h-1):
            for j in range((i+1), self.apr_emim_h):
                self.r_0[index] = norm(np.float64(pl[j]) - np.float64(pl[i]))
                #self.bin[index]    = int(self.r_0[index])
                index += 1

        self.dipt_0 = dipt_0
        index = 0
        for i in range(self.apr_emim_h-1):
            for j in range((i+1), self.apr_emim_h):
                self.r_0[index]    = norm(np.float64(pl[j]) - np.float64(pl[i]))
                #self.bin[index]    = int(self.r_0[index])

                self.dipt_0[index] = dipTen(np.float64(pl[j]), np.float64(pl[i]))
                rv[0] = pl[i,0] - pl[j,0]
                rv[1] = pl[i,1] - pl[j,1]
                rv[2] = pl[i,2] - pl[j,2]
                r2 = np.dot(rv,rv) #norm squared of rv
                f1 = r2**-1.5      #prefactor 1: 1/r**3
                f2 = 3/r2          #prefactor 2: 3*1/r**2

                self.dipt_0[index,0]=f1*(rv[0]*rv[0]*f2-1)     #calculate elements
                self.dipt_0[index,1]=f1*(rv[1]*rv[1]*f2-1)     #minus 1 in definition of dipole dipole tensor
                self.dipt_0[index,2]=f1*(rv[2]*rv[2]*f2-1)
                self.dipt_0[index,3]=f1*(rv[0]*rv[1]*f2)*2     #times 2 because off diag elements appear twice
                self.dipt_0[index,4]=f1*(rv[0]*rv[2]*f2)*2
                self.dipt_0[index,5]=f1*(rv[1]*rv[2]*f2)*2

                index += 1


@cython.boundscheck(False)
class pairobject:
    def __init__(self, int p1, int p2, double [:,:] p1_coor, double [:,:] p2_coor, int apr_emim_h, int n_pairs_h, int [:] emim_h): #rename: p1 -> resnum1; p2 -> resnum2
        self.p1 = p1
        self.p2 = p2
        self.apr_emim_h = apr_emim_h
        self.n_pairs_h  = n_pairs_h
        self.emim_h     = emim_h

        cdef double [:,:] p1l = np.zeros((self.apr_emim_h, 3), dtype='float64')
        cdef double [:,:] p2l = np.zeros((self.apr_emim_h, 3), dtype='float64')
        cdef int i, j, index = 0

        cdef double [:] rv = np.zeros(3) #distance vector
        cdef double r2
        cdef double f1
        cdef double f2

        for i in self.emim_h:
            p1l[index,0] = np.copy(p1_coor[i,0])
            p1l[index,1] = np.copy(p1_coor[i,1])
            p1l[index,2] = np.copy(p1_coor[i,2])

            p2l[index,0] = np.copy(p2_coor[i,0])
            p2l[index,1] = np.copy(p2_coor[i,1])
            p2l[index,2] = np.copy(p2_coor[i,2])
            index += 1

        cdef double [:]   r_0    = np.zeros(self.n_pairs_h, dtype='float64')
        cdef int    [:]   bins   = np.zeros(self.n_pairs_h, dtype='int32')
        cdef double [:,:] dipt_0 = np.zeros((self.n_pairs_h, 6), dtype='float64')
        self.r_0    = r_0
        self.bin    = bins

        index = 0
        for i in range(self.apr_emim_h):
            for j in range(i, self.apr_emim_h):
                self.r_0[index]    = norm(np.float64(p2l[j]) - np.float64(p1l[i]))
                self.bin[index]    = int(self.r_0[index])
                index += 1

        ################
        #dipten_double_loop2_(&p1l[0,0], &p2l[0,0], &dipt_0[0,0], self.apr_emim_h-1, self.apr_emim_h, 1)
        #self.dipt_0 = dipt_0
        ################# COMMIT 5: UNCOMMENT ABOVE, COMMENT BELOW
        self.dipt_0 = dipt_0

        index = 0

        for i in range(self.apr_emim_h):
            for j in range(i, self.apr_emim_h):
                self.r_0[index]    = norm(np.float64(p2l[j]) - np.float64(p1l[i]))
                self.bin[index]    = int(self.r_0[index])

                self.dipt_0[index] = dipTen(np.float64(p1l[j]), np.float64(p2l[i]))
                rv[0] = p2l[i,0] - p1l[j,0]
                rv[1] = p2l[i,1] - p1l[j,1]
                rv[2] = p2l[i,2] - p1l[j,2]
                r2 = np.dot(rv,rv) #norm squared of rv
                f1 = r2**-1.5      #prefactor 1: 1/r**3
                f2 = 3/r2          #prefactor 2: 3*1/r**2

                self.dipt_0[index,0]=f1*(rv[0]*rv[0]*f2-1)     #calculate elements
                self.dipt_0[index,1]=f1*(rv[1]*rv[1]*f2-1)     #minus 1 in definition of dipole dipole tensor
                self.dipt_0[index,2]=f1*(rv[2]*rv[2]*f2-1)
                self.dipt_0[index,3]=f1*(rv[0]*rv[1]*f2)*2     #times 2 because off diag elements appear twice
                self.dipt_0[index,4]=f1*(rv[0]*rv[2]*f2)*2
                self.dipt_0[index,5]=f1*(rv[1]*rv[2]*f2)*2

                index += 1
        #################

@cython.boundscheck(False)
class noe_task:
    @cython.boundscheck(False)
    def __init__(self, int apr_emim_h, int [:] emim_h, int n_pairs_h, int n_self_pairs_h, int max_distance, int n_res_emim, int apr_pair, double [:,:,:] coors):
        self.apr_emim_h = apr_emim_h
        self.emim_h = emim_h
        self.n_pairs_h = n_pairs_h
        self.max_distance = max_distance
        self.pairlist = []
        self.selflist = []
        self.n_self_pairs_h = n_self_pairs_h

        cdef int i, j, atom
        cdef double [:,:] coor_i = np.zeros((apr_pair, 3), dtype='float64')
        cdef double [:,:] coor_j = np.zeros((apr_pair, 3), dtype='float64')

        for i in range(n_res_emim):
            for atom in range(apr_pair):
                    coor_i[atom, 0] = coors[i, atom, 0]
                    coor_i[atom, 1] = coors[i, atom, 1]
                    coor_i[atom, 2] = coors[i, atom, 2]
            self.selflist.append(selfobject(i, coor_i, apr_emim_h, n_self_pairs_h, emim_h))

        for i in range(n_res_emim):
            for j in range((i+1), n_res_emim):
                for atom in range(apr_pair):
                    coor_i[atom, 0] = coors[i, atom, 0]
                    coor_i[atom, 1] = coors[i, atom, 1]
                    coor_i[atom, 2] = coors[i, atom, 2]

                    coor_j[atom, 0] = coors[j, atom, 0]
                    coor_j[atom, 1] = coors[j, atom, 1]
                    coor_j[atom, 2] = coors[j, atom, 2]
                self.pairlist.append(pairobject(i, j, coor_i, coor_j, apr_emim_h, n_pairs_h, emim_h))
        #print("Pairs: ", len(self.pairlist), "Should be 499500")
        #exit()


    @cython.boundscheck(False)
    def pairiter(self, double [:,:,:] run):
        cdef double [:,:] cl = np.zeros((self.n_pairs_h, self.max_distance), dtype='float64') #rename: cl->correlationlist
        #cl = np.zeros((self.n_pairs_h, self.max_distance), dtype='float64') #ROLLBACK: ERASE THIS, UNCOMMENT BELOW
        cdef double [:,:] dipt_t = np.zeros((self.n_pairs_h, 6), dtype='float64')
        cdef double [:,:] p1l = np.zeros((self.apr_emim_h, 3), dtype='float64')
        cdef double [:,:] p2l = np.zeros((self.apr_emim_h, 3), dtype='float64')

        cdef double [:]   cl_self     = np.zeros(self.n_self_pairs_h,      dtype='float64') #Add bin information?
        #cl_self     = np.zeros(self.n_self_pairs_h,      dtype='float64') #ROLLBACK: ERASE THIS, UNCOMMENT BELOW
        cdef double [:,:] dipt_t_self = np.zeros((self.n_self_pairs_h, 6), dtype='float64')
        cdef double [:,:] pl          = np.zeros((self.apr_emim_h, 3),     dtype='float64')

        cdef int i, j, index = 0
        #cdef int [:] local_emim_h = np.int32(self.emim_h) #COMMIT 4: UNCOMMENT

        cdef double [:] rv = np.zeros(3) #distance vector
        cdef double r2
        cdef double f1
        cdef double f2
        cdef double [:] distvec = np.zeros(3, dtype='float64')
        cdef double dist_sq, dist_2, dist_3

        for selfobj in self.selflist:
            index = 0

            for i in self.emim_h:
                pl[index, 0] = run[selfobj.p][i,0]
                pl[index, 1] = run[selfobj.p][i,1]
                pl[index, 2] = run[selfobj.p][i,2]
                index += 1

            dipten_double_loop_(&pl[0,0], &pl[0,0], &dipt_t_self[0,0], self.apr_emim_h-1, self.apr_emim_h, 1)
            dipt_0_self = selfobj.dipt_0

            for i in range(self.n_self_pairs_h):
                cl_self[i] += np.dot(dipt_0_self[i], dipt_t_self[i])

        for pair in self.pairlist:
            index = 0

            for i in self.emim_h:
        
                p1l[index, 0] = run[pair.p1][i,0]
                p1l[index, 1] = run[pair.p1][i,1]
                p1l[index, 2] = run[pair.p1][i,2]
        
                p2l[index, 0] = run[pair.p2][i,0]     
                p2l[index, 1] = run[pair.p2][i,1]   
                p2l[index, 2] = run[pair.p2][i,2]
        
                index += 1
    
            dipten_double_loop_(&p1l[0,0], &p2l[0,0], &dipt_t[0,0], self.apr_emim_h, self.apr_emim_h, 0)
            dipt_0 = pair.dipt_0
  
            for i in range(self.n_pairs_h):
                cl[i][pair.bin[i]] += np.dot(dipt_0[i], dipt_t[i])
        return cl, cl_self
###END DANIEL NOE HELPERS

@cython.boundscheck(False)
def getBestIndex(double [:] aufpunkt, double [:,:] coor):
    '''
    getBestIndex(point, coor)

    This function takes a point in space and a set of coordinates. It returns the index of the particle in the set closest to said point.
    '''

    cdef int particles = coor.shape[0]
    cdef int best_index

    best_index = get_best_index(&aufpunkt[0], &coor[0,0], particles)
    return best_index

@cython.boundscheck(False)
def getBestIndices(double [:] aufpunkt, double [:,:] coor, double [:,:] dip, int count):
    '''
    coor ... (n_particles, 3)
    '''

    cdef int n_particles = coor.shape[0]
    if count > n_particles: count = n_particles
    cdef int point = 0, ctr = 0
    cdef double cutoff = 0
    cdef double [:] distances = np.zeros(n_particles)
    cdef double [:] distances_copy = np.zeros(n_particles)
    cdef double [:,:] coor_res = np.zeros((count,3))
    cdef double [:,:] dip_res = np.zeros((count,3))

    for point in range(n_particles):
        distances[point] = pow((aufpunkt[0] - coor[point, 0]),2) + pow((aufpunkt[1] - coor[point, 1]),2) + pow((aufpunkt[2] - coor[point, 2]),2)

    distances_copy = np.sort(distances, kind='mergesort')
    cutoff = distances_copy[(count-1)] #Largest allowed entry

    for point in range(n_particles):
        if(distances[point] <= cutoff): 
            coor_res[ctr] = coor[point]
            dip_res[ctr] = dip[point]
            ctr += 1

    return coor_res, dip_res

@cython.boundscheck(False)
def writeMrHistogram(double [:,:] coor, double [:,:] dip, double [:] aufpunkt, double [:] antagonist, double [:] histogram, double max_distance, int segments_per_angstroem, int order=0):
    '''
    coor        ... (number_of_particles, 3)
    dip         ... (number_of_particles, 3)
    aufpunkt    ... (3)
    antagonist  ... (3)
    histogram   ... (max_distance * segments_per_angstroem)
    '''

    cdef int number_of_particles = coor.shape[0]
    write_Mr_diagram(&coor[0,0], &dip[0,0], number_of_particles, &aufpunkt[0], &antagonist[0], &histogram[0], max_distance, segments_per_angstroem, order)

@cython.boundscheck(False)
def writeKirkwoodHistogram(double [:] aufpunkt, double[:] aufpunkt_dipole, double [:,:] coor, double [:,:] dip, double [:] histogram, double max_distance, int segments_per_angstroem, int order = 1):
    #TODO documentation

    cdef int particles = coor.shape[0]
    write_Kirkwood_diagram(&aufpunkt[0], &aufpunkt_dipole[0], &coor[0,0], &dip[0,0], particles, &histogram[0], max_distance, segments_per_angstroem, order)

def writeKirkwoodHistogramShellwise(double [:] aufpunkt, double[:] aufpunkt_dipole, double [:,:] coor, double [:,:] dip, double [:,:] histogram, double [:] norm, char [:] dataset, double max_distance, int segments_per_angstroem, int order = 1):
    '''
    aufpunkt        ... (3)
    aufpunkt_dipole ... (3)
    coor            ... (number_of_particles, 3)
    dip             ... (number_of_particles, 3)
    histogram       ... (maxshell, max_distance*segments_per_angstroem)
    norm            ... (maxshell)
    dataset         ... (number_of_particles)
    '''

    cdef int particles = coor.shape[0]
    cdef int maxshell = histogram.shape[0]
    cdef int i, shell = 0

    for i in range(particles):
        shell = dataset[i] - 1
        if shell >= maxshell:
            shell = maxshell-1
        norm[shell] += 1

    write_Kirkwood_diagram_shellwise(&aufpunkt[0], &aufpunkt_dipole[0], &coor[0,0], &dip[0,0], particles, &histogram[0,0], &dataset[0], maxshell, max_distance, segments_per_angstroem, order)

def writeKirkwoodHistogramShellwise(double [:,:] aufpunkte, double [:,:] aufpunkt_dipole, double [:,:] coor, double [:,:] dip, double [:,:] histogram, double [:] norm, char [:] dataset, double max_distance, int segments_per_angstroem, int order = 1):
    '''
    aufpunkt        ... (number_aufpunkte, 3)
    aufpunkt_dipole ... (number_aufpunkte, 3)
    coor            ... (number_of_particles, 3)
    dip             ... (number_of_particles, 3)
    histogram       ... (maxshell, max_distance*segments_per_angstroem)
    norm            ... (maxshell)
    dataset         ... (number_of_particles)
    '''

    cdef int number_aufpunkte = aufpunkte.shape[0]
    cdef int particles = coor.shape[0]
    cdef int maxshell = histogram.shape[0]
    cdef int i, shell = 0
    cdef double [:] aufpunkt = np.zeros(3)
    cdef double [:] aufpunkt_dipol = np.zeros(3)

    for i in range(particles):
        shell = dataset[i] - 1
        if shell >= maxshell:
            shell = maxshell-1
        norm[shell] += 1

    for i in range(number_aufpunkte):
        aufpunkt = aufpunkte[i]
        aufpunkt_dipol = aufpunkt_dipole[i]

        write_Kirkwood_diagram_shellwise(&aufpunkt[0], &aufpunkt_dipol[0], &coor[0,0], &dip[0,0], particles, &histogram[0,0], &dataset[0], maxshell, max_distance, segments_per_angstroem, order)

@cython.boundscheck(False)
def writeKirkwoodHistogram2D(double [:] aufpunkt, double[:] aufpunkt_dipole, double [:,:] coor, double [:,:] dip, double [:,:] histogram, double max_distance, int segments_per_angstroem, int segments_per_pi, int order = 1):
    #TODO documentation

    cdef int particles = coor.shape[0]
    write_Kirkwood_diagram_2D(&aufpunkt[0], &aufpunkt_dipole[0], &coor[0,0], &dip[0,0], particles, &histogram[0,0], max_distance, segments_per_angstroem, segments_per_pi, order)

@cython.boundscheck(False)
def getBondTable(double [:,:] coor, int [:,:] bond_table, int cutoff = 3):
    '''
    coor        ... (n_particles, 3)
    bond_table  ... (n_particles, n_particles)
    '''

    cdef int n_particles = coor.shape[0]
    calc_donor_grid(&coor[0,0], &bond_table[0,0], n_particles, cutoff)

#@cython.boundscheck(False)
#def sumVennMDCageSingle(double [:,:,:] mdcage_timeseries, double [:,:] dipoles, char [:] dataset1, char [:] dataset2, int maxshell1, int maxshell2):
#    '''
#    mdcage_timeseries   ... (number_of_shells1, number_of_shells2, 3)
#    dipoles             ... (number_of_particles, 3)
#    dataset1            ... (number_of_particles)
#    dataset2            ... (number_of_particles)
#    '''
#
#    cdef int n_particles = dipoles.shape[0]
#    calc_sum_Venn_MD_Cage_Single(&mdcage_timeseries[0,0,0], &dipoles[0,0], n_particles, &dataset1[0], &dataset2[0], maxshell1, maxshell2)

@cython.boundscheck(False)
def sumVennMDCageSingle(double [:,:,:,:] mdcage_timeseries, double [:,:] dipoles, char [:] dataset1, char [:] dataset2, int maxshell1, int maxshell2, int frame):
    
    cdef int w, shell1, shell2, i, nres_wat = dipoles.shape[0]

    for w in range(nres_wat):
        shell1 = dataset1[w]-1
        shell2 = dataset2[w]-1

        if shell1 >= maxshell1:
            shell1 = maxshell1

        if shell2 >= maxshell2:
            shell2 = maxshell2

        for i in range(3):
            mdcage_timeseries[shell1, shell2, frame, i] += dipoles[w, i]

@cython.boundscheck(False)
def sumMultiMDCageSingle(double [:,:,:] mdcage_ts, double [:,:] dip_wat, char [:] ds1, char [:] ds2, char [:] ds3, char [:] ds4, char [:] ds5, int maxshell, int frame):
    
    cdef int w, shell, i, nres_wat = dip_wat.shape[0]
    cdef int shell1, shell2, shell3, shell4, shell5

    for w in range(nres_wat):
        shell1 = ds1[w]-1
        shell2 = ds2[w]-1
        shell3 = ds3[w]-1
        shell4 = ds4[w]-1
        shell5 = ds5[w]-1

        shell = min(shell1, shell2, shell3, shell4, shell5)

        if shell < maxshell:
            for i in range(3):
                mdcage_ts[shell,frame,i] += dip_wat[w,i]
        else:
            for i in range(3):
                mdcage_ts[maxshell,frame,i] += dip_wat[w,i]

@cython.boundscheck(False)
def multiVecShellCorrelate(double [:,:,:] rotTs, int [:,:] ds1, long nshells1, long maxdt, long startingpoints):
    """
    rotTs              ... (nmol, n (time length), 3D)
    (,self,cross)coor  ... (shell1, n (time length))
    """

    cdef long nmol = <long> len(rotTs) # number of molecules
    cdef long n = <long> len(rotTs[0]) # number of time steps
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef double [:,:] corr = np.zeros((nshells1,maxdt)) 
    cdef double [:,:] selfcorr = np.zeros((nshells1,maxdt))
    cdef double [:,:] crosscorr = np.zeros((nshells1,maxdt))
    cdef double [:,:] ctr = np.zeros((nshells1,maxdt)) # counter array of correlation entries
    cdef double [:,:] selfctr = np.zeros((nshells1,maxdt))
    cdef double [:,:] crossctr = np.zeros((nshells1,maxdt))
    cdef double t1
    cdef long i,j,k,l,start,point,dt,mol,shellstart1,shelldt1
   
    #for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t ROLLBACK: Uncomment this, erase below line
    for dt in range(maxdt):
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shellstart1 = ds1[mol,start]-1

                if (shellstart1 < nshells1):
                    ctr[shellstart1,dt] += 1
                    
                    t1 = 0
                    for k in range(3): # skalar produkt
                        t1 += rotTs[mol,start,k]*rotTs[mol,start+dt,k]
                    corr[shellstart1,dt] += t1 # l=1

                    shelldt1 =  ds1[mol,start+dt]-1
                    if (shellstart1 == shelldt1):
                        selfctr[shellstart1,dt] += 1
                        selfcorr[shellstart1,dt] += t1

    for i in range(nshells1):
        for j in range(maxdt):
            if ctr[i,j] != 0:
                corr[i,j] /= ctr[i,j]
            else:
                print 'tot too sparse'
            if selfctr[i,j] != 0:
                selfcorr[i,j] /= selfctr[i,j]
            else:
                print 'self too sparse'

    for i in range(nshells1):
        for j in range(maxdt):
            crossctr[i,j]  = ctr[i,j]  - selfctr[i,j]
            crosscorr[i,j] = corr[i,j] - selfcorr[i,j]
       
    return corr,selfcorr,crosscorr,ctr,selfctr,crossctr

@cython.boundscheck(False)
def multiVecVennShellCorrelate(double [:,:,:] rotTs, int [:,:] ds1, int [:,:] ds2, long nshells1, long nshells2, long maxdt, long startingpoints):
    """
    rotTs              ... (nmol, n (time length), 3D)
    (,self,cross)coor  ... (shell1, shell2, n (time length))
    """

    cdef long nmol = <long> len(rotTs) # number of molecules
    cdef long n = <long> len(rotTs[0]) # number of time steps
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef double [:,:,:] corr = np.zeros((nshells1,nshells2,maxdt)) 
    cdef double [:,:,:] selfcorr = np.zeros((nshells1,nshells2,maxdt))
    cdef double [:,:,:] crosscorr = np.zeros((nshells1,nshells2,maxdt))
    cdef double [:,:,:] ctr = np.zeros((nshells1,nshells2,maxdt)) # counter array of correlation entries
    cdef double [:,:,:] selfctr = np.zeros((nshells1,nshells2,maxdt))
    cdef double [:,:,:] crossctr = np.zeros((nshells1,nshells2,maxdt))
    cdef double t1
    cdef long i,j,k,l,start,point,dt,mol,shellstart1,shellstart2,shelldt1,shelldt2
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shellstart1 = ds1[mol,start]-1
                shellstart2 = ds2[mol,start]-1
                if (shellstart1 < nshells1) and (shellstart2 < nshells2):
                    ctr[shellstart1,shellstart2,dt] += 1
                    
                    t1 = 0
                    for k in range(3): # skalar produkt
                        t1 += rotTs[mol,start,k]*rotTs[mol,start+dt,k]
                    corr[shellstart1,shellstart2,dt] += t1 # l=1

                    shelldt1 =  ds1[mol,start+dt]-1
                    shelldt2 =  ds2[mol,start+dt]-1

                    if (shellstart1 == shelldt1) and (shellstart2 == shelldt2):
                        selfctr[shellstart1,shellstart2,dt] += 1
                        selfcorr[shellstart1,shellstart2,dt] += t1

    for i in range(nshells1):
        for l in range(nshells2):
            for j in range(maxdt):
                if ctr[i,l,j] != 0:
                    corr[i,l,j] /= ctr[i,l,j]
                else:
                    print 'tot too sparse'
                if selfctr[i,l,j] != 0:
                    selfcorr[i,l,j] /= selfctr[i,l,j]
                else:
                    print 'self too sparse'

    for i in range(nshells1):
        for l in range(nshells2):
            for j in range(maxdt):
                crossctr[i,l,j]  = ctr[i,l,j]  - selfctr[i,l,j]
                crosscorr[i,l,j] = corr[i,l,j] - selfcorr[i,l,j]
        
    return corr,selfcorr,crosscorr,ctr,selfctr,crossctr

@cython.boundscheck(False)
def rotationMatrixVennShellCorrelate(double [:,:,:,:] rotTs, int [:,:] ds1, int [:,:] ds2, long nshells1, long nshells2, long maxdt, long startingpoints):
    cdef long nmol = <long> len(rotTs) # number of molecules
    cdef long n = <long> len(rotTs[0]) # number of time steps
#    cdef long nds = <long> len(ds) # number of steps in delaunay array
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef double [:,:,:,:] corr = np.zeros((12,nshells1,nshells2,maxdt)) # 3x l={1,2} rotautocorr+ 3x l={1,2} rotcrosscorr = 12
    cdef double [:,:,:,:] selfcorr = np.zeros((12,nshells1,nshells2,maxdt))
    cdef double [:,:,:,:] crosscorr = np.zeros((12,nshells1,nshells2,maxdt))
    cdef double [:,:,:] ctr = np.zeros((nshells1,nshells2,maxdt)) # counter array of correlation entries
    cdef double [:,:,:] selfctr = np.zeros((nshells1,nshells2,maxdt))
    cdef double [:,:,:] crossctr = np.zeros((nshells1,nshells2,maxdt))
    cdef double t1,t2,t3,t4,t5,t6
    cdef long i,j,k,l,start,point,dt,mol,shellstart1,shellstart2,shelldt1,shelldt2
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shellstart1 = ds1[mol,start]-1
                shellstart2 = ds2[mol,start]-1
                if (shellstart1 < nshells1) and (shellstart2 < nshells2):
                    ctr[shellstart1,shellstart2,dt] += 1
                    
                    t1,t2,t3,t4,t5,t6 = 0,0,0,0,0,0
                    for k in range(3): # skalar produkt
                        t1 += rotTs[mol,start,0,k]*rotTs[mol,start+dt,0,k]
                        t2 += rotTs[mol,start,1,k]*rotTs[mol,start+dt,1,k]
                        t3 += rotTs[mol,start,2,k]*rotTs[mol,start+dt,2,k]
                        t4 += rotTs[mol,start,0,k]*rotTs[mol,start+dt,1,k]
                        t5 += rotTs[mol,start,0,k]*rotTs[mol,start+dt,2,k]
                        t6 += rotTs[mol,start,1,k]*rotTs[mol,start+dt,2,k]
                    corr[0,shellstart1,shellstart2,dt] += t1 # l=1
                    corr[1,shellstart1,shellstart2,dt] += t2
                    corr[2,shellstart1,shellstart2,dt] += t3
                    corr[3,shellstart1,shellstart2,dt] += t4
                    corr[4,shellstart1,shellstart2,dt] += t5
                    corr[5,shellstart1,shellstart2,dt] += t6
                    corr[6,shellstart1,shellstart2,dt] += 1.5*t1*t1-0.5 # l=2
                    corr[7,shellstart1,shellstart2,dt] += 1.5*t2*t2-0.5
                    corr[8,shellstart1,shellstart2,dt] += 1.5*t3*t3-0.5
                    corr[9,shellstart1,shellstart2,dt] += 1.5*t4*t4-0.5
                    corr[10,shellstart1,shellstart2,dt] += 1.5*t5*t5-0.5
                    corr[11,shellstart1,shellstart2,dt] += 1.5*t6*t6-0.5
                    
                    shelldt1 =  ds1[mol,start+dt]-1
                    shelldt2 =  ds2[mol,start+dt]-1

                    if (shellstart1 == shelldt1) and (shellstart2 == shelldt2):
                        selfctr[shellstart1,shellstart2,dt] += 1
                        
                        selfcorr[0,shellstart1,shellstart2,dt] += t1
                        selfcorr[1,shellstart1,shellstart2,dt] += t2
                        selfcorr[2,shellstart1,shellstart2,dt] += t3
                        selfcorr[3,shellstart1,shellstart2,dt] += t4
                        selfcorr[4,shellstart1,shellstart2,dt] += t5
                        selfcorr[5,shellstart1,shellstart2,dt] += t6
                        selfcorr[6,shellstart1,shellstart2,dt] += 1.5*t1*t1-0.5
                        selfcorr[7,shellstart1,shellstart2,dt] += 1.5*t2*t2-0.5
                        selfcorr[8,shellstart1,shellstart2,dt] += 1.5*t3*t3-0.5
                        selfcorr[9,shellstart1,shellstart2,dt] += 1.5*t4*t4-0.5
                        selfcorr[10,shellstart1,shellstart2,dt] += 1.5*t5*t5-0.5
                        selfcorr[11,shellstart1,shellstart2,dt] += 1.5*t6*t6-0.5

    for i in range(nshells1):
        for l in range(nshells2):
            for j in range(maxdt):
                if ctr[i,l,j] != 0:
                    for k in range(12):
                        corr[k,i,l,j] /= ctr[i,l,j]
                else:
                    print 'tot too sparse'
                if selfctr[i,l,j] != 0:
                    for k in range(12):
                        selfcorr[k,i,l,j] /= selfctr[i,l,j]
                else:
                    print 'self too sparse'

    for i in range(nshells1):
        for l in range(nshells2):
            for j in range(maxdt):
                crossctr[i,l,j] = ctr[i,l,j] - selfctr[i,l,j]
                for k in range(12):
                    crosscorr[k,i,l,j] = corr[k,i,l,j] - selfcorr[k,i,l,j]
        
    return corr,selfcorr,crosscorr,ctr,selfctr,crossctr


@cython.boundscheck(False)
def rotationMatrixMultiShellCorrelate(double [:,:,:,:] rotTs, int [:,:] ds1, int [:,:] ds2, int [:,:] ds3, int [:,:] ds4, int [:,:] ds5, long nshells, long maxdt, long startingpoints):
    cdef long nmol = <long> len(rotTs) # number of molecules
    cdef long n = <long> len(rotTs[0]) # number of time steps
#    cdef long nds = <long> len(ds) # number of steps in delaunay array
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef double [:,:,:] corr = np.zeros((12,nshells,maxdt)) # 3x l={1,2} rotautocorr+ 3x l={1,2} rotcrosscorr = 12
    cdef double [:,:,:] selfcorr = np.zeros((12,nshells,maxdt))
    cdef double [:,:,:] crosscorr = np.zeros((12,nshells,maxdt))
    cdef double [:,:] ctr = np.zeros((nshells,maxdt)) # counter array of correlation entries
    cdef double [:,:] selfctr = np.zeros((nshells,maxdt))
    cdef double [:,:] crossctr = np.zeros((nshells,maxdt))
    cdef double t1,t2,t3,t4,t5,t6
    cdef long i,j,k,start,point,dt,mol,shellstart,shelldt
    cdef long shell1, shell2, shell3, shell4, shell5
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shell1 = ds1[mol,start]-1
                shell2 = ds2[mol,start]-1
                shell3 = ds3[mol,start]-1
                shell4 = ds4[mol,start]-1
                shell5 = ds5[mol,start]-1
                shellstart = min(shell1, shell2, shell3, shell4, shell5)
                if shellstart < nshells:
                    ctr[shellstart,dt] += 1
                    
                    t1,t2,t3,t4,t5,t6 = 0,0,0,0,0,0
                    for k in range(3): # skalar produkt
                        t1 += rotTs[mol,start,0,k]*rotTs[mol,start+dt,0,k]
                        t2 += rotTs[mol,start,1,k]*rotTs[mol,start+dt,1,k]
                        t3 += rotTs[mol,start,2,k]*rotTs[mol,start+dt,2,k]
                        t4 += rotTs[mol,start,0,k]*rotTs[mol,start+dt,1,k]
                        t5 += rotTs[mol,start,0,k]*rotTs[mol,start+dt,2,k]
                        t6 += rotTs[mol,start,1,k]*rotTs[mol,start+dt,2,k]
                    corr[0,shellstart,dt] += t1 # l=1
                    corr[1,shellstart,dt] += t2
                    corr[2,shellstart,dt] += t3
                    corr[3,shellstart,dt] += t4
                    corr[4,shellstart,dt] += t5
                    corr[5,shellstart,dt] += t6
                    corr[6,shellstart,dt] += 1.5*t1*t1-0.5 # l=2
                    corr[7,shellstart,dt] += 1.5*t2*t2-0.5
                    corr[8,shellstart,dt] += 1.5*t3*t3-0.5
                    corr[9,shellstart,dt] += 1.5*t4*t4-0.5
                    corr[10,shellstart,dt] += 1.5*t5*t5-0.5
                    corr[11,shellstart,dt] += 1.5*t6*t6-0.5

                    shell1 = ds1[mol,start+dt]-1
                    shell2 = ds2[mol,start+dt]-1
                    shell3 = ds3[mol,start+dt]-1
                    shell4 = ds4[mol,start+dt]-1
                    shell5 = ds5[mol,start+dt]-1                    
                    shelldt = min(shell1, shell2, shell3, shell4, shell5)
                    if shellstart == shelldt:
                        selfctr[shellstart,dt] += 1
                        
                        selfcorr[0,shellstart,dt] += t1
                        selfcorr[1,shellstart,dt] += t2
                        selfcorr[2,shellstart,dt] += t3
                        selfcorr[3,shellstart,dt] += t4
                        selfcorr[4,shellstart,dt] += t5
                        selfcorr[5,shellstart,dt] += t6
                        selfcorr[6,shellstart,dt] += 1.5*t1*t1-0.5
                        selfcorr[7,shellstart,dt] += 1.5*t2*t2-0.5
                        selfcorr[8,shellstart,dt] += 1.5*t3*t3-0.5
                        selfcorr[9,shellstart,dt] += 1.5*t4*t4-0.5
                        selfcorr[10,shellstart,dt] += 1.5*t5*t5-0.5
                        selfcorr[11,shellstart,dt] += 1.5*t6*t6-0.5

    for i in range(nshells):
        for j in range(maxdt):
            if ctr[i,j] != 0:
                for k in range(12):
                    corr[k,i,j] /= ctr[i,j]
            else:
                print 'tot too sparse'
            if selfctr[i,j] != 0:
                for k in range(12):
                    selfcorr[k,i,j] /= selfctr[i,j]
            else:
                print 'self too sparse'
    for i in range(nshells):
        for j in range(maxdt):
            crossctr[i,j] = ctr[i,j] - selfctr[i,j]
            for k in range(12):
                crosscorr[k,i,j] = corr[k,i,j] - selfcorr[k,i,j]
        
    return corr,selfcorr,crosscorr,ctr,selfctr,crossctr

@cython.boundscheck(False)
def collectiveDipolesCutoff(double [:,:] coor, double[:,:] dipoles, double[:] aufpunkt, double[:] dip_inside, double[:] dip_outside, double cutoff):
    #TODO documentation

    cdef int particles = coor.shape[0]
    separateCollectiveDipolesSpherically(&coor[0,0], &dipoles[0,0], particles, &aufpunkt[0], &dip_inside[0], &dip_outside[0], cutoff)

#TODO: As a C++ function?
@cython.boundscheck(False)
def shellHistograms(double [:,:] histograms, double [:] aufpunkt, double [:,:] coors, char [:] ds, int maxshell, double max_distance, int segs_per_angstroem):
    #TODO documentation

    #cdef int w, shell, i, nres_wat = dip_wat.shape[0]
    cdef int i, shell, bin_dist, particles = coors.shape[0]
    cdef double distance

    for i in range(particles):
        shell = ds[i] - 1
        if shell < maxshell:
            distance = np.sqrt(pow((coors[i][0] - aufpunkt[0]), 2) + pow((coors[i][1] - aufpunkt[1]), 2) + pow((coors[i][2] - aufpunkt[2]), 2))
            if distance < max_distance:
                bin_dist = int(distance * segs_per_angstroem)
                histograms[shell][bin_dist] += 1

#TODO: As a C++ function?
#@cython.boundscheck(False)
#def sumMDVennCageSingle(double [:,:,:,:] mdcage_ts, double [:,:] dip_wat, char [:] ds1, char [:] ds2, int maxshell1, int maxshell2, int frame):
#    
#    cdef int w, shell1, shell2, i, nres_wat = dip_wat.shape[0]
#
#    for w in range(nres_wat):
#        shell1 = ds1[w]-1
#        shell2 = ds2[w]-1
#
#        if shell1 < maxshell1 and shell2 < maxshell2:
#            for i in range(3):
#                mdcage_ts[shell1, shell2, frame, i] += dip_wat[w, i]
#
#        else:
#            if shell1 < maxshell1 and shell2 >= maxshell2:
#                for i in range(3):
#                    mdcage_ts[shell1, maxshell2, frame, i] += dip_wat[w, i]
#
#            elif shell2 < maxshell2 and shell1 >= maxshell1:
#                for i in range(3):
#                    mdcage_ts[maxshell1, shell2, frame, i] += dip_wat[w, i]
#
#            else:
#                for i in range(3):
#                    mdcage_ts[maxshell1, maxshell2, frame, i] += dip_wat[w, i]



@cython.boundscheck(False)
def correlateSingleVector(double[:,:] timeseries, double[:] result, int order = 1):
    """
    Correlate the timeseries of a single vector (e.g. collective dipole, one single particle, etc...)
    timeseries  ... (number_of_frames, 3)
    result      ... (number_of_frames)
    order       ... Which order of Lagrangian Polynom (1-6 available)
    """

    cdef int number_of_frames = timeseries.shape[0]
    correlateSingleVectorTS(&timeseries[0,0], &result[0], number_of_frames, order)

@cython.boundscheck(False)
def crossCorrelateSingleVector(double[:,:] timeseries1, double[:,:] timeseries2, double[:] result, int both_directions = 1, int order = 1):
    """
    Correlate the timeseries of two different single vectors (e.g. collective dipole, one single particle, etc...)
    ATTN: Both timeseries need to be equally long! #TODO: Aendern?

    timeseries1     ... (number_of_frames1, 3)
    timeseries2     ... (number_of_frames2, 3)
    result          ... (number_of_frames)
    both_directions ... Do you want to cross-correlate both vectors, that is <A(0) B(t)> + <B(0) A(t)> yes/no.
    order           ... Which order of Lagrangian Polynom (1-6 available)
    """

    cdef int number_of_frames1 = timeseries1.shape[0]
    cdef int number_of_frames2 = timeseries2.shape[0]
    crossCorrelateSingleVectorTS(&timeseries1[0,0], &timeseries2[0,0], &result[0], number_of_frames1, number_of_frames2, both_directions, order)

@cython.boundscheck(False)
def correlateMultiVector(double[:,:,:] timeseries, double[:] result, int order = 1):
    """
    Correlate the timeseries of multiple vectors (e.g. single-particle dynamics of a collection of dipoles,...)
    timeseries  ... (number_of_particles, number_of_frames, 3)
    result      ... (number_of_frames)
    order       ... Which order of Lagrangian Polynom (1-6 available)
    """

    cdef int number_of_particles = timeseries.shape[0]
    cdef int number_of_frames = timeseries.shape[1]
    correlateMultiVectorTS(&timeseries[0,0,0], &result[0], number_of_particles, number_of_frames, order)

@cython.boundscheck(False)
def correlateMultiVectorShellwise(double [:,:,:] timeseries, double [:,:] dataset, double [:,:] result, int maxshell, int order = 1):
    """
    timeseries  ... (number_of_particles, number_of_frames, 3)
    dataset     ... (number_of_frames, number_of_particles)
    result      ... (number_of_shells, number_of_frames)
    order       ... Which order of Lagrangian Polynom (1-6 available)  
    """

    cdef int number_of_particles = timeseries.shape[0]
    cdef int number_of_frames = timeseries.shape[1]
    correlateMultiVectorShellwiseTS(&timeseries[0,0,0], &dataset[0,0], &result[0,0], number_of_particles, number_of_frames, maxshell, order)

def correlateMultiVectorVennShellwise(double [:,:,:] timeseries, double [:,:] dataset1, double [:,:] dataset2, double [:,:,:] result, int maxshell1, int maxshell2, int order = 1):
    """
    timeseries  ... (number_of_particles, number_of_frames, 3)
    dataset1    ... (number_of_frames, number_of_particles)
    dataset2    ... (number_of_frames, number_of_particles)
    result      ... (number_of_shells1, number_of_shells2, number_of_frames)
    order       ... Which order of Lagrangian Polynom (1-6 available)  
    """

    cdef int number_of_particles = timeseries.shape[0]
    cdef int number_of_frames = timeseries.shape[1]
    correlateMultiVectorVennShellwiseTS(&timeseries[0,0,0], &dataset1[0,0],  &dataset2[0,0], &result[0,0,0], number_of_particles, number_of_frames, maxshell1, maxshell2, order)

@cython.boundscheck(False)
def crossCorrelateMultiVector(double[:,:,:] timeseries1, double[:,:,:] timeseries2, double[:] result, int both_directions = 1, int order = 1):
    """
    Correlate the timeseries of multiple vectors (e.g. single-particle dynamics of a collection of dipoles,...)
    ATTN: Both timeseries need to be equally long and contain the equal amount of particles! #TODO: Aendern?

    timeseries1      ... (number_of_particles, number_of_frames, 3)
    timeseries2      ... (number_of_particles, number_of_frames, 3)
    result           ... (number_of_frames)
    both_directions  ... Do you want to cross-correlate both vectors, that is <A(0) B(t)> + <B(0) A(t)> yes/no.
    order            ... Which order of Lagrangian Polynom (1-6 available)
    """

    cdef int number_of_particles = timeseries1.shape[0]
    cdef int number_of_frames1 = timeseries1.shape[1]
    cdef int number_of_frames2 = timeseries2.shape[1]
    crossCorrelateMultiVectorTS(&timeseries1[0,0,0], &timeseries2[0,0,0], &result[0], number_of_particles, number_of_frames1, number_of_frames2, both_directions, order)

@cython.boundscheck(False)
def correlateSingleParticleMuShellwise(double[:,:,:] timeseries, double[:,:] result, int[:,:] dataset, int number_of_startingpoints):
    """
    timeseries... (N_particle, number_of_frames, mu)
    result    ... (N_shell, correlation_length)
    """

    cdef int i, j, k, shell
    cdef int particles = timeseries.shape[1]
    cdef int number_of_frames = timeseries.shape[0]
    cdef int number_of_shells = result.shape[0]
    cdef int correlation_length = result.shape[1]
    cdef double [:] sub_correlation = np.zeros((correlation_length))
    cdef double norm
    cdef double [:] counter = np.zeros((number_of_shells))

    if(correlation_length > number_of_frames):
        print("Correlation length must not be longer than the available trajectory length")
        return

    if(number_of_frames - correlation_length)/number_of_startingpoints == 0:
        print("Number of starting points too high, or else trajectory length or correlation length too close to each other")
        return

    if((number_of_frames - correlation_length)/float(number_of_startingpoints) - int((number_of_frames - correlation_length)/number_of_startingpoints) != 0):
        print("Warning: number_of_frames - correlation_length not evenly divideable regarding the number or starting points")

    #Loop over each starting point
    for i in range(0, (number_of_frames - correlation_length),  int((number_of_frames - correlation_length)/number_of_startingpoints)):
        #Loop over each particle
        for j in range(0, particles):
            shell = dataset[j, i] - 1
            if(shell < number_of_shells):
                #Loop over each distance
                norm = timeseries[i, j, 0]*timeseries[i, j, 0] + timeseries[i, j, 1]*timeseries[i, j, 1] + timeseries[i, j, 2]*timeseries[i, j, 2]

                if(norm != 0):
                    for k in range(i, i+correlation_length):
                        result[shell][k-i] += (timeseries[k, j, 0]*timeseries[i, j, 0] + timeseries[k, j, 1]*timeseries[i, j, 1] + timeseries[k, j, 2]*timeseries[i, j, 2])/norm

                    #for k in range(0, correlation_length):
                        #result[shell][k] += sub_correlation[k]
                    counter[shell] += 1

    for i in range(0, number_of_shells):
           for k in range(0, correlation_length):
                if(counter[i] > 0):
                    result[i][k] /= counter[i]

@cython.boundscheck(False)
def correlateSingleParticleMuVennShellwise(double[:,:,:] timeseries, double[:,:,:] result, int[:,:] dataset1, int[:,:] dataset2, int number_of_startingpoints):
    """
    timeseries... (N_particle, number_of_frames, mu)
    result    ... (N_shell, correlation_length)
    """

    cdef int i, j, k, shell1, shell2
    cdef int particles = timeseries.shape[1]
    cdef int number_of_frames = timeseries.shape[0]
    cdef int number_of_shells1 = result.shape[0]
    cdef int number_of_shells2 = result.shape[1]
    cdef int correlation_length = result.shape[2]
    cdef double norm
    cdef double [:,:] counter = np.zeros((number_of_shells1, number_of_shells2))

    if(correlation_length > number_of_frames):
        print("Correlation length must not be longer than the available trajectory length")
        return

    if(number_of_frames - correlation_length)/number_of_startingpoints == 0:
        print("Number of starting points too high, or else trajectory length or correlation length too close to each other")
        return

    if((number_of_frames - correlation_length)/float(number_of_startingpoints) - int((number_of_frames - correlation_length)/number_of_startingpoints) != 0):
        print("Warning: number_of_frames - correlation_length not evenly divideable regarding the number or starting points")

    #Loop over each starting point
    for i in range(0, (number_of_frames - correlation_length),  int((number_of_frames - correlation_length)/number_of_startingpoints)):
        #Loop over each particle
        for j in range(0, particles):
            shell1 = dataset1[j, i] - 1
            shell2 = dataset2[j, i] - 1
            if(shell1 < number_of_shells1 and shell2 < number_of_shells2):
                #Loop over each distance
                norm = timeseries[i, j, 0]*timeseries[i, j, 0] + timeseries[i, j, 1]*timeseries[i, j, 1] + timeseries[i, j, 2]*timeseries[i, j, 2]

                for k in range(i, i+correlation_length):
                    result[shell1][shell2][k-i] += (timeseries[k, j, 0]*timeseries[i, j, 0] + timeseries[k, j, 1]*timeseries[i, j, 1] + timeseries[k, j, 2]*timeseries[i, j, 2])/norm

                #for k in range(0, correlation_length):
                    #result[shell][k] += sub_correlation[k]
                counter[shell1][shell2] += 1

    for i in range(0, number_of_shells1):
        for j in range(0, number_of_shells2):
            for k in range(0, correlation_length):
                if(counter[i][j] > 0):
                    result[i][j][k] /= counter[i][j]

@cython.boundscheck(False)
def vanHoveSingleVector(double[:,:] timeseries, double[:,:] histogram):
    """
    timeseries  ... (correlation_length * 3)
    histogram   ... (correlation_length * cos_segs)
    """

    cdef int correlation_length = timeseries.shape[0]
    cdef int cos_segs = histogram.shape[1]

    calcVanHoveSingleVector(&timeseries[0,0], &histogram[0,0], correlation_length, cos_segs)

@cython.boundscheck(False)
def vanHoveMultiVector(double[:,:,:] timeseries, double[:,:] histogram):
    """
    timeseries  ... (n_particles * correlation_length * 3)
    histogram   ... (correlation_length * cos_segs)
    """

    cdef int n_particles = timeseries.shape[0]
    cdef int correlation_length = timeseries.shape[1]
    cdef int cos_segs = histogram.shape[1]

    calcVanHoveMultiVector(&timeseries[0,0,0], &histogram[0,0], n_particles, correlation_length, cos_segs)

@cython.boundscheck(False)
def sortCollectiveDipNNShells(char[:,:] ds, double [:,:] dip_wat, double [:,:] dip_shell):
    """
    ds        ... (n_particles, n_particles)
    dip_wat   ... (n_particles, 3)
    dip_shell ... (n_shells, 3)
    """

    cdef int n_particles = dip_wat.shape[0]
    sort_collective_dip_NN_shells(&ds[0,0], &dip_wat[0,0], &dip_shell[0,0], n_particles)
    
@cython.boundscheck(False)
def sortCollectiveDipNNShellsInt(int[:,:] ds, double [:,:] dip_wat, double [:,:] dip_shell):
    """
    ds        ... (n_particles, n_particles)
    dip_wat   ... (n_particles, 3)
    dip_shell ... (n_shells, 3)
    """

    cdef int n_particles = dip_wat.shape[0]
    sort_collective_dip_NN_shells_int(&ds[0,0], &dip_wat[0,0], &dip_shell[0,0], n_particles)

@cython.boundscheck(False)
def calcDipTenCollective(double[:,:] coor, double[:] results):
    """
    coor    ... (n_particles, 3)
    results ... (6) 
    """
    cdef int n_particles = coor.shape[0]

    calc_dip_ten_collective(&coor[0,0], n_particles, &results[0])

@cython.boundscheck(False)
def calcDipTenCollectivePerAtom(int idx, double[:,:] coor, double[:] results):
    """
    coor    ... (n_particles, 3)
    results ... (6) 
    """
    cdef int n_particles = coor.shape[0]

    calc_dip_ten_collective_per_atom(idx, &coor[0,0], n_particles, &results[0])

@cython.boundscheck(False)
def calcDipTenCollectiveCross(double[:,:] coor1, double[:,:] coor2, double[:] results):
    """
    coor1   ... (n_particles1, 3)
    coor2   ... (n_particles2, 3)
    results ... (6) 
    """
    cdef int n_particles1 = coor1.shape[0]
    cdef int n_particles2 = coor2.shape[0]

    calc_dip_ten_collective_cross(&coor1[0,0], n_particles1, &coor2[0,0], n_particles2, &results[0])


@cython.boundscheck(False)
def calcDipTenCollectiveNNShellwise(double[:,:] coor, char [:,:] ds, double[:,:] results):
    """
    coor    ... (n_particles, 3)
    ds      ... (n_particles, n_particles)
    results ... (shells, 6)
    """
    cdef int n_particles = coor.shape[0]
    cdef int shells = results.shape[0]

    calc_dip_ten_collective_NNshellwise(&coor[0,0], n_particles, &ds[0,0], shells, &results[0,0])


@cython.boundscheck(False)
def calcDipTenCollectiveNNShellwiseSelf(double[:,:] coor, int[:] f2c, char [:,:] ds, int ds_idx, double[:,:] results):
    """
    coor    ... (n_particles, 3)
    ds      ... (n_particles_tot, n_particles_tot)
    results ... (shells, 6)
    """
    cdef int n_particles = coor.shape[0]
    cdef int n_particles_tot = ds.shape[0]
    cdef int shells = results.shape[0]

    calc_dip_ten_collective_NNshellwise_self(&coor[0,0], &f2c[0], n_particles, n_particles_tot, &ds[0,0], ds_idx, shells, &results[0,0])


@cython.boundscheck(False)
def calcDipTenCollectiveNNShellwiseCross(double[:,:] coor1, double[:,:] coor2, int[:] f2c, char [:,:] ds, int ds1_idx, int ds2_idx, double[:,:] results):
    """
    coor1   ... (n_particles1, 3)
    coor2   ... (n_particles2, 3)
    ds      ... (n_particles_tot, n_particles_tot)
    results ... (shells, 6)
    """
    cdef int n_particles1 = coor1.shape[0]
    cdef int n_particles2 = coor2.shape[0]
    cdef int n_particles_tot = ds.shape[0]
    cdef int shells = results.shape[0]

    calc_dip_ten_collective_NNshellwise_cross(&coor1[0,0], &coor2[0,0], &f2c[0], n_particles1, n_particles2, n_particles_tot, &ds[0,0], ds1_idx, ds2_idx, shells, &results[0,0])


@cython.boundscheck(False)
def calcDipTenCollective1NShellwiseSelf(double[:,:] coor, char [:] ds, double[:,:,:] results):
    """
    coor    ... (n_particles, 3)
    ds      ... (n_particles)
    results ... (shells, shells, 6)
    """
    cdef int n_particles = coor.shape[0]
    cdef int shells = results.shape[0]

    calc_dip_ten_collective_1Nshellwise_self(&coor[0,0], n_particles, &ds[0], shells, &results[0,0,0])


@cython.boundscheck(False)
def calcDipTenCollective1NShellwiseCross(double[:,:] coor1, double[:,:] coor2, char [:] ds, double[:,:] results):
    """
    coor1   ... (n_particles1, 3)
    coor2   ... (n_particles2, 3)
    ds      ... (n_particles1)       #IMPORTANT!
    results ... (shells, 6)
    """
    cdef int n_particles1 = coor1.shape[0]
    cdef int n_particles2 = coor2.shape[0]
    cdef int shells = results.shape[0]

    calc_dip_ten_collective_1Nshellwise_cross(&coor1[0,0], n_particles1, &coor2[0,0], n_particles2, &ds[0], shells, &results[0,0])


# Deprecated!
@cython.boundscheck(False)
def calcDipTenCollectiveShellwise(double[:,:] coor, char [:] ds, int maxshell, double[:,:] results):
    """
    coor    ... (n_particles, 3)
    ds      ... (n_particles)
    results ... (shells, 6)
    """
    cdef int n_particles = coor.shape[0]

    calc_dip_ten_collective_shellwise(&coor[0,0], n_particles, &ds[0], maxshell, &results[0,0])

# Deprecated!
@cython.boundscheck(False)
def calcDipTenCollectiveVennShellwise(double[:,:] coor, char [:] ds1, char [:] ds2, int maxshell1, int maxshell2, double[:,:,:] results):
    """
    coor    ... (n_particles, 3)
    ds1     ... (n_particles)
    ds2     ... (n_particles)
    results ... (shells1, shells2, n_particles)
    """
    cdef int n_particles = coor.shape[0]

    calc_dip_ten_collective_vennshellwise(&coor[0,0], n_particles, &ds1[0], &ds2[0], maxshell1, maxshell2, &results[0,0,0])

@cython.boundscheck(False)
def calcDistanceDelaunyMindist(double[:,:] coor, double boxlength, int number_of_shells, double bin_width=1.0):
    """
    coor    ... (n_particles, 3)
    """

    cdef int n_particles = coor.shape[0]
    cdef int[:,:] delauny_matrix = np.zeros((n_particles, n_particles), dtype = "int32")
    calc_distance_delauny_mindist(&delauny_matrix[0,0], &coor[0,0], n_particles, boxlength, number_of_shells, bin_width)

    return delauny_matrix

@cython.boundscheck(False)
def calcDistanceDelauny(double[:,:] coor, int number_of_shells, double bin_width=1.0):
    """
    coor    ... (n_particles, 3)
    """

    cdef int n_particles = coor.shape[0]
    cdef int[:,:] delauny_matrix = np.zeros((n_particles, n_particles), dtype = "int32")
    calc_distance_delauny(&delauny_matrix[0,0], &coor[0,0], n_particles, number_of_shells, bin_width)

    return delauny_matrix


# Deprecated!
@cython.boundscheck(False)
def minDistTesselation(double [:] dataset, double [:,:] coor_core, double [:,:] coor_surround, double binwidth = 1):
    """
    dataset         ... (n_particles_surround)
    coor_core       ... (n_particles_core, 3)
    coor_surround   ... (n_particles_surround, 3)
    """

    cdef int n_particle_core = coor_core.shape[0]
    cdef int n_particle_surround = coor_surround.shape[0]

    calc_min_dist_tesselation(&dataset[0], &coor_core[0,0], &coor_surround[0,0], n_particle_core, n_particle_surround, binwidth)

@cython.boundscheck(False)
def getPolarizabilityMatrix(double[:,:] coor, double[:,:,:] inv_atom_polarizabilities, double[:,:] matrix):
    """
    coor                        ... (n_atoms, 3)
    inv_atom_polarizabilities   ... (n_atoms, 3, 3)
    matrix                      ... (3*n_atoms, 3*n_atoms)
    """

    cdef int n_atoms = coor.shape[0]

    construct_relay_matrix(&coor[0,0], &inv_atom_polarizabilities[0,0,0], &matrix[0,0], n_atoms)

    #TODO: invert relay matrix
