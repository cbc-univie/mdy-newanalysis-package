# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport fabs, sqrt, floor, pow

cdef inline int sign(double a) nogil:
     return ((<double> 0.0 < a) - (a < <double> 0.0))

cdef extern from "BertholdHorn.h":
    void GetRotation(double *R_ptr, int n, double *APoints_ptr, double *BPoints_ptr, int debug)

@cython.boundscheck(False)
def unfoldBox(double [:,:] coor_unfold, double [:,:] coor_prev, double [:,:] coor_curr, double boxl):
    """
    unfoldBox(xyz_unfold,xyz_prev,xyz_curr,boxlength)
    
    You have to hand over 3 coordinate sets and the boxlength as arguments: 
    - the in-place-unfolding coordinates (from the previous timestep)
    - the trajectory coordinates from the previous timestep
    - the trajectory coordinates from the current timestep
    - boxlength

    This functions returns the unfolded coordinates of the current timestep.

    Arguments:
        coor_unfold     .. numpy array (float64, ndim=2) in-place-unfolding coordinate set: [[x,y,z],[x,y,z],...,[x,y,z]]
        coor_prev  .. numpy array (float64, ndim=2) trajectory coordinates, previous timestep: [[x,y,z],[x,y,z],...,[x,y,z]]
        coor_curr  .. numpy array (float64, ndim=2) trajectory coordinates, current timestep: [[x,y,z],[x,y,z],...,[x,y,z]]
        boxlength      .. the box length

    Usage:
        With this function one can unfold 
        the coordinates of a trajectory on the fly, timestep per timestep.

        Example: 
            unfoldBox(xyz_unfold, xyz_traj_prev, xyz_traj_curr, boxlength)
    """
    
    cdef double boxl2 = boxl/2
#    cdef int n = 3*len(xyz_unfold)
    cdef int n = coor_unfold.shape[0], i
    cdef double DX,DY,DZ
    
    for i in prange(n, nogil=True):
        DX = coor_curr[i,0] - coor_prev[i,0]
        DY = coor_curr[i,1] - coor_prev[i,1]
        DZ = coor_curr[i,2] - coor_prev[i,2]
        if fabs(DX) > boxl2:
            DX = DX - sign(DX)*boxl
        if fabs(DY) > boxl2:
            DY = DY - sign(DY)*boxl
        if fabs(DZ) > boxl2:
            DZ = DZ -sign(DZ)*boxl
        coor_unfold[i,0] += DX
        coor_unfold[i,1] += DY
        coor_unfold[i,2] += DZ

@cython.boundscheck(False)
def unfoldOcta(np.ndarray[np.float64_t,ndim=2] xyz_unfold,
           np.ndarray[np.float64_t,ndim=2] xyz_traj_prev,
           np.ndarray[np.float64_t,ndim=2] xyz_traj_curr,
           boxlength):
    """
    unfoldBox(xyz_unfold,xyz_prev,xyz_curr,boxlength)
    
    You have to hand over 3 coordinate sets and the boxlength as arguments: 
    - the in-place-unfolding coordinates (from the previous timestep)
    - the trajectory coordinates from the previous timestep
    - the trajectory coordinates from the current timestep
    - boxlength

    This functions returns the unfolded coordinates of the current timestep.

    Arguments:
        xyz_unfold     .. numpy array (float64, ndim=2) in-place-unfolding coordinate set: [[x,y,z],[x,y,z],...,[x,y,z]]
        xyz_traj_prev  .. numpy array (float64, ndim=2) trajectory coordinates, previous timestep: [[x,y,z],[x,y,z],...,[x,y,z]]
        xyz_traj_curr  .. numpy array (float64, ndim=2) trajectory coordinates, current timestep: [[x,y,z],[x,y,z],...,[x,y,z]]
        boxlength      .. the box length

    Usage:
        With this function one can unfold 
        the coordinates of a trajectory on the fly, timestep per timestep.

        Example: 
            unfoldOcta(xyz_unfold, xyz_traj_prev, xyz_traj_curr, boxlength)
    """
    cdef double boxl = boxlength
    cdef double boxl2 = boxl/2
    cdef double boxl34 = 3*boxl/4
    cdef int n = 3*len(xyz_unfold)
    cdef int i
    cdef double DX,DY,DZ,DD
    cdef np.ndarray[np.float64_t,ndim=2] c_xyz_traj_prev = np.copy(xyz_traj_prev)
    cdef np.ndarray[np.float64_t,ndim=2] c_xyz_traj_curr = np.copy(xyz_traj_curr)
    cdef double *unfold = <double *> xyz_unfold.data
    cdef double *prev = <double *> c_xyz_traj_prev.data
    cdef double *curr = <double *> c_xyz_traj_curr.data

    for i in range(0,n,3):
        DX = curr[i] - prev[i]
        DY = curr[i+1] - prev[i+1]
        DZ = curr[i+2] - prev[i+2]
        DD = fabs(DX) + fabs(DY) + fabs(DZ)
        if DD  > boxl34:
            DX -= sign(DX)*boxl2
            DY -= sign(DY)*boxl2
            DZ -= sign(DZ)*boxl2
        else:
            if fabs(DX) > boxl2:
                DX -= sign(DX)*boxl
            if fabs(DY) > boxl2:
                DY -= sign(DY)*boxl
            if fabs(DZ) > boxl2:
                DZ -= sign(DZ)*boxl
        unfold[i] += DX
        unfold[i+1] += DY
        unfold[i+2] += DZ

@cython.boundscheck(False)
def unfoldCharmmOcta(np.ndarray[np.float64_t,ndim=2] xyz_unfold,
                 np.ndarray[np.float64_t,ndim=2] xyz_traj_prev,
                 np.ndarray[np.float64_t,ndim=2] xyz_traj_curr,
                 np.ndarray[np.float32_t,ndim=1] dimensions):
    """
    unfoldCharmmOcta(xyz_unfold,xyz_prev,xyz_curr,dimensions)
    
    You have to hand over 3 coordinate sets and the dimensions array as arguments: 
    - the in-place-unfolding coordinates (from the previous timestep)
    - the trajectory coordinates from the previous timestep
    - the trajectory coordinates from the current timestep
    - dimensions = matrix with entries for basis vectors

    This functions returns the unfolded coordinates of the current timestep.

    Arguments:
        xyz_unfold     .. numpy array (float64, ndim=2) in-place-unfolding coordinate set: [[x,y,z],[x,y,z],...,[x,y,z]]
        xyz_traj_prev  .. numpy array (float64, ndim=2) trajectory coordinates, previous timestep: [[x,y,z],[x,y,z],...,[x,y,z]]
        xyz_traj_curr  .. numpy array (float64, ndim=2) trajectory coordinates, current timestep: [[x,y,z],[x,y,z],...,[x,y,z]]
        dimensions     .. numpy array (float32, ndim=1) output from u.coord.dimensions

    Usage:
        With this function one can unfold 
        the coordinates of a trajectory on the fly, timestep per timestep.
        Compatible with every box-shape in CHARMM.

        Example: 
            unfoldCharmmOcta(xyz_unfold, xyz_traj_prev, xyz_traj_curr, dimensions)
    """

    cdef double diag = <double> np.round(dimensions[0],4)
    cdef double offdiag = <double> np.round(dimensions[3],4)
    cdef double norm = sqrt(diag*diag+2*offdiag*offdiag)
    cdef double norm2 = norm/2.0, norm22 = norm2*norm2, norm32 = 2*norm/3.0
    cdef double A = diag/norm, B = offdiag/norm, C = 1/sqrt(3)
    cdef double DX, DY, DZ, D, T1, T2, T3
    cdef int n = 3*len(xyz_unfold)
    cdef int i
    cdef double *unfold = <double *> xyz_unfold.data
    cdef double *prev = <double *> xyz_traj_prev.data
    cdef double *curr = <double *> xyz_traj_curr.data

    for i in range(0,n,3):
        DX = curr[i] - prev[i]
        DY = curr[i+1] - prev[i+1]
        DZ = curr[i+2] - prev[i+2]
        D = DX*DX+DY*DY+DZ*DZ
        if D > norm22: # if there was a jump
            T1 = DX*A+DY*B+DZ*B
            T2 = DX*B+DY*A+DZ*B
            T3 = DX*B+DY*B+DZ*A
            if sign(T1) == sign(T2) == sign(T3): # jump across hexagon, non-orthogonal to basis
                DX -= sign(T1)*norm*C
                DY -= sign(T1)*norm*C
                DZ -= sign(T1)*norm*C
            elif  fabs(fabs(T1)-fabs(T2))+fabs(fabs(T1)-fabs(T3))+fabs(fabs(T2)-fabs(T3)) < norm32: # jump across squares, differences in absolute Ts should be ~ 0
                if sign(T1) == sign(T2) or sign(T1)==sign(T3): # if T1 has an equal sign as T2 or T3
                    DX -= sign(T1)*norm*A
                    DY -= sign(T1)*norm*B
                    DZ -= sign(T1)*norm*B
                if sign(T2) == sign(T1) or sign(T2)==sign(T3): # if T2 has an equal sign as T1 or T3
                    DX -= sign(T2)*norm*B
                    DY -= sign(T2)*norm*A
                    DZ -= sign(T2)*norm*B
                if sign(T3) == sign(T1) or sign(T3)==sign(T2): # if T3 has an equal sign as T1 or T2
                    DX -= sign(T3)*norm*B
                    DY -= sign(T3)*norm*B
                    DZ -= sign(T3)*norm*A
            else: # jump across one of three hexagons, orthogonal to a basis vector
                if fabs(T1) > norm2:
                    DX -= sign(T1)*norm*A
                    DY -= sign(T1)*norm*B
                    DZ -= sign(T1)*norm*B
                if fabs(T2) > norm2:
                    DX -= sign(T2)*norm*B
                    DY -= sign(T2)*norm*A
                    DZ -= sign(T2)*norm*B
                if fabs(T3) > norm2:
                    DX -= sign(T3)*norm*B
                    DY -= sign(T3)*norm*B
                    DZ -= sign(T3)*norm*A
        unfold[i] += DX
        unfold[i+1] += DY
        unfold[i+2] += DZ


@cython.boundscheck(False)
def minDistBox(np.ndarray[np.float64_t,ndim=1,mode="c"] aufpunkt,
               np.ndarray[np.float64_t,ndim=2,mode="c"] com,
               np.ndarray[np.float64_t,ndim=2,mode="c"] xyz,
               boxlength, 
               np.ndarray[np.int32_t,ndim=1,mode="c"] atoms_per_residue,
               np.ndarray[np.int32_t,ndim=1,mode="c"] residue_first_atom):
    """
    translateBox(vec, com, xyz, boxlength, atoms_per_residue, residue_first_atom)

    Translate the box by a given vector

    Args:
        aufpunkt           .. vector by which the coordinates are translated
        com                .. centers of mass of all molecules in the box
        xyz                .. coordinates of all atoms
        boxlength          .. edge length of the cubic box
        atoms_per_residue  .. array showing how many atoms each residue contains
        residue_first_atom .. index of the first atom of each residue
    """

    cdef double[3] shift
    cdef double* auf = <double *> aufpunkt.data
    cdef double boxl = <double> boxlength
    cdef int i, j, k, nmol=len(com)
    cdef double *ccom    = <double *> com.data
    cdef double *cxyz    = <double *> xyz.data
    cdef int* apr        = <int *> atoms_per_residue.data
    cdef int* rfa        = <int *> residue_first_atom.data
    cdef int natom = <int> len(xyz)

    for i in range(nmol):
        for j in range(3):
            ccom[i*3+j]-=auf[j]
            shift[j] = boxl*floor(ccom[i*3+j]/boxl + 0.5)
            ccom[i*3+j]-=shift[j]

        for k in range(rfa[i],rfa[i]+apr[i]):
            for j in range(3):
                cxyz[k*3+j]-= shift[j]

                
@cython.boundscheck(False)
def minDistCenterBox(double [:] aufpunkt, double [:,:] com, double [:,:] coor, double boxl, int [:] apr, int [:] rfa):
    """
    translateBox(vec, com, xyz, boxlength, atoms_per_residue, residue_first_atom)

    Translate the box by a given vector

    Args:
        aufpunkt           .. vector by which the coordinates are translated
        com                .. centers of mass of all molecules in the box
        xyz                .. coordinates of all atoms
        boxlength          .. edge length of the cubic box
        atoms_per_residue  .. array showing how many atoms each residue contains
        residue_first_atom .. index of the first atom of each residue
    """

    cdef double[3] shift
    cdef int i, j, k, nmol = com.shape[0], natom = coor.shape[0]

    for i in range(nmol):
        for j in range(3):
            com[i,j] -= aufpunkt[j]
            shift[j] = boxl * floor(com[i,j] / boxl + 0.5)
            com[i,j] -= shift[j]
            
        for k in range(rfa[i], rfa[i]+apr[i]):
            for j in range(3):
                coor[k,j] -= (aufpunkt[j] + shift[j])

@cython.boundscheck(False)
def minVec(double [:] coor1, double [:] coor2, double boxl, double boxl2):
    """
    minVec(xyz1, xyz2, boxl, boxl2)

    Gives the shortes vector pointing from xyz1 to xyz2 considering periodic boundary conditions.
    """
    cdef double [:] delta = np.zeros(3)
    cdef int i

    for i in range(3):
        delta[i] = coor2[i] - coor1[i]
        if fabs(delta[i]) > boxl2:
            delta[i] -= sign(delta[i]) * boxl
    
    return np.asarray(delta)

@cython.boundscheck(False)
def minDist(double [:] coor1, double [:] coor2, double boxl, double boxl2):
    """
    minDist(xyz1, xyz2, boxl, boxl2)

    Gives the shortest distance between xyz1 and xyz2 considering periodic boundary conditions.
    """

    cdef double delta [3]
    cdef int i

    for i in range(3):
        delta[i] = coor2[i] - coor1[i]
        if fabs(delta[i]) > boxl2:
            delta[i] -= sign(delta[i]) * boxl

    return float(sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]))

@cython.boundscheck(False)
def findMinDist(double [:,:] coor, double boxl):
    cdef int n = coor.shape[0], i, j
    cdef double minimum = 1000000.0, dist
    cdef double boxl2 = boxl/2.0

    for i in range(n):
        for j in range(n):
            if i == j: continue
            dist = minDist(coor[i], coor[j], boxl, boxl2)
            if dist < minimum:
                minimum = dist

    return float(minimum)

@cython.boundscheck(False)
def bertholdHorn(double [:,:] coor, double [:,:] coorA, double [:,:] coorB):
    """
    bertholdHorn(coor,coorA,coorB)

    Applies the Berthold-Horn-algorithm to a set of coordinates xyz, inplace.

    Args:
        coor               .. coordinates of all atoms
        coorA              .. template coordinates
        coorB              .. corresponding coordinates of the current frame
    """
    cdef double [:,:] R = np.zeros((3,3))
    cdef double tmpx, tmpy, tmpz
    cdef int i    

    GetRotation(&R[0,0], coorA.shape[0], &coorA[0,0], &coorB[0,0], 0)

    for i in prange(coor.shape[0], nogil=True):
        tmpx = R[0,0] * coor[i,0] + R[0,1] * coor[i,1] + R[0,2] * coor[i,2]
        tmpy = R[1,0] * coor[i,0] + R[1,1] * coor[i,1] + R[1,2] * coor[i,2]
        tmpz = R[2,0] * coor[i,0] + R[2,1] * coor[i,1] + R[2,2] * coor[i,2]
        coor[i,0] = tmpx
        coor[i,1] = tmpy
        coor[i,2] = tmpz
