# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport fabs, sqrt, floor, pow

    
@cython.boundscheck(False)           
def msdMJdecomp(np.ndarray[np.float64_t,ndim=3] coms_cat,
                np.ndarray[np.float64_t,ndim=3] coms_an,
                np.ndarray[np.float64_t,ndim=3] coms_sep,
                int charge,
                maxlen=None):
    """
    msdMJdecomp(coms_cat, coms_an, coms_sep, charge, maxlen=None)

    Calculates part of total autocorrelation, namely <dMj(species,t)dMj(total,t)>.

    Takes three center-of-mass arrays of the whole unfolded trajectory, one for cations, the second for anions,
    the third of the separate species and an integer that specifies the charge of the separate species.

    Usage:
        msdmj = msdMJdecomp(coms_cat, coms_an, coms_sep, charge)
    """

    cdef int n1,n2,n3,m,i,j,k,ch
    cdef double dx,dy,dz,dxs,dys,dzs,result

    ch = charge
    
    n1 = len(coms_cat)
    n2 = len(coms_cat[0])
    n3 = len(coms_sep[0])
    
    if maxlen == None:
        m = n1
    else:
        m = <int> maxlen

    cdef double *cat = <double *> coms_cat.data
    cdef double *an  = <double *> coms_an.data
    cdef double *sep = <double *> coms_sep.data

    cdef np.ndarray[np.float64_t,ndim=2] mj = np.zeros((n1,3),dtype=np.float64)
    cdef double *cmj = <double *> mj.data

    cdef np.ndarray[np.float64_t,ndim=2] mj_sep = np.zeros((n1,3),dtype=np.float64)
    cdef double *cmj_sep = <double *> mj_sep.data

    cdef np.ndarray[np.float64_t,ndim=1] msdmj = np.zeros(m,dtype=np.float64)
    cdef double *msd = <double *> msdmj.data

    cdef np.ndarray[np.int32_t,ndim=1] ctr = np.zeros(m,dtype=np.int32)
    cdef int *c = <int *> ctr.data


    for i in prange(n1,nogil=True):
        for j in range(n2):
            cmj[i*3]   += cat[i*n2*3+3*j]
            cmj[i*3+1] += cat[i*n2*3+3*j+1]
            cmj[i*3+2] += cat[i*n2*3+3*j+2]
            cmj[i*3]   -= an[i*n2*3+3*j]
            cmj[i*3+1] -= an[i*n2*3+3*j+1]
            cmj[i*3+2] -= an[i*n2*3+3*j+2]
            
    for i in prange(n1,nogil=True):
        for j in range(n3):
            cmj_sep[i*3]  += ch*sep[i*n3*3+3*j]
            cmj_sep[i*3+1]+= ch*sep[i*n3*3+3*j+1]
            cmj_sep[i*3+2]+= ch*sep[i*n3*3+3*j+2]


    for i in prange(m,nogil=True):
        for j in range(n1-i):
            dx = cmj[j*3] - cmj[(j+i)*3]
            dy = cmj[j*3+1] - cmj[(j+i)*3+1]
            dz = cmj[j*3+2] - cmj[(j+i)*3+2]

            dxs= cmj_sep[j*3] - cmj_sep[(j+i)*3]
            dys= cmj_sep[j*3+1] - cmj_sep[(j+i)*3+1]
            dzs= cmj_sep[j*3+2] - cmj_sep[(j+i)*3+2]

            result = dx*dxs+dy*dys+dz*dzs
            msd[i]+=result
            c[i]+=1

    for i in range(m):
        msd[i]/=c[i]
    
    return msdmj

    
@cython.boundscheck(False)  
def msdMJcrossterms(np.ndarray[np.float64_t,ndim=3] coms_s1,
                     np.ndarray[np.float64_t,ndim=3] coms_s2,
                     int charge_s1,
                     int charge_s2,
                     maxlen=None):
    """
    msdMJcrossterms(coms_s1, coms_s2, charge_s1, charge_s2, maxlen=None)

    Calculates the MJ^2 correlation function for two arbitrary species.
    Takes two center-of-mass arrays of the whole unfolded trajectory, one for first species, the other for second species.

    Usage:
        msdmj = msdMJ(coms_cat, coms_an)
    """

    cdef int n1,c1,c2,s1_n2,s2_n2,m,i,j,k
    cdef double dx1,dy1,dz1,dx2,dy2,dz2,result


    c1 = charge_s1
    c2 = charge_s2
    
    n1 = len(coms_s1)
    s1_n2 = len(coms_s1[0])
    s2_n2 = len(coms_s2[0])
    if maxlen == None:
        m = n1
    else:
        m = <int> maxlen

    cdef double *s1 = <double *> coms_s1.data
    cdef double *s2  = <double *> coms_s2.data

    cdef np.ndarray[np.float64_t,ndim=2] mj1 = np.zeros((n1,3),dtype=np.float64)
    cdef double *cmj1 = <double *> mj1.data

    cdef np.ndarray[np.float64_t,ndim=2] mj2 = np.zeros((n1,3),dtype=np.float64)
    cdef double *cmj2 = <double *> mj2.data

    cdef np.ndarray[np.float64_t,ndim=1] msdmj = np.zeros(m,dtype=np.float64)
    cdef double *msd = <double *> msdmj.data

    cdef np.ndarray[np.int32_t,ndim=1] ctr = np.zeros(m,dtype=np.int32)
    cdef int *c = <int *> ctr.data

    for i in prange(n1,nogil=True):
        for j in range(s1_n2):
            cmj1[i*3]   += c1*s1[i*s1_n2*3+3*j]
            cmj1[i*3+1] += c1*s1[i*s1_n2*3+3*j+1]
            cmj1[i*3+2] += c1*s1[i*s1_n2*3+3*j+2]
            
    for i in prange(n1,nogil=True):
        for j in range(s2_n2):
            cmj2[i*3]   += c2*s2[i*s2_n2*3+3*j]
            cmj2[i*3+1] += c2*s2[i*s2_n2*3+3*j+1]
            cmj2[i*3+2] += c2*s2[i*s2_n2*3+3*j+2]
        
    for i in prange(m,nogil=True):
        for j in range(n1-i):
            dx1 = cmj1[j*3] - cmj1[(j+i)*3]
            dy1 = cmj1[j*3+1] - cmj1[(j+i)*3+1]
            dz1 = cmj1[j*3+2] - cmj1[(j+i)*3+2]

            dx2 = cmj2[j*3] - cmj2[(j+i)*3]
            dy2 = cmj2[j*3+1] - cmj2[(j+i)*3+1]
            dz2 = cmj2[j*3+2] - cmj2[(j+i)*3+2]
            
            result = dx1*dx2+dy1*dy2+dz1*dz2
            msd[i]+=result
            c[i]+=1

    for i in range(m):
        msd[i]/=c[i]
    
    return msdmj


@cython.boundscheck(False)           
def nonGaussianParameter(np.ndarray[np.float64_t,ndim=3] xyz,
                        n=2,
                        maxlen=None):
    """
    nonGaussianParameter(xyz, n=2, maxlen=None)

    This function takes as input a three-dimensional coordinate array of the molecular centers of mass of a whole unfolded trajectory 
    [timestep, molecule, xyz] and calculates the non-Gaussian Parameter according to Rahman, A. Physical Review 136, A405–A411 (1964).
    n is the "moment" of the distribution and 2 per default. The non-Gaussian parameter describes the deviation of the time-dependent
    molecular displacement from theGaussian distribution and is zero for n>1 if the distribution of the displacements is actually Gaussian. 
    The optional parameter maxlen can be set to limit the length of the resulting MSD. This can be useful
    when dealing with very long trajectories, e.g. maxlen=n/2.

    Usage:
        nonGaussianParameter = nonGaussianParameter(xyz)
    """

    if n < 1:
        raise ValueError("n must be >1")

    cdef double factor, result, dx, dy, dz, res2
    cdef int deg, n1, n2, i, j, k, m, pos

    n1 = len(xyz)
    n2 = len(xyz[0])
    deg = n

    
    if maxlen == None:
        m = n1
    else:
        m = <int> maxlen

    cdef np.ndarray[np.float64_t,ndim=1] msd = np.zeros(m,dtype=np.float64)
    cdef double* msd_data = <double *> msd.data

    cdef np.ndarray[np.float64_t,ndim=1] msddeg = np.zeros(m, dtype=np.float64)
    cdef double* msddeg_data = <double *> msddeg.data

    cdef np.ndarray[np.float64_t,ndim=1] alpha = np.zeros(m, dtype=np.float64)
    cdef double* alpha_data = <double *> alpha.data

    cdef double *data = <double *> xyz.data 

    cdef np.ndarray[np.int32_t,ndim=1] ctr = np.zeros(m,dtype=np.int32)
    cdef int *c = <int *> ctr.data

    # loop over all delta t
    for i in prange(m,nogil=True,schedule=guided):
        # loop over all possible interval start points
        for j in range(n1-i):
            # loop over residues
            for k in range(n2):
                dx = data[j*n2*3+3*k] - data[(j+i)*n2*3+3*k]
                dy = data[j*n2*3+3*k+1] - data[(j+i)*n2*3+3*k+1]
                dz = data[j*n2*3+3*k+2] - data[(j+i)*n2*3+3*k+2]
                result = dx*dx+dy*dy+dz*dz
                msd_data[i] += result
                msddeg_data[i] += result**deg
                c[i]+=1
        
    factor = 1
    for i in range(1,deg+1):
        factor = factor * (2*i+1)
    factor /= 3**deg

    alpha[0] = 0
    for i in range(1,m):
        msd[i] /= <double> ctr[i]
        msddeg[i] /= <double> ctr[i]
        alpha[i] = msddeg[i]/(msd[i]**deg * factor) - 1
        
    return alpha
