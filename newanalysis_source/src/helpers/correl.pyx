# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport fabs, sqrt, floor, pow 

cdef extern from "mod_Correl.h":
    cdef cppclass plan:
        plan(double *data, double *data2, int ncut) except +
    cdef cppclass Correl:
        Correl() except +
        void correl( double* data1, double* data2, unsigned long ncut, 
                     double *ans, int m, int ltc, int lowhigh,plan *p) nogil
        void msd_angvelmf(double *coor, double *msd, int natoms, int maxTime, double dt) nogil
        void xcorrel_angvelmf(double *coor1, double *coor2, double *xcf, int natoms1, int natoms2, int maxTime, double dt) nogil
        void rel_angvelmf(double *coor1, double *coor2, double *wcorr_fun1, double *wcorr_fun2, double *wxcorr_fun, int natoms1, int natoms2, int maxTime, int limTime, double dt) nogil
        void transform( double *data, int ncut, double *ans_re, double *ans_im, int ltc ) nogil

@cython.boundscheck(False)
def correlateParallel(np.ndarray[np.float64_t,ndim=2] data1,
                      np.ndarray[np.float64_t,ndim=2] data2,
                      np.ndarray[np.float64_t,ndim=1] out,
                      ltc=0):
    """
    correlateParallel(data1, data2, out, ltc=0)

    Takes two data sets and calculates column-wise correlation functions in parallel and sums them up afterwards. The result is written into the out array.

    Args:
        data1 .. numpy array, float64, ndim=2
        data2 .. numpy array, float64, ndim=2
        out   .. numpy array, float64, ndim=1
        ltc   .. type of long tail correction used
                 0 = none (default)
                 1 = the average of the time series is subtracted from it before the correlation
                 2 = the result is modified

    Usage example: dipole autocorrelation function
        correlateParallel(dipoles, dipoles, mu0mut)

    NOTE: 
        for this to work, the data arrays have to be organised as follows:
        
        |x1|y1|z1|x2|y2|z2|...|xn|yn|zn
      ---------------------------------
      t1|
      t2|
      . |           ........
      . |
      . |
      tm|

        Each column is the x/y/z component of each particle, each row is a time step.
    """
    cdef long _n = <long> len(data1)
    cdef long ncut = <long> len(data1[0])
    cdef int m=-1, _ltc=0, lowhigh=0
    cdef long i, j

    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] d1=np.ascontiguousarray(data1)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] d2=np.ascontiguousarray(data2)

    cdef double *p_data1 = <double *> d1.data
    cdef double *p_data2 = <double *> d2.data
    cdef double *p_out = <double *> out.data

    cdef Correl *myCorrel = new Correl()
    cdef plan *myplan = new plan(p_data1,p_data2,ncut)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] tmp = np.zeros((_n,ncut),dtype=np.float64)
    cdef double *p_tmp = <double *> tmp.data

    if ltc==1 or ltc==2:
        _ltc = ltc

    for i in prange(_n,nogil=True):
        myCorrel.correl(&(p_data1[i*ncut]),&(p_data2[i*ncut]),<int> ncut,&(p_tmp[i*ncut]),m,_ltc,lowhigh,myplan)

    for i in range(ncut):
        for j in range(_n):
            p_out[i]+=p_tmp[j*ncut+i]

    del myCorrel
    del tmp
    del myplan

def correlate(np.ndarray[np.float64_t,ndim=1] data1,
              np.ndarray[np.float64_t,ndim=1] data2,
              ltc=0):
    """
    correlate(data1, data2, ltc=0)

    Takes two data sets and calculates their correlation function.

    Args:
        data1 .. numpy array, float64, ndim=1
        data2 .. numpy array, float64, ndim=1
        ltc   .. type of long tail correction used
                 0 = none (default)
                 1 = the average of the time series is subtracted from it before the correlation
                 2 = the result is modified

    Usage example:
        result = correlate(data1, data2)
    """
    cdef int ncut = len(data1)
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] ans = np.zeros(ncut,dtype=np.float64)
    cdef int m=-1, _ltc=0, lowhigh=0

    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] d1=np.ascontiguousarray(data1)
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] d2=np.ascontiguousarray(data2)

    cdef double *p_data1 = <double *> d1.data
    cdef double *p_data2 = <double *> d2.data
    cdef double *p_ans = <double *> ans.data

    cdef Correl *myCorrel = new Correl()
    cdef plan *myplan = new plan(p_data1,p_data2,ncut)

    if ltc==1 or ltc==2:
        _ltc = ltc

    myCorrel.correl(p_data1,p_data2,ncut,p_ans,m,_ltc,lowhigh,myplan)
    
    del myCorrel
    del myplan

    return ans

def transform(double [:] data, ltc = 0):
    """
    transform(data, ltc=0)
    """
    cdef int ncut = len(data)
    cdef int nc = ncut + 1

    cdef double [:,:] ans = np.zeros((2,nc), dtype=np.float64)

    cdef Correl *myCorrel = new Correl()

    myCorrel.transform(&data[0], ncut, &ans[0,0], &ans[1,0], <int> ltc)

    return np.asarray(ans)

@cython.boundscheck(False)
def sparsecrosscorrelate(np.ndarray[np.float64_t,ndim=2] pdata,
                         np.ndarray[np.int32_t,ndim=2] pds,
                         int pnshells,
                         maxlen=None,):
    cdef int nmol = len(pdata) # number of molecules
    cdef int n = len(pdata[0]) # number of time steps
    cdef int nds = len(pds[0]) # number of steps in delaunay array
    cdef int maxdt = n
    cdef int nshells = pnshells
    if maxlen != None:
        maxdt = maxlen
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata = np.ascontiguousarray(pdata)
    cdef np.ndarray[np.int32_t,ndim=2,mode="c"] contds = np.ascontiguousarray(pds)
    cdef double *cdata = <double *> contdata.data
    cdef int *cds = <int *> contds.data
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] corr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] ctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef double *ccorr = <double *> corr.data
    cdef double *cctr = <double *> ctr.data
    cdef double value
    cdef int start,dt,mol,shell
    
    for dt in prange(maxdt,nogil=True): # loop over all delta t
        for start in range(n-dt):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                shell = cds[mol*nds+start]-1
                if shell < nshells:
                    value = cdata[mol*n+start]*cdata[mol*n+start+dt]
                    ccorr[maxdt*shell+dt] += value
                    cctr[maxdt*shell+dt] += 1

    for i in range(maxdt*nshells):
        if cctr[i] != 0:
            ccorr[i] /= cctr[i]
        else:
            print 'too sparse'

    return corr


@cython.boundscheck(False)
def sparsefullcorrelate(np.ndarray[np.float64_t,ndim=2] pdata,
                         np.ndarray[np.int32_t,ndim=2] pds,
                         int pnshells,
                         maxlen=None,):
    cdef int nmol = len(pdata) # number of molecules
    cdef int n = len(pdata[0]) # number of time steps
    cdef int nds = len(pds[0]) # number of steps in delaunay array
    cdef int maxdt = n
    cdef int nshells = pnshells+1
    if maxlen != None:
        maxdt = maxlen
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata = np.ascontiguousarray(pdata)
    cdef np.ndarray[np.int32_t,ndim=2,mode="c"] contds = np.ascontiguousarray(pds)
    cdef double *cdata = <double *> contdata.data
    cdef int *cds = <int *> contds.data
    pcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pselfcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pcrosscorr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] corr = pcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfcorr = pselfcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crosscorr = pcrosscorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] ctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crossctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef double *ccorr = <double *> corr.data
    cdef double *cselfcorr = <double *> selfcorr.data
    cdef double *ccrosscorr = <double *> crosscorr.data
    cdef double *cctr = <double *> ctr.data
    cdef double *cselfctr = <double *> selfctr.data
    cdef double *ccrossctr = <double *> crossctr.data
    cdef double value
    cdef int i,start,dt,mol,shellstart,shelldt,idx
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for start in range(n-dt):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                shellstart = cds[mol*nds+start]-1
                shelldt = cds[mol*nds+start+dt]-1
#                if shellstart < nshells:
                value = cdata[mol*n+start]*cdata[mol*n+start+dt]
                idx = maxdt*shellstart+dt
                ccorr[idx] += value
                cctr[idx] += 1
                if shellstart == shelldt:
                    cselfcorr[idx] += value
                    cselfctr[idx] += 1
        
    for i in range(maxdt*nshells):
        if cctr[i] != 0:
            ccorr[i] /= cctr[i]
        else:
            print 'tot too sparse'
        if cctr[i] != 0:
            cselfcorr[i] /= cselfctr[i]
        else:
            print 'self too sparse'
            
    for i in range(maxdt*nshells):
        ccrosscorr[i] = ccorr[i] - cselfcorr[i]
        ccrossctr[i] = cctr[i] - cselfctr[i]
        
    return corr,selfcorr,crosscorr

# @cython.boundscheck(False)
# def sparsefullnobulkcorrelate(np.ndarray[np.float64_t,ndim=2] pdata,
#                               np.ndarray[np.int32_t,ndim=2] pds,
#                               int pnshells,
#                               maxlen=None,):
#     cdef int nmol = len(pdata) # number of molecules
#     cdef int n = len(pdata[0]) # number of time steps
#     cdef int nds = len(pds[0]) # number of steps in delaunay array
#     cdef int maxdt = n
#     cdef int nshells = pnshells
#     if maxlen != None:
#         maxdt = maxlen
#     cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata = np.ascontiguousarray(pdata)
#     cdef np.ndarray[np.int32_t,ndim=2,mode="c"] contds = np.ascontiguousarray(pds)
#     cdef double *cdata = <double *> contdata.data
#     cdef int *cds = <int *> contds.data
#     pcorr = np.zeros((nshells,maxdt),dtype=np.float64)
#     pselfcorr = np.zeros((nshells,maxdt),dtype=np.float64)
#     pcrosscorr = np.zeros((nshells,maxdt),dtype=np.float64)
#     cdef np.ndarray[np.float64_t,ndim=2,mode="c"] corr = pcorr
#     cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfcorr = pselfcorr
#     cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crosscorr = pcrosscorr
#     cdef np.ndarray[np.float64_t,ndim=2,mode="c"] ctr = np.zeros((nshells,maxdt),dtype=np.float64)
#     cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfctr = np.zeros((nshells,maxdt),dtype=np.float64)
#     cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crossctr = np.zeros((nshells,maxdt),dtype=np.float64)
#     cdef double *ccorr = <double *> corr.data
#     cdef double *cselfcorr = <double *> selfcorr.data
#     cdef double *ccrosscorr = <double *> crosscorr.data
#     cdef double *cctr = <double *> ctr.data
#     cdef double *cselfctr = <double *> selfctr.data
#     cdef double *ccrossctr = <double *> crossctr.data
#     cdef double value
#     cdef int i,start,dt,mol,shellstart,shelldt,idx
    
#     for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
#         for start in range(n-dt):     # loop over all possible interval start points
#             for mol in range(nmol):   # loop over molecules/entries
#                 shellstart = cds[mol*nds+start]-1
#                 shelldt = cds[mol*nds+start+dt]-1
#                 if shellstart < nshells:
#                     value = cdata[mol*n+start]*cdata[mol*n+start+dt]
#                     idx = maxdt*shellstart+dt
#                     ccorr[idx] += value
#                     cctr[idx] += 1
#                     if shellstart == shelldt:
#                         cselfcorr[idx] += value
#                         cselfctr[idx] += 1
        
#     for i in range(maxdt*nshells):
#         if cctr[i] != 0:
#             ccorr[i] /= cctr[i]
#         else:
#             print 'tot too sparse'
#         if cctr[i] != 0:
#             cselfcorr[i] /= cselfctr[i]
#         else:
#             print 'self too sparse'
            
#     for i in range(maxdt*nshells):
#         ccrosscorr[i] = ccorr[i] - cselfcorr[i]
#         ccrossctr[i] = cctr[i] - cselfctr[i]
        
#     return corr,selfcorr,crosscorr


@cython.boundscheck(False)
def sparsefullnobulkcorrelate(np.ndarray[np.float64_t,ndim=2] pdata,
                              int pcorrdim,
                              np.ndarray[np.int32_t,ndim=2] pds,
                              int pnshells,
                              maxlen=None,):
    
    cdef long corrdim = <long> pcorrdim
    cdef long nmol = <long> len(pdata)/pcorrdim # number of molecules
    cdef long n = <long> len(pdata[0]) # number of time steps
    cdef long nds = <long> len(pds[0]) # number of steps in delaunay array
    cdef long maxdt = <long> n
    cdef long nshells = <long> pnshells
    if maxlen != None:
        maxdt = maxlen
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata = np.ascontiguousarray(pdata)
    cdef np.ndarray[np.int32_t,ndim=2,mode="c"] contds = np.ascontiguousarray(pds)
    cdef double *cdata = <double *> contdata.data
    cdef int *cds = <int *> contds.data
    pcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pselfcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pcrosscorr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] corr = pcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfcorr = pselfcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crosscorr = pcrosscorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] ctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crossctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef double *ccorr = <double *> corr.data
    cdef double *cselfcorr = <double *> selfcorr.data
    cdef double *ccrosscorr = <double *> crosscorr.data
    cdef double *cctr = <double *> ctr.data
    cdef double *cselfctr = <double *> selfctr.data
    cdef double *ccrossctr = <double *> crossctr.data
    cdef double value
    cdef long i,start,dt,mol,shellstart,shelldt,idx,dim,validx,dsidx
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for start in range(n-dt):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                dsidx = mol*nds+start
                shellstart = <long> cds[dsidx]-1
                if shellstart < nshells:
                    shelldt = <long> cds[dsidx+dt]-1
                    for dim in range(corrdim):
                        validx = mol*corrdim*n+dim*n+start
                        value = cdata[validx]*cdata[validx+dt]
                        idx = maxdt*shellstart+dt
                        ccorr[idx] += value
                        cctr[idx] += 1
                        if shellstart == shelldt:
                            cselfcorr[idx] += value
                            cselfctr[idx] += 1
        
    for i in range(maxdt*nshells):
        if cctr[i] != 0:
            ccorr[i] /= cctr[i]
        else:
            print 'tot too sparse'
        if cctr[i] != 0:
            cselfcorr[i] /= cselfctr[i]
        else:
            print 'self too sparse'
            
    for i in range(maxdt*nshells):
        ccrosscorr[i] = ccorr[i] - cselfcorr[i]
        ccrossctr[i] = cctr[i] - cselfctr[i]
        
    return corr,selfcorr,crosscorr

@cython.boundscheck(False)
def rotationMatrixShellCorrelateDist(double [:,:,:,:] rotTs, int [:,:] ds, long nshells, long maxdt, long startingpoints):
    cdef long nmol = <long> len(rotTs) # number of molecules
    cdef long n = <long> len(rotTs[0]) # number of time steps
#    cdef long nds = <long> len(ds) # number of steps in delaunay array
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef double [:,:,:,:] corr = np.zeros((6,nshells,maxdt,20)) # 3x l={1,2} rotautocorr+ 3x l={1,2} rotcrosscorr = 12
    cdef double [:,:] ctr = np.zeros((nshells,maxdt))
    cdef double t1,t2,t3,t4,t5,t6
    cdef long i,j,k,start,point,dt,mol,shellstart,shelldt
    
    for dt in prange(maxdt,nogil=True, schedule=dynamic): # loop over all delta t
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shellstart = ds[mol,start]-1
                if shellstart < nshells:
                    
                    ctr[shellstart,dt] += 1

                    t1 = ((rotTs[mol,start,0,0]*rotTs[mol,start+dt,0,0]+rotTs[mol,start,0,1]*rotTs[mol,start+dt,0,1]+rotTs[mol,start,0,2]*rotTs[mol,start+dt,0,2])+1.0)*9.99999999999
                    t2 = ((rotTs[mol,start,1,0]*rotTs[mol,start+dt,1,0]+rotTs[mol,start,1,1]*rotTs[mol,start+dt,1,1]+rotTs[mol,start,1,2]*rotTs[mol,start+dt,1,2])+1.0)*9.99999999999
                    t3 = ((rotTs[mol,start,2,0]*rotTs[mol,start+dt,2,0]+rotTs[mol,start,2,1]*rotTs[mol,start+dt,2,1]+rotTs[mol,start,2,2]*rotTs[mol,start+dt,2,2])+1.0)*9.99999999999
                    t4 = ((rotTs[mol,start,0,0]*rotTs[mol,start+dt,1,0]+rotTs[mol,start,0,1]*rotTs[mol,start+dt,1,1]+rotTs[mol,start,0,2]*rotTs[mol,start+dt,1,2])+1.0)*9.99999999999
                    t5 = ((rotTs[mol,start,0,0]*rotTs[mol,start+dt,2,0]+rotTs[mol,start,0,1]*rotTs[mol,start+dt,2,1]+rotTs[mol,start,0,2]*rotTs[mol,start+dt,2,2])+1.0)*9.99999999999
                    t6 = ((rotTs[mol,start,1,0]*rotTs[mol,start+dt,2,0]+rotTs[mol,start,1,1]*rotTs[mol,start+dt,2,1]+rotTs[mol,start,1,2]*rotTs[mol,start+dt,2,2])+1.0)*9.99999999999
                    corr[0,shellstart,dt,<int> t1] += 1.0 # l=1
                    corr[1,shellstart,dt,<int> t2] += 1.0
                    corr[2,shellstart,dt,<int> t3] += 1.0
                    corr[3,shellstart,dt,<int> t4] += 1.0
                    corr[4,shellstart,dt,<int> t5] += 1.0
                    corr[5,shellstart,dt,<int> t6] += 1.0

    for i in range(nshells):
        for j in range(maxdt):
            if ctr[i,j] != 0:
                for k in range(6):
                    for l in range(20):
                        corr[k,i,j,l] /= ctr[i,j]

    
    return corr



@cython.boundscheck(False)
def rotationMatrixShellCorrelate(double [:,:,:,:] rotTs, int [:,:] ds, long nshells, long maxdt, long startingpoints):
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
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shellstart = ds[mol,start]-1
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
                    
                    shelldt =  ds[mol,start+dt]-1
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
def rotationMatrixShellCorrelateTot(double [:,:,:,:] rotTs, int [:,:] ds, long nshells, long maxdt, long startingpoints):
    cdef long nmol = <long> len(rotTs) # number of molecules
    cdef long n = <long> len(rotTs[0]) # number of time steps
#    cdef long nds = <long> len(ds) # number of steps in delaunay array
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef double [:,:,:] corr = np.zeros((12,nshells,maxdt)) # 3x l={1,2} rotautocorr+ 3x l={1,2} rotcrosscorr = 12
    cdef double [:,:] ctr = np.zeros((nshells,maxdt)) # counter array of correlation entries
    cdef double t1,t2,t3,t4,t5,t6
    cdef long i,j,k,start,point,dt,mol,shellstart,shelldt
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shellstart = ds[mol,start]-1
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

    for i in range(nshells):
        for j in range(maxdt):
            if ctr[i,j] != 0:
                for k in range(12):
                    corr[k,i,j] /= ctr[i,j]
            else:
                print 'tot too sparse'
        
    return corr

@cython.boundscheck(False)
def rotationMatrixShellCorrelateTotParts(double [:,:,:] rotTs1,double [:,:,:] rotTs2,double [:,:,:] rotTs3, int [:,:] ds, long nshells, long maxdt, long startingpoints):
    cdef long nmol = <long> len(rotTs1) # number of molecules
    cdef long n = <long> len(rotTs1[0]) # number of time steps
#    cdef long nds = <long> len(ds) # number of steps in delaunay array
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef double [:,:,:] corr = np.zeros((12,nshells,maxdt)) # 3x l={1,2} rotautocorr+ 3x l={1,2} rotcrosscorr = 12
    cdef double [:,:] ctr = np.zeros((nshells,maxdt)) # counter array of correlation entries
    cdef double t1,t2,t3,t4,t5,t6
    cdef long i,j,k,start,point,dt,mol,shellstart,shelldt
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for point in range(startingpoints):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = point*startskip
                shellstart = ds[mol,start]-1
                if shellstart < nshells:
                    ctr[shellstart,dt] += 1
                    
                    t1,t2,t3,t4,t5,t6 = 0,0,0,0,0,0

                    t1 += rotTs1[mol,start,0]*rotTs1[mol,start+dt,0]
                    t2 += rotTs1[mol,start,1]*rotTs1[mol,start+dt,1]
                    t3 += rotTs1[mol,start,2]*rotTs1[mol,start+dt,2]
                    t4 += rotTs1[mol,start,0]*rotTs1[mol,start+dt,1]
                    t5 += rotTs1[mol,start,0]*rotTs1[mol,start+dt,2]
                    t6 += rotTs1[mol,start,1]*rotTs1[mol,start+dt,2]
                    t1 += rotTs2[mol,start,0]*rotTs2[mol,start+dt,0]
                    t2 += rotTs2[mol,start,1]*rotTs2[mol,start+dt,1]
                    t3 += rotTs2[mol,start,2]*rotTs2[mol,start+dt,2]
                    t4 += rotTs2[mol,start,0]*rotTs2[mol,start+dt,1]
                    t5 += rotTs2[mol,start,0]*rotTs2[mol,start+dt,2]
                    t6 += rotTs2[mol,start,1]*rotTs2[mol,start+dt,2]
                    t1 += rotTs3[mol,start,0]*rotTs3[mol,start+dt,0]
                    t2 += rotTs3[mol,start,1]*rotTs3[mol,start+dt,1]
                    t3 += rotTs3[mol,start,2]*rotTs3[mol,start+dt,2]
                    t4 += rotTs3[mol,start,0]*rotTs3[mol,start+dt,1]
                    t5 += rotTs3[mol,start,0]*rotTs3[mol,start+dt,2]
                    t6 += rotTs3[mol,start,1]*rotTs3[mol,start+dt,2]

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

    for i in range(nshells):
        for j in range(maxdt):
            if ctr[i,j] != 0:
                for k in range(12):
                    corr[k,i,j] /= ctr[i,j]
            else:
                print 'tot too sparse'
        
    return corr


@cython.boundscheck(False)
def vectorRotation(double [:,:] rotTs, int datapoints):
#    cdef long datapoints = <long> len(rotTs) 
    cdef double [:,:] corr = np.zeros((2,datapoints)) 
    cdef double t
    cdef long i,j,k,start,point,dt,mol,shellstart,shelldt
    
    for dt in prange(datapoints,nogil=True,schedule=dynamic): # loop over all delta t
        for start in range(datapoints-dt):     # loop over all possible interval start points
            t = 0
            for k in range(3): # skalar produkt
                t += rotTs[start,k]*rotTs[start+dt,k]
            corr[0,dt] += t # l=1
            corr[1,dt] += 1.5*t*t-0.5 # l=2
    for i in range(datapoints):
        corr[0,i] = corr[0,i] / (datapoints-i)
        corr[1,i] = corr[1,i] / (datapoints-i)
    return corr
    
@cython.boundscheck(False)
def sparsefullnobulkthincorrelate(np.ndarray[np.float64_t,ndim=2] pdata,
                              int pcorrdim,
                              np.ndarray[np.int32_t,ndim=2] pds,
                              int pnshells,
                              maxlen=None,
                              startingpoints=None):
    
    cdef long corrdim = <long> pcorrdim
    cdef long nmol = <long> len(pdata)/pcorrdim # number of molecules
    cdef long n = <long> len(pdata[0]) # number of time steps
    cdef long nds = <long> len(pds[0]) # number of steps in delaunay array
    cdef long maxdt = <long> maxlen
    cdef long nshells = <long> pnshells
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef long startmax = (n-maxdt)/startskip
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata = np.ascontiguousarray(pdata)
    cdef np.ndarray[np.int32_t,ndim=2,mode="c"] contds = np.ascontiguousarray(pds)
    cdef double *cdata = <double *> contdata.data
    cdef int *cds = <int *> contds.data
    pcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pselfcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pcrosscorr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] corr = pcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfcorr = pselfcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crosscorr = pcrosscorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] ctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crossctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef double *ccorr = <double *> corr.data
    cdef double *cselfcorr = <double *> selfcorr.data
    cdef double *ccrosscorr = <double *> crosscorr.data
    cdef double *cctr = <double *> ctr.data
    cdef double *cselfctr = <double *> selfctr.data
    cdef double *ccrossctr = <double *> crossctr.data
    cdef double value
    cdef long i,start,dt,mol,shellstart,shelldt,idx,dim,validx,dsidx,istart
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for istart in range(startmax):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = istart*startskip
                dsidx = mol*nds+start
                shellstart = <long> cds[dsidx]-1
                if shellstart < nshells:
                    shelldt = <long> cds[dsidx+dt]-1
                    for dim in range(corrdim):
                        validx = mol*corrdim*n+dim*n+start
                        value = cdata[validx]*cdata[validx+dt]
                        idx = maxdt*shellstart+dt
                        ccorr[idx] += value
                        cctr[idx] += 1
                        if shellstart == shelldt:
                            cselfcorr[idx] += value
                            cselfctr[idx] += 1
        
    for i in range(maxdt*nshells):
        if cctr[i] != 0:
            ccorr[i] /= cctr[i]
        else:
            print 'tot too sparse'
        if cselfctr[i] != 0:
            cselfcorr[i] /= cselfctr[i]
        else:
            print 'self too sparse'
            
    for i in range(maxdt*nshells):
        ccrosscorr[i] = ccorr[i] - cselfcorr[i]
        ccrossctr[i] = cctr[i] - cselfctr[i]
        
    return corr,selfcorr,crosscorr


@cython.boundscheck(False)
def sparsefullnobulkthingreenkubo(np.ndarray[np.float64_t,ndim=2] pdata,
                              int pcorrdim,
                              np.ndarray[np.int32_t,ndim=2] pds,
                              int pnshells,
                              maxlen=None,
                              startingpoints=None):
    
    cdef long corrdim = <long> pcorrdim
    cdef long nmol = <long> len(pdata)/pcorrdim # number of molecules
    cdef long n = <long> len(pdata[0]) # number of time steps
    cdef long nds = <long> len(pds[0]) # number of steps in delaunay array
    cdef long maxdt = <long> maxlen
    cdef long nshells = <long> pnshells
    cdef long startskip = <long> (n-maxdt)/startingpoints
    cdef long startmax = (n-maxdt)/startskip
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata = np.ascontiguousarray(pdata)
    cdef np.ndarray[np.int32_t,ndim=2,mode="c"] contds = np.ascontiguousarray(pds)
    cdef double *cdata = <double *> contdata.data
    cdef int *cds = <int *> contds.data
    pcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pselfcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pcrosscorr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] corr = pcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfcorr = pselfcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crosscorr = pcrosscorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] ctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crossctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef double *ccorr = <double *> corr.data
    cdef double *cselfcorr = <double *> selfcorr.data
    cdef double *ccrosscorr = <double *> crosscorr.data
    cdef double *cctr = <double *> ctr.data
    cdef double *cselfctr = <double *> selfctr.data
    cdef double *ccrossctr = <double *> crossctr.data
    cdef double value,tempvalue
    cdef long i,start,dt,mol,shellstart,shelldt,idx,dim,validx,dsidx,istart
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for istart in range(startmax):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                start = istart*startskip
                dsidx = mol*nds+start
                shellstart = <long> cds[dsidx]-1
                if shellstart < nshells:
                    shelldt = <long> cds[dsidx+dt]-1
                    for dim in range(corrdim):
                        validx = mol*corrdim*n+dim*n+start
                        tempvalue = cdata[validx+dt]-cdata[validx]
                        value =  tempvalue*tempvalue
                        idx = maxdt*shellstart+dt
                        ccorr[idx] += value
                        cctr[idx] += 1
                        if shellstart == shelldt:
                            cselfcorr[idx] += value
                            cselfctr[idx] += 1
        
    for i in range(maxdt*nshells):
        if cctr[i] != 0:
            ccorr[i] /= cctr[i]
        else:
            print 'tot too sparse'
        if cctr[i] != 0:
            cselfcorr[i] /= cselfctr[i]
        else:
            print 'self too sparse'
            
    for i in range(maxdt*nshells):
        ccrosscorr[i] = ccorr[i] - cselfcorr[i]
        ccrossctr[i] = cctr[i] - cselfctr[i]
        
    return corr,selfcorr,crosscorr



@cython.boundscheck(False)
def sparsefullcrosscorrelate(np.ndarray[np.float64_t,ndim=2] pdata1,
                             np.ndarray[np.float64_t,ndim=2] pdata2,
                         np.ndarray[np.int32_t,ndim=2] pds,
                         int pnshells,
                         maxlen=None,):
    cdef int nmol = len(pdata1) # number of molecules
    cdef int n = len(pdata1[0]) # number of time steps
    cdef int nds = len(pds[0]) # number of steps in delaunay array
    cdef int maxdt = n
    cdef int nshells = pnshells+1
    if maxlen != None:
        maxdt = maxlen
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata1 = np.ascontiguousarray(pdata1)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata2 = np.ascontiguousarray(pdata2)
    cdef np.ndarray[np.int32_t,ndim=2,mode="c"] contds = np.ascontiguousarray(pds)
    cdef double *cdata1 = <double *> contdata1.data
    cdef double *cdata2 = <double *> contdata2.data
    cdef int *cds = <int *> contds.data
    pcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pselfcorr = np.zeros((nshells,maxdt),dtype=np.float64)
    pcrosscorr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] corr = pcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfcorr = pselfcorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crosscorr = pcrosscorr
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] ctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] selfctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] crossctr = np.zeros((nshells,maxdt),dtype=np.float64)
    cdef double *ccorr = <double *> corr.data
    cdef double *cselfcorr = <double *> selfcorr.data
    cdef double *ccrosscorr = <double *> crosscorr.data
    cdef double *cctr = <double *> ctr.data
    cdef double *cselfctr = <double *> selfctr.data
    cdef double *ccrossctr = <double *> crossctr.data
    cdef double value
    cdef int i,start,dt,mol,shellstart,shelldt,idx
    
    for dt in prange(maxdt,nogil=True,schedule=dynamic): # loop over all delta t
        for start in range(n-dt):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                shellstart = cds[mol*nds+start]-1
                shelldt = cds[mol*nds+start+dt]-1
#                if shellstart < nshells:
                value = cdata1[mol*n+start]*cdata2[mol*n+start+dt]
                idx = maxdt*shellstart+dt
                ccorr[idx] += value
                cctr[idx] += 1
                if shellstart == shelldt:
                    cselfcorr[idx] += value
                    cselfctr[idx] += 1
        
    for i in range(maxdt*nshells):
        if cctr[i] != 0:
            ccorr[i] /= cctr[i]
        else:
            print 'tot too sparse'
        if cctr[i] != 0:
            cselfcorr[i] /= cselfctr[i]
        else:
            print 'self too sparse'
            
    for i in range(maxdt*nshells):
        ccrosscorr[i] = ccorr[i] - cselfcorr[i]
        ccrossctr[i] = cctr[i] - cselfctr[i]
        
    return corr,selfcorr,crosscorr


@cython.boundscheck(False)
def sparsecorrelate(np.ndarray[np.float64_t,ndim=2] pdata,
                    maxlen=None):
    cdef int nmol = len(pdata) # number of molecules
    cdef int n = len(pdata[0]) # number of time steps
    cdef int maxdt = n
    if maxlen != None:
        maxdt = maxlen
    cdef np.ndarray[np.float64_t,ndim=2,mode="c"] contdata = np.ascontiguousarray(pdata)
    cdef double *cdata = <double *> contdata.data
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] corr = np.zeros(maxdt,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1,mode="c"] ctr = np.zeros(maxdt,dtype=np.float64)
    cdef double *ccorr = <double *> corr.data
    cdef double *cctr = <double *> ctr.data
    cdef double value
    cdef int start,dt,mol
    
    for dt in prange(maxdt,nogil=True): # loop over all delta t
        for start in range(n-dt):     # loop over all possible interval start points
            for mol in range(nmol):   # loop over molecules/entries
                value = cdata[mol*n+start]*cdata[mol*n+start+dt]
                if value != 0:
                    ccorr[dt] += value
                    cctr[dt] += 1

                
    for i in range(maxdt):
        if cctr[i] != 0:
            ccorr[i] /= cctr[i]
        else:
            print 'too sparse'

    return corr

@cython.boundscheck(False)
def calcAngularDisplacement(np.ndarray[np.float64_t,ndim=3] coor_ts, delta):

    cdef int maxTime = <int> coor_ts.shape[0]
    cdef int natoms = <int> coor_ts.shape[1]
    cdef double dt = <double> delta

    cdef np.ndarray[np.float64_t,ndim=2] msd = np.zeros((maxTime,3))

    cdef Correl *myCorrel = new Correl()
    
    myCorrel.msd_angvelmf(<double *> coor_ts.data, <double *> msd.data, natoms, maxTime, dt)

    del myCorrel

    return msd

@cython.boundscheck(False)
def calcAngularVelocityXCorrel(np.ndarray[np.float64_t,ndim=3] coor1_ts,
                               np.ndarray[np.float64_t,ndim=3] coor2_ts,
                               delta):

    cdef int maxTime = <int> coor1_ts.shape[0]
    cdef int natoms1 = <int> coor1_ts.shape[1]
    cdef int natoms2 = <int> coor2_ts.shape[1]
    cdef double dt = <double> delta

    cdef np.ndarray[np.float64_t,ndim=2] xcf = np.zeros((maxTime,3))

    cdef Correl *myCorrel = new Correl()
    
    myCorrel.xcorrel_angvelmf(<double *> coor1_ts.data, <double *> coor2_ts.data, <double *> xcf.data, natoms1, natoms2, maxTime, dt)

    del myCorrel

    return xcf

@cython.boundscheck(False)
def calcRelativeAngDisp(np.ndarray[np.float64_t,ndim=3] coor1_ts,
                        np.ndarray[np.float64_t,ndim=3] coor2_ts,
                        delta, ncut):

    cdef int maxTime = <int> coor1_ts.shape[0]
    cdef int natoms1 = <int> coor1_ts.shape[1]
    cdef int natoms2 = <int> coor2_ts.shape[1]
    cdef double dt = <double> delta

    cdef np.ndarray[np.float64_t,ndim=1] wcorr1 = np.zeros(ncut)
    cdef np.ndarray[np.float64_t,ndim=1] wcorr2 = np.zeros(ncut)
    cdef np.ndarray[np.float64_t,ndim=1] wxcorr = np.zeros(ncut)

    cdef Correl *myCorrel = new Correl()
    
    myCorrel.rel_angvelmf(<double *> coor1_ts.data, <double *> coor2_ts.data, <double *> wcorr1.data, <double *> wcorr2.data, <double *> wxcorr.data, natoms1, natoms2, maxTime, <int> ncut, dt)

    del myCorrel

    return wcorr1, wcorr2, wxcorr


