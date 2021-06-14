# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport fabs, sqrt, floor, pow

def unfoldTraj(np.ndarray[np.float64_t,ndim=3] xyz,
               boxlength):
    """
    unfoldTraj(xyz,boxlength)

    Arguments:
        xyz       .. numpy array (float64, ndim=3) of the centers-of-mass of the whole trajectory [timestep, molecule, xyz]
        boxlength .. the box length

    Usage:
        This function unfolds a trajectory as a whole.

        Example: 
            com = unfoldTraj(com, boxlength)
    """
    cdef double boxl = boxlength
    cdef double b2 = boxl/2.0
    cdef int n = len(xyz[0])
    cdef int nframes = len(xyz)
    cdef int i, j, k
    cdef double d
    cdef np.ndarray[np.float64_t,ndim=2] xyz_tmp = np.copy(xyz[0])
    cdef double *data = <double *> xyz.data
    cdef double *data_tmp = <double *> xyz_tmp.data

    for i in range(1,nframes):
        for j in range(n):            
            for k in range(3):
                d = data[i*n*3+j*3+k] - data_tmp[j*3+k]
                if d <= -b2:
                    d+=boxl
                elif d > b2:
                    d-=boxl
                data_tmp[j*3+k] = data[i*n*3+j*3+k]
                data[i*n*3+j*3+k] = data[(i-1)*n*3+j*3+k] + d

    return xyz

@cython.boundscheck(False)           
def msdCOMvHnG(np.ndarray[np.float64_t,ndim=3] xyz,
               maxlen=None, vh_maxdist=200.0):
    """
    msdCOMvHnG(xyz, maxlen=None, vh_maxdist=200.0)

    This function calculates the single-particle mean square displacement, the van Hove function and the non-Gaussian parameter 
    simultaneously.

    Args:
        xyz        .. unfolded centers of mass of a whole trajectory, as yielded from unfoldTraj()
        maxlen     .. maximum length of the sliding window
        vh_maxdist .. maximum distance to consider for the van Hove function

    Usage:
        msd, van_hove, non_gaussian = msdCOMvHnG(xyz)
    """
    cdef double tmp, result, dx, dy, dz, res2
    cdef int n1, n2, i, j, k, m, pos, idx1, idx2, vhmax = vh_maxdist*10

    n1 = len(xyz)
    n2 = len(xyz[0])

    if maxlen == None:
        m = n1
    else:
        m = <int> maxlen

    cdef np.ndarray[np.float64_t,ndim=1] msd = np.zeros(m,dtype=np.float64)
    cdef double* msd_data = <double *> msd.data

    cdef double *data = <double *> xyz.data 

    cdef np.ndarray[np.float64_t,ndim=2] van_hove = np.zeros((m,vhmax),dtype=np.float64)
    cdef double *vh = <double *> van_hove.data

    cdef np.ndarray[np.float64_t,ndim=1] non_gaussian = np.zeros(m,dtype=np.float64)
    cdef double *ng = <double *> non_gaussian.data

    cdef np.ndarray[np.int32_t,ndim=1] ctr = np.zeros(m,dtype=np.int32)
    cdef int *c = <int *> ctr.data
    
    # loop over all delta t
    for i in prange(m,nogil=True,schedule=guided):
        # loop over all possible interval start points
        for j in range(n1-i):
            # loop over residues
            for k in range(n2):
                idx1 = j*n2*3+3*k
                idx2 = (j+i)*n2*3+3*k
                dx = data[idx1]   - data[idx2]
                dy = data[idx1+1] - data[idx2+1]
                dz = data[idx1+2] - data[idx2+2]
                result = dx*dx+dy*dy+dz*dz
                res2 = result*result
                pos= <int> floor(sqrt(result)*10.0)
                if pos<vhmax:
                    vh[i*vhmax+pos]+=1
                msd_data[i]+=result
                ng[i]+=res2
                c[i]+=1
    
    for i in range(m):
        msd[i] /= <double> ctr[i]
        van_hove[i] /= <double> ctr[i]
        if floor(msd[i]) != 0:
            ng[i] = 0.6 * (ng[i]/<double>ctr[i]) / (msd[i]*msd[i]) - 1.0
        
    return msd, van_hove, non_gaussian

@cython.boundscheck(False)           
def msdCOM(np.ndarray[np.float64_t,ndim=3] xyz,
           maxlen=None):
    """
    msdCOM(xyz, maxlen=None)

    This function takes as input a three-dimensional coordinate array of the molecular centers of mass of a whole unfolded trajectory 
    [timestep, molecule, xyz]. The optional parameter maxlen can be set to limit the length of the resulting MSD. This can be useful
    when dealing with very long trajectories, e.g. maxlen=n/2.

    Usage:
        msd = msdCOM(xyz)
    """
    cdef double tmp, result, dx, dy, dz, res2
    cdef int n1, n2, i, j, k, m, pos

    n1 = len(xyz)
    n2 = len(xyz[0])

    if maxlen == None:
        m = n1
    else:
        m = <int> maxlen

    cdef np.ndarray[np.float64_t,ndim=1] msd = np.zeros(m,dtype=np.float64)
    cdef double* msd_data = <double *> msd.data

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
                c[i]+=1
    
    for i in range(m):
        msd[i] /= <double> ctr[i]
        
    return msd
   
@cython.boundscheck(False)           
def pairDisplacement(np.ndarray[np.float64_t,ndim=3] xyz,
                      np.ndarray[np.float64_t,ndim=3] xyz2,
                      maxlen=None):
    """
    pairDisplacement(xyz1, xyz2, maxlen=None)

    Takes two unfolded center of mass coordinate sets. maxlen can be set to cap the length of the pair displacement.

    Usage:
        pair_displacement = pairDisplacement(xyz1, xyz2)
    """
    cdef double tmp, result, dx, dy, dz, res2, dx2, dy2, dz2, dx3, dy3, dz3
    cdef int n1, n2, i, j, k, l, irel, m, nstep, idx1, idx2, idx3, idx4

    n1 = len(xyz)
    n2 = len(xyz[0])

    if maxlen == None:
        m = n1
    else:
        m = <int> maxlen

    nstep=<int> floor(m/10.)

    cdef bint diag=True
    if (xyz[0][0] == xyz2[0][0]).all() and (xyz[-1][0] == xyz2[-1][0]).all():
        diag = False

    # msd contains pair cross term, msd2 contains pair self term
    cdef np.ndarray[np.float64_t,ndim=1] msd = np.zeros(m,dtype=np.float64)
    cdef double* msd_data = <double *> msd.data
    cdef double *data = <double *> xyz.data
    cdef double *data2 = <double *> xyz2.data

    cdef np.ndarray[np.int32_t,ndim=1] ctr = np.zeros(m,dtype=np.int32)
    cdef int *c = <int *> ctr.data
    
    # loop over all starting points
    for j in range(0,n1-m,nstep):
        idx1=j*n2*3
        # loop over all delta t
        for i in prange(j,j+m,nogil=True,schedule=guided):
            idx2=i*n2*3
            irel=i-j
            # loop over residues
            for k in range(n2):
                idx3=3*k
                for l in range(n2):
                    if k == l and not diag:
                        continue

                    idx4=3*l
                    dx = data[idx2+idx3] - data[idx1+idx3]
                    dy = data[idx2+idx3+1] - data[idx1+idx3+1]
                    dz = data[idx2+idx3+2] - data[idx1+idx3+2]
                    dx2 = data2[idx2+idx4] - data2[idx1+idx4]
                    dy2 = data2[idx2+idx4+1] - data2[idx1+idx4+1]
                    dz2 = data2[idx2+idx4+2] - data2[idx1+idx4+2]
                    dx3 = dx-dx2
                    dy3 = dy-dy2
                    dz3 = dz-dz2
                    result = dx3*dx3+dy3*dy3+dz3*dz3
                    msd_data[irel]+=result
                    c[irel]+=1
    
    for i in range(m):
        msd[i]/=ctr[i]
        
    return msd

@cython.boundscheck(False)           
def msdMJ(np.ndarray[np.float64_t,ndim=3] coms_cat,
          np.ndarray[np.float64_t,ndim=3] coms_an,
          maxlen=None,
          charge_cat = None,
          charge_an = None):
    """
    msdMJ(coms_cat, coms_an, maxlen=None, charge_cat=None, charge_an=None)

    Takes two center-of-mass arrays of the whole unfolded trajectory, one for cations, the other for anions.

    Usage:
        msdmj = msdMJ(coms_cat, coms_an)
    """

    cdef int n1,n2,m,i,j,k,ccat,can
    cdef double dx,dy,dz,result

    n1 = len(coms_cat)
    n2 = len(coms_cat[0])
    if maxlen == None:
        m = n1
    else:
        m = <int> maxlen
    if charge_cat is None:
        ccat = 1
    else:
        ccat = <int> charge_cat

    if charge_an is None:
        can = -1
    else:
        can = <int> charge_an

    cdef double *cat = <double *> coms_cat.data
    cdef double *an  = <double *> coms_an.data

    cdef np.ndarray[np.float64_t,ndim=2] mj = np.zeros((n1,3),dtype=np.float64)
    cdef double *cmj = <double *> mj.data

    cdef np.ndarray[np.float64_t,ndim=1] msdmj = np.zeros(m,dtype=np.float64)
    cdef double *msd = <double *> msdmj.data

    cdef np.ndarray[np.int32_t,ndim=1] ctr = np.zeros(m,dtype=np.int32)
    cdef int *c = <int *> ctr.data

    for i in prange(n1,nogil=True):
        for j in range(n2):
            cmj[i*3]   += ccat*cat[i*n2*3+3*j]
            cmj[i*3+1] += ccat*cat[i*n2*3+3*j+1]
            cmj[i*3+2] += ccat*cat[i*n2*3+3*j+2]
            cmj[i*3]   += can*an[i*n2*3+3*j]
            cmj[i*3+1] += can*an[i*n2*3+3*j+1]
            cmj[i*3+2] += can*an[i*n2*3+3*j+2]
        
    for i in prange(m,nogil=True):
        for j in range(n1-i):
            dx = cmj[j*3] - cmj[(j+i)*3]
            dy = cmj[j*3+1] - cmj[(j+i)*3+1]
            dz = cmj[j*3+2] - cmj[(j+i)*3+2]
            result = dx*dx+dy*dy+dz*dz
            msd[i]+=result
            c[i]+=1

    for i in range(m):
        msd[i]/=c[i]
    
    return msdmj


@cython.boundscheck(False)           
def msdMJCharges(np.ndarray[np.float64_t,ndim=3] coms_cat,
          np.ndarray[np.float64_t,ndim=3] coms_an,
          maxlen=None,
          charge_cat=None,
          charge_an=None):
    """
    msdMJCharges(coms_cat, coms_an, maxlen=None, charge_cat=None, charge_an=None)

    Takes two center-of-mass arrays of the whole unfolded trajectory, one for cations, the other for anions.
    Multiplies CoM by charge. Cations and Anions can have different n_residues

    Usage:
        msdmj = msdMJCharges(coms_cat, coms_an)
    """

    cdef int n1,n2,m,i,j,k,c1,c2,n4,upper,lower
    cdef double dx,dy,dz,result
    cdef double *shorter
    cdef double *longer

    n1 = len(coms_cat) # n_frames
    n2 = len(coms_cat[0]) # n_residues
    n4 = len(coms_an[0])
    if charge_cat is None:
        ccat = 1
    else:
        ccat = <int> charge_cat

    if charge_an is None:
        can = -1
    else:
        can = <int> charge_an

    if maxlen is None:
        m = n1
    else:
        m = <int> maxlen

    if n2 > n4:
        shorter = <double *> coms_an.data
        longer = <double *> coms_cat.data
        upper = n2
        lower = n4
        c1 = can
        c2 = ccat
    elif n2 < n4:
        shorter = <double *> coms_cat.data
        longer = <double *> coms_an.data
        upper = n4
        lower = n2
        c2 = can
        c1 = ccat
    elif n2 == n4:
        print("WARNING: same number of cation/anion species. Use msdMJ instead.")

    cdef np.ndarray[np.float64_t,ndim=2] mj = np.zeros((n1,3),dtype=np.float64)
    cdef double *cmj = <double *> mj.data

    cdef np.ndarray[np.float64_t,ndim=1] msdmj = np.zeros(m,dtype=np.float64)
    cdef double *msd = <double *> msdmj.data

    cdef np.ndarray[np.int32_t,ndim=1] ctr = np.zeros(m,dtype=np.int32)
    cdef int *c = <int *> ctr.data

    for i in prange(n1,nogil=True):
        for j in range(lower):
            cmj[i*3]   += c1*(shorter[i*lower*3+3*j])
            cmj[i*3+1] += c1*(shorter[i*lower*3+3*j+1])
            cmj[i*3+2] += c1*(shorter[i*lower*3+3*j+2])
        for k in range(upper):
            cmj[i*3]   += c2*(longer[i*upper*3+3*k])
            cmj[i*3+1] += c2*(longer[i*upper*3+3*k+1])
            cmj[i*3+2] += c2*(longer[i*upper*3+3*k+2])
        
    for i in prange(m,nogil=True):
        for j in range(m-i):
            dx = cmj[j*3] - cmj[(j+i)*3]
            dy = cmj[j*3+1] - cmj[(j+i)*3+1]
            dz = cmj[j*3+2] - cmj[(j+i)*3+2]
            result = dx*dx+dy*dy+dz*dz
            msd[i]+=result
            c[i]+=1

    for i in range(m):
        msd[i]/=c[i]
    
    return msdmj

@cython.boundscheck(False)
def crossDisplacementMdMj(double [:, :, :] coms_cat,
                          double [:, :, :] coms_an,
                          double [:, :]    md):

    cdef int i, j, nts=coms_cat.shape[0], nion=coms_cat.shape[1]
    
    mj_arr = cython.view.array(shape=(nts,3), itemsize=sizeof(double), format="d")
    cdef double [:, :] mj = mj_arr
    ctr_arr = np.zeros(nts, dtype=np.int32)
    cdef int [:] ctr = ctr_arr
    disp_arr = np.zeros(nts)
    cdef double [:] disp = disp_arr
    
    cdef double dx, dy, dz, result
    
    for i in prange(nts, nogil=True):
        for j in range(nion):
            mj[i][0] += coms_cat[i][j][0]
            mj[i][1] += coms_cat[i][j][1]
            mj[i][2] += coms_cat[i][j][2]
            mj[i][0] -= coms_an[i][j][0]
            mj[i][1] -= coms_an[i][j][1]
            mj[i][2] -= coms_an[i][j][2]

    for i in prange(nts, nogil=True):
        for j in range(nts-i):
            dx = md[j][0] - mj[i+j][0]
            dy = md[j][1] - mj[i+j][1]
            dz = md[j][2] - mj[i+j][2]
            result = dx*dx + dy*dy + dz*dz
            disp[i] = disp[i] + result
            ctr[i] = ctr[i] + 1
        disp[i] /= ctr[i]

    return disp_arr
    
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
