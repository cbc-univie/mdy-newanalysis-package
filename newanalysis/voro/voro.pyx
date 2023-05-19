# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.string cimport const_char
from cpython.version cimport PY_MAJOR_VERSION

cdef extern from "mod_voro.h":
    void _calcTessellation(double *xyz, float boxlength, int *f2c, int natoms, int nmolecules, int maxshell, char *ds, int *corelist, int ncore)
    void _calcTessellationVolSurf(double *xyz, float boxlength, int *f2c, int natoms, int nmolecules, int maxshell, char *ds, int *corelist, int ncore, float* vols, float* face_area_ptr)
    void _drawTessellation(double *xyz, float box_x, float box_y, float box_z, int npoints, int *points_to_draw, int npoints_to_draw, double cylinder_radius, bint triangles, int nmol, int color_id, const_char* filename)
    void _calcTessellationParallel(double *xyz, int *f2c, int *corelist, int *surroundlist, char *delaunay_ts, float boxl, int natoms, int nmolecules, int maxshell, int ncore, int nsurr, int ncpu)
    void _calcTessellationParallelAll(double *xyz, int *f2c, int *corelist, char *delaunay_ts, float boxl, int natoms, int nmolecules, int maxshell, int ncore, int ncpu)
    void _calcTessellationVolSurfAtomic(double *xyz, float boxlength, int *f2c, int natoms, int nmolecules, int maxshell, char *ds_ptr, int *corelist, int ncore, float* vols, float* face_area_ptr)

cdef unicode _ustring(s):
    if type(s) is unicode:
        # fast path for most common case(s)
        return <unicode>s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        # only accept byte strings in Python 2.x, not in Py3
        return (<bytes>s).decode('ascii')
    elif isinstance(s, unicode):
        # an evil cast to <unicode> might work here in some(!) cases,
        # depending on what the further processing does.  to be safe,
        # we can always create a copy instead
        return unicode(s)
    else:
        raise TypeError("Unknown string type!")

@cython.boundscheck(False)
def calcTessellation(np.ndarray[np.float64_t,ndim=2, mode="c"] xyz, float boxlength,
                     np.ndarray[np.int32_t,ndim=1, mode="c"] f2c, natoms, nmolecules, maxshell, 
                     np.ndarray[np.int32_t,ndim=1, mode="c"] corelist,
                     np.ndarray[np.float32_t,ndim=1,mode="c"] vols = None,
                     np.ndarray[np.float32_t,ndim=2,mode="c"] fa = None,
                     np.ndarray[np.int8_t,ndim=1,mode="c"] ds_ts = None, ts=None, ires=None,
                     np.ndarray[np.int32_t,ndim=1] surroundlist = None, h5file=None):
    """ 
        calcTessellation(xyz, boxlength, f2c, natoms, nmolecules, maxshell, corelist, vols, fa)

        This function wraps a c++ function calling the voro++ library\n
        args: \txyz\t\t---\t2-dim numpy array of 64 bit floats in C format with coordinates
              \tboxlength\t---\tboxlength as a float
              \tf2c\t\t---\tfine2coarse array, 1-dim, C-format, 32 bit integers
              \tnatoms\t\t---\tnumber of atoms
              \tnmolecules\t---\tnumber of molecules
              \tmaxshell\t---\toutermost shell to be calculated
              \tcorelist\t---\tresidue numbers of the 'core' molecules
              \tvols\t\t---\tarray in which the calculated volumes are to be inserted, 1-dim, C-format, 64-bit floats [optional]
              \tfa\t\t---\tarray in which the calculated interface areas are to be inserted, 2-dim, C-format, 64-bit floats [optional]

        ###############################################################################
        NOTE: do not call this function directly! instead, use the AtomGroup interface!
              E.g. if you have a selection "sel", call

              shells = sel.calcTessellation(maxshell=3,core_sel=None,volumes=None,face_areas=None)

              Doc-String:
                  Calculates the Voronoi tessellation of the whole box with the current AtomGroup as the core selection and returns the Delaunay distance matrix.

                  Arguments:
                  \tmaxshell:\tMaximum shell to calculate (default is 3)
                  \tcore_sel:\tAtomGroup object of core selection (default is self)
                  \tvolumes:\t1-dim numpy.float64 array for cell volume calculation (only together with face_areas!)
                  \tface_areas:\t2-dim numpy.float64 array for surface area calculation (only together with volumes!)

               The returned shells array will have n x n size, with n being the total number of residues in the box,
               but only those lines corresponding to the residue numbers of the residues in sel will be filled
               (cf. the output of sel.f2c(), meaning fine2coarse -- atom number -> residue number).
               In this way, one can save time by not generating the neighborhood information for unneeding residues.
"""
    cdef np.ndarray[np.int8_t,ndim=2,mode="c"] delaunay = np.empty((nmolecules,nmolecules),dtype=np.int8,order="C")
    cdef char *cds = <char *> delaunay.data
    cdef int *cts
    cdef double* cxyz = <double *> xyz.data
    cdef int* cf2c = <int *> f2c.data
    cdef int* ccorelist = <int *> corelist.data
    cdef int *csurrlist
    cdef float* cvols
    cdef float* cfa
    cdef int i, j, nsurr, ncore = <int> len(corelist), mindist, dist, icore, isurr
    if (vols is not None and fa is None) or (vols is None and fa is not None):
        print "ERROR: Either pass vols AND fa or neither of them! Neither volumes nor face areas were calculated!"
    if vols is not None:
        cvols = <float *> vols.data
    if fa is not None:
        cfa = <float *> fa.data
    if vols is not None and fa is not None:
        _calcTessellationVolSurf(cxyz, boxlength, cf2c, natoms, nmolecules, maxshell, cds, ccorelist,len(corelist),cvols,cfa)
    else:
        _calcTessellation(cxyz, boxlength, cf2c, natoms, nmolecules, maxshell, cds, ccorelist,len(corelist))
    if h5file is not None:
        dataset = h5file['delaunay']
        dataset[ts] = delaunay
    elif ds_ts is None:
        return delaunay
    else:
        cts = <int *> ds_ts.data
        if ires is not None:
            for i in range(nmolecules):
                cts[ts * nmolecules + i] = cds[ires * nmolecules + i]
        elif surroundlist is not None:
            nsurr = <int> len(surroundlist)
            csurrlist = <int *> surroundlist.data
            for i in range(nsurr):
                mindist = maxshell
                isurr = csurrlist[i]
                for j in range(ncore):
                    icore = ccorelist[j]
                    dist = cds[icore * nmolecules + isurr]
                    if dist < mindist:
                        mindist = dist
                cts[ts*nsurr+isurr] = mindist
        else:
            for i in range(nmolecules):
                for j in range(nmolecules):
                    cts[ts * nmolecules * nmolecules + nmolecules * j + i] = cds[j * nmolecules + i]

def calcTessellationVolSurfAtomic(double [:,:] xyz, float boxlength,
                                  int [:] f2c, int natoms, int nmolecules, int maxshell, 
                                  int [:] corelist, float [:] vols, float [:] fa):
    
    cdef char [:,:] delaunay = np.empty((nmolecules, nmolecules), dtype=np.int8)

    _calcTessellationVolSurfAtomic(&xyz[0,0], boxlength, &f2c[0], natoms, nmolecules, maxshell,
                                   &delaunay[0,0], &corelist[0], corelist.shape[0], &vols[0], &fa[0])

    return np.asarray(delaunay)

def calcTessellationParallel(double [:,:,:] xyz_ts, int [:] f2c, int [:] corelist, int [:] surroundlist, char [:,:] delaunay_ts, boxlength, natoms, nmolecules, maxshell):
    "calcTessellationParallel(xyz_ts,f2c,corelist,surroundlist,delaunay_ts,boxl,nat,nmol,maxshell)"
    cdef int nat = <int> natoms, nmol = <int> nmolecules, maxsh = <int> maxshell, ncore = <int> len(corelist), nsurr = <int> len(surroundlist), ncpu = <int> len(delaunay_ts)
    cdef float boxl = <float> boxlength

    _calcTessellationParallel(&(xyz_ts[0,0,0]), &f2c[0], &corelist[0], &surroundlist[0], &(delaunay_ts[0,0]), boxl, nat, nmol, maxsh, ncore, nsurr, ncpu)
    
def calcTessellationParallelAll(double [:,:,:] xyz_ts, int [:] f2c, int [:] corelist, char [:,:,:] delaunay_ts, boxlength, natoms, nmolecules, maxshell):
    "calcTessellationParallelAll(xyz_ts,f2c,corelist,delaunay_ts,boxl,nat,nmol,maxshell)"
    cdef int nat = <int> natoms, nmol = <int> nmolecules, maxsh = <int> maxshell, ncore = <int> len(corelist), ncpu = <int> len(delaunay_ts)
    cdef float boxl = <float> boxlength

    _calcTessellationParallelAll(&(xyz_ts[0,0,0]), &f2c[0], &corelist[0], &(delaunay_ts[0,0,0]), boxl, nat, nmol, maxsh, ncore, ncpu)

@cython.boundscheck(False)
def nearestNeighbor(int [:] corelist, int [:] surroundlist, char [:,:,:] delaunay_tmp, char [:,:,:] delaunay_ts, int iubq, int maxshell):
    cdef int icore, isurr, c, j, k, nsurr = len(surroundlist), ncore = len(corelist), ncpu = delaunay_tmp.shape[0]
    cdef char dist, mindist

    for c in range(ncpu):
        for j in range(nsurr):
            mindist = maxshell + 1
            isurr = surroundlist[j]
            for k in range(ncore):
                icore = corelist[k]
                dist = delaunay_tmp[c,icore,isurr]
                if dist < mindist and dist != -1:
                    mindist = dist
            delaunay_ts[c,iubq,j] = mindist    
                    
def drawTessellation(np.ndarray[np.float64_t,ndim=2, mode="c"] xyz, 
                     np.ndarray[np.int32_t,ndim=1, mode="c"] points_to_draw, cylinder_radius,
                     triangles, int nmol, int color_id, filename, box_x, box_y=None, box_z=None):
    """
    drawTessellation(xyz, points_to_draw, cylinder_radius, triangles, nmol, color_id, filename, box_x, box_y=None, box_z=None)
    """

    cdef int len_xyz = len(xyz), len_points_to_draw = len(points_to_draw)
    py_byte_string = _ustring(filename).encode('UTF-8')
    cdef const_char* fn = py_byte_string
    cdef double *xyz_ptr = <double *> xyz.data
    cdef int *points_ptr = <int *> points_to_draw.data
    cdef bint draw_triangles = triangles
    cdef double radius = <double> cylinder_radius
    cdef float boxl_x, boxl_y, boxl_z

    boxl_x = <float> box_x
    if box_y:
        boxl_y = <float> box_y
    else:
        boxl_y = boxl_x
    if box_z:
        boxl_z = <float> box_z
    else:
        boxl_z = boxl_x
    
    _drawTessellation(xyz_ptr, boxl_x, boxl_y, boxl_z, len_xyz, points_ptr, len_points_to_draw, radius, draw_triangles, nmol, color_id, fn)

@cython.boundscheck(False)
def buildNeighborList(np.ndarray[np.int32_t,ndim=2] ds,
                       np.ndarray[np.int8_t,ndim=3] result,
                       core_first, core_last, surr_first, surr_last,
                       ctr, minshell=1, maxshell=1):
    """
    buildNeighborList(ds, result, core_first, core_last, surr_first, surr_last, ctr, minshell=1, maxshell=1)

    Add the delaunay distance matrix information from the current frame to a residence timeseries for later correlation.
    """

    cdef int ncore = result.shape[1]
    cdef int nsurr = result.shape[2]
    cdef int n2 = ncore * nsurr
    cdef int ntot = ds.shape[0]
    cdef int i, j, k = ctr
    cdef int *cds = <int *> ds.data
    cdef np.int8_t* cres = <np.int8_t *> result.data
    cdef int c1 = core_first, c2 = core_last, s1=surr_first, s2=surr_last
    cdef int lower_limit = minshell, upper_limit = maxshell
    
    for i in prange(c1,c2,nogil=True):
        for j in range(s1,s2):
            if cds[i*ntot+j]>=lower_limit and cds[i*ntot+j]<=upper_limit:
                cres[k*n2+i*ncore+j] = <np.int8_t> 1
    
def calcCN(np.ndarray[np.int32_t,ndim=2] ds,
           np.ndarray[np.float64_t,ndim=1] result,
           core_begin, core_end, surr_begin, surr_end, shell_begin, shell_end, ntot):

    """
    calcCN(ds,result,core_begin,core_end,surr_begin,surr_end,shell_begin,shell_end,ntot)

    Calculates the coordination number between two selections and adds the found value to a histogram.
    
    Args:
        ds          .. delaunay distance matrix of the whole system
        result      .. result histogram
        core_begin  .. first residue number of the core selection
        core_end    .. last residue number of the core selection
        surr_begin  .. first residue number of the surround selection
        surr_end    .. last residue number of the surround selection
        shell_begin .. lowest delaunay distance to consider
        shell_end   .. highest delaunay distance to consider
        ntot        .. total number of residues in the system (the number of columns in the ds matrix)
    """
    
    cdef int c1=core_begin, c2=core_end, s1=surr_begin, s2=surr_end, i, j, sh1=shell_begin, sh2=shell_end, n=ntot
    cdef int *cds = <int *> ds.data
    cdef double *cres = <double *> result.data
    cdef int ctr, nmax=len(result)

    # e.g. c1=0, c2=499
    for i in range(c1,c2+1):
        ctr=0
        for j in range(s1,s2+1):
            if cds[i*n+j]>=sh1 and cds[i*n+j]<=sh2:
                ctr+=1
        if ctr >= nmax:
            print "Result array is too short!\n"
        else:
            cres[ctr]+=1

def calcCN_int8(np.ndarray[np.int8_t,ndim=2] ds,
                np.ndarray[np.float64_t,ndim=1] result,
                core_begin, core_end, surr_begin, surr_end, shell_begin, shell_end, ntot):

    """
    calcCN(ds,result,core_begin,core_end,surr_begin,surr_end,shell_begin,shell_end,ntot)

    Calculates the coordination number between two selections and adds the found value to a histogram.
    
    Args:
        ds          .. delaunay distance matrix of the whole system
        result      .. result histogram
        core_begin  .. first residue number of the core selection
        core_end    .. last residue number of the core selection
        surr_begin  .. first residue number of the surround selection
        surr_end    .. last residue number of the surround selection
        shell_begin .. lowest delaunay distance to consider
        shell_end   .. highest delaunay distance to consider
        ntot        .. total number of residues in the system (the number of columns in the ds matrix)
    """
    
    cdef int c1=core_begin, c2=core_end, s1=surr_begin, s2=surr_end, i, j, sh1=shell_begin, sh2=shell_end, n=ntot
    cdef char *cds = <char *> ds.data
    cdef double *cres = <double *> result.data
    cdef int ctr, nmax=len(result)

    # e.g. c1=0, c2=499
    for i in range(c1,c2+1):
        ctr=0
        for j in range(s1,s2+1):
            if cds[i*n+j]>=sh1 and cds[i*n+j]<=sh2:
                ctr+=1
        if ctr >= nmax:
            print "Result array is too short!\n"
        else:
            cres[ctr]+=1

