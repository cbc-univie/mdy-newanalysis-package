# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport fabs, sqrt, floor, pow

# CUDA handler interface
class CUDA(object):
    def __init__(self):
        self.cuda = True
        try:
            import pycuda.autoinit
            import pycuda.driver as drv
            from pycuda.compiler import SourceModule
            if pycuda.autoinit.device.compute_capability() < (1, 3):
                self.cuda = False
        except:
            self.cuda = False

        if self.cuda:
            self.drv = drv
            self.dev = pycuda.autoinit.device
            self.source = SourceModule("""
/*            __device__ double atomicAdd(double* address, double val)
            {
            unsigned long long int* address_as_ull = (unsigned long long int*)address;
            unsigned long long int old = *address_as_ull, assumed;
        
            do {
              assumed = old;
              old = atomicCAS(address_as_ull, assumed,
              __double_as_longlong(val +
              __longlong_as_double(assumed)));
              // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);

            return __longlong_as_double(old);
            }
*/
            __global__ void calcHistoGPU(float *core_xyz, float *surr_xyz, float *core_dip, float *surr_dip, float *dest, float* box_dim, int ncore, int nsurround, float histo_min, float histo_max, int histo_n, float invdx, int mode_sel) {
            
            const int i = threadIdx.x + blockIdx.x * blockDim.x;
            const int j = threadIdx.y + blockIdx.y * blockDim.y;
            float off_diag = 1.0;
            if (i >= ncore || j >= nsurround || (j<i && ncore==nsurround)) return;
            if (j > i && ncore == nsurround) off_diag = 2.0;
            const int ix = i*3, ix3 = i*histo_n, ix4=ncore*histo_n;
            float c_dx=core_xyz[ix], c_dy=core_xyz[ix+1], c_dz=core_xyz[ix+2], dx, dy, dz;
            float s_len, incr, dist;
            float c_px, c_py, c_pz, s_px, s_py, s_pz;
            int pos;
            float c_len;
            const int ix2 = j*3;

            if (mode_sel & 2 || mode_sel & 4 || mode_sel & 16 || mode_sel & 32) {
                c_px = core_dip[ix];
                c_py = core_dip[ix+1];
                c_pz = core_dip[ix+2];
                c_len = sqrt(c_px*c_px+c_py*c_py+c_pz*c_pz);
            }
            if (mode_sel & 2 || mode_sel & 8 || mode_sel & 16 || mode_sel & 64) {
                s_px = surr_dip[ix2];
                s_py = surr_dip[ix2+1];
                s_pz = surr_dip[ix2+2];
                s_len = sqrt(s_px*s_px+s_py*s_py+s_pz*s_pz);
            }

            dx = surr_xyz[ix2] - c_dx; dy=surr_xyz[ix2+1] - c_dy; dz=surr_xyz[ix2+2] - c_dz;
            if (fabs(dx) > box_dim[3]) dx-=((0.0 < dx) - (dx < 0.0))*box_dim[0];
            if (fabs(dy) > box_dim[4]) dy-=((0.0 < dy) - (dy < 0.0))*box_dim[1];
            if (fabs(dz) > box_dim[5]) dz-=((0.0 < dz) - (dz < 0.0))*box_dim[2];
            dist = float(sqrt(dx*dx+dy*dy+dz*dz));
            
            if (dist<histo_min || dist>histo_max || floor(dist*invdx)==0) return;
            
            pos = (int) floor((dist-histo_min)*invdx);
            
            if (mode_sel & 1) atomicAdd(&(dest[ix3+pos]),1.0*off_diag);
            
            if (mode_sel & 2 || mode_sel & 16) {
                incr = (c_px*s_px+c_py*s_py+c_pz*s_pz)/(c_len*s_len);
                if (mode_sel & 2)  atomicAdd(&(dest[ix4+ix3+pos]),incr*off_diag);
                if (mode_sel & 16) atomicAdd(&(dest[4*ix4+ix3+pos]),(1.5*incr*incr-0.5)*off_diag); 
            }
            if (mode_sel & 4 || mode_sel & 32) {    
                incr = (c_px*dx+c_py*dy+c_pz*dz)/(c_len*dist);
                if (mode_sel & 4)  atomicAdd(&(dest[2*ix4+ix3+pos]),incr*off_diag);
                if (mode_sel & 32) atomicAdd(&(dest[5*ix4+ix3+pos]),(1.5*incr*incr-0.5)*off_diag);
            }
            if (mode_sel & 8 || mode_sel & 64) {
                incr = (s_px*dx+s_py*dy+s_pz*dz)/(s_len*dist);
                if (mode_sel & 8)  atomicAdd(&(dest[3*ix4+ix3+pos]),incr*off_diag);
                if (mode_sel & 64) atomicAdd(&(dest[6*ix4+ix3+pos]),(1.5*incr*incr-0.5)*off_diag);
            } 
            } 
            """)
            self._calcHistoGPU = self.source.get_function("calcHistoGPU")

    def calcHistoGPU(self,coor_sel1_gpu, coor_sel2_gpu, dip_sel1_gpu, dip_sel2_gpu, histo_gpu, box_dim_gpu, n1, n2, histo_min, histo_max, histo_n, histo_inv_dr, mode_sel, block, grid):
        self._calcHistoGPU(coor_sel1_gpu, coor_sel2_gpu, dip_sel1_gpu, dip_sel2_gpu, histo_gpu, box_dim_gpu, n1, n2, histo_min, histo_max, histo_n, histo_inv_dr, mode_sel,block=block,grid=grid)

cu=CUDA()
        
# CPU version
cdef void calcHisto(double *core_xyz, double *surr_xyz, double *core_dip, double *surr_dip, 
                    np.ndarray[np.float64_t, ndim=1] np_dest, np.ndarray[np.float64_t,ndim=1] np_box_dim, 
                    int ncore, int nsurround, float histo_min, float histo_max, int histo_n, double invdx, int mode_sel):

    cdef int i, j, pos, ix1=ncore*histo_n, ix2, j_begin=0
    cdef double dx, dy, dz, dist, cpx, cpy, cpz, spx, spy, spz, slen, clen, incr, off_diag=1.0
    cdef double *dest = <double*> np_dest.data
    cdef double *box_dim = <double*> np_box_dim.data

    for i in prange(ncore, nogil=True):
        ix2=i*histo_n
        if mode_sel & 2 or mode_sel & 4 or mode_sel & 16 or mode_sel & 32:
            cpx = core_dip[i*3]
            cpy = core_dip[i*3+1]
            cpz = core_dip[i*3+2]
            clen = sqrt(cpx*cpx+cpy*cpy+cpz*cpz)
        if ncore == nsurround:
            j_begin = i
        else:
            j_begin = 0
        for j in range(j_begin,nsurround):
            if ncore == nsurround and i != j:
                off_diag = 2.0
            else:
                off_diag = 1.0
            if mode_sel & 2 or mode_sel & 8 or mode_sel & 16 or mode_sel & 64:
                spx = surr_dip[j*3]
                spy = surr_dip[j*3+1]
                spz = surr_dip[j*3+2]
                slen = sqrt(spx*spx+spy*spy+spz*spz)

            dx = surr_xyz[j*3]   - core_xyz[i*3]
            dy = surr_xyz[j*3+1] - core_xyz[i*3+1]
            dz = surr_xyz[j*3+2] - core_xyz[i*3+2]

            if fabs(dx) > box_dim[3]: dx = dx-((0.0 < dx) - (dx < 0.0))*box_dim[0]
            if fabs(dy) > box_dim[4]: dy = dy-((0.0 < dy) - (dy < 0.0))*box_dim[1]
            if fabs(dz) > box_dim[5]: dz = dz-((0.0 < dz) - (dz < 0.0))*box_dim[2]
            dist = sqrt(dx*dx+dy*dy+dz*dz)
            if dist<histo_min or dist>histo_max or floor(dist*invdx)==0: continue
            pos = <int> floor((dist-histo_min)*invdx)

            if mode_sel & 1: dest[ix2+pos]+=1.0*off_diag
            
            if mode_sel & 2 or mode_sel & 16:
                incr = (cpx*spx+cpy*spy+cpz*spz)/(clen*slen)
                if mode_sel & 2: dest[ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 16: dest[4*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 4 or mode_sel & 32:
                incr = (cpx*dx+cpy*dy+cpz*dz)/(clen*dist)
                if mode_sel & 4: dest[2*ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 32: dest[5*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 8 or mode_sel & 64:
                incr = (spx*dx+spy*dy+spz*dz)/(slen*dist)
                if mode_sel & 8: dest[3*ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 64: dest[6*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
                
# RDF time series
@cython.boundscheck(False)
def calcHistoTS(double [:,:] core_xyz, double [:,:] surr_xyz, double [:,:,:] result,
                double boxl, int t, float histo_min, float histo_max, double invdx):

    cdef int ncore = core_xyz.shape[0]
    cdef int nsurr = surr_xyz.shape[0]
    cdef int i, j, pos
    cdef double dx, dy, dz, dist
    cdef double boxl2 = boxl / 2.0
    
    for i in prange(ncore, nogil=True):
        for j in range(nsurr):
            dx = surr_xyz[j,0] - core_xyz[i,0]
            dy = surr_xyz[j,1] - core_xyz[i,1]
            dz = surr_xyz[j,2] - core_xyz[i,2]

            if fabs(dx) > boxl2: dx = dx-((0.0 < dx) - (dx < 0.0))*boxl
            if fabs(dy) > boxl2: dy = dy-((0.0 < dy) - (dy < 0.0))*boxl
            if fabs(dz) > boxl2: dz = dz-((0.0 < dz) - (dz < 0.0))*boxl

            dist = sqrt(dx*dx+dy*dy+dz*dz)
            if dist<histo_min or dist>histo_max or floor(dist*invdx)==0: continue
            pos = <int> floor((dist-histo_min)*invdx)

            result[t,i,pos] += 1.0

# RDF time series with voronoi decomposition
@cython.boundscheck(False)
def calcHistoTSVoro(double [:,:] core_xyz, double [:,:] surr_xyz, double [:,:,:,:] result, char [:,:,:] ds, int maxshell,
                    double boxl, int t, float histo_min, float histo_max, double invdx):

    cdef int ncore = core_xyz.shape[0]
    cdef int nsurr = surr_xyz.shape[0]
    cdef int i, j, pos
    cdef char shell
    cdef double dx, dy, dz, dist
    cdef double boxl2 = boxl / 2.0

    for i in prange(ncore, nogil=True):
        for j in range(nsurr):
            dx = surr_xyz[j,0] - core_xyz[i,0]
            dy = surr_xyz[j,1] - core_xyz[i,1]
            dz = surr_xyz[j,2] - core_xyz[i,2]

            if fabs(dx) > boxl2: dx = dx-((0.0 < dx) - (dx < 0.0))*boxl
            if fabs(dy) > boxl2: dy = dy-((0.0 < dy) - (dy < 0.0))*boxl
            if fabs(dz) > boxl2: dz = dz-((0.0 < dz) - (dz < 0.0))*boxl

            dist = sqrt(dx*dx+dy*dy+dz*dz)
            if dist<histo_min or dist>histo_max or floor(dist*invdx)==0: continue
            pos = <int> floor((dist-histo_min)*invdx)

            shell = ds[t,i,j]
            if shell > maxshell:
                shell = maxshell

            result[t,i,shell-1,pos] += 1.0

            
class RDF(object):
    """
    class RDF

    A container object for calculating radial distribution functions.
    """

    def __init__(self, modes, histo_min, histo_max, histo_dr, sel1_n, sel2_n, sel1_nmol, sel2_nmol, box_x, box_y=None, box_z=None, use_cuda=False, norm_volume=None):
        """
        RDF.__init__(modes, histo_min, histo_max, histo_dr, sel1_n, sel2_n, sel1_nmol, sel2_nmol, box_x, box_y=None, box_z=None, use_cuda=False)

        Creates a new rdf container.

        modes .. a list of gfunctions to calculate
                 e.g. ["all"] -> calculate all gfunctions
                      ["000","110","220"] -> calculate only g000, g110 and g220
        histo_min .. lower bound of histogram
        histo_max .. upper bound of histogram
        histo_dr  .. histogram bin width
        sel1_n    .. number of particles in first selection
        sel2_n    .. number of particles in second selection
        sel1_nmol .. number of molecules in first selection; needed for correct normalisation, if multiple atoms from the same molecule are selected
        sel2_nmol .. number of molecules in second selection
        box_x     .. box length, x dim
        box_y     .. box length, y dim (optional; if None, y=x)
        box_z     .. box length, z dim (optional; if None, z=x)
        use_cuda  .. whether to use cuda, if available (default=False)
        norm_volume .. volume to use for calculating the density during normalization (default = None)

        Usage example:
        my_rdf = RDF(["all"], 0.0, 50.0, 0.05, 200, 200, 200, 200, 25.0)
        """

        if use_cuda and cu.cuda:
            self.cuda = True
        elif use_cuda and not cu.cuda:
            print "No suitable CUDA device was found to calculate radial distribution functions. Using CPU instead."
            self.cuda = False
        else:
            self.cuda = False

        if self.cuda:
            self.dtype = np.float32
        else:
            self.dtype = np.float64
            
        # always initialize the histogram to 7 dimensions, but only fill the desired ones
        cdef int mode_sel = 0

        # select the gfunctions to calculate using bitwise operators
        for mode in modes:
            if mode == "all": mode_sel = mode_sel | 1 | 2 | 4 | 8 | 16 | 32 | 64
            if mode == "000": mode_sel = mode_sel | 1
            if mode == "110": mode_sel = mode_sel | 2
            if mode == "101": mode_sel = mode_sel | 4
            if mode == "011": mode_sel = mode_sel | 8
            if mode == "220": mode_sel = mode_sel | 16
            if mode == "202": mode_sel = mode_sel | 32
            if mode == "022": mode_sel = mode_sel | 64

        # remember the selection
        self.mode_sel = np.int32(mode_sel)
        
        # set histogram parameters
        self.histo_min = np.float32(histo_min)
        self.histo_max = np.float32(histo_max)
        self.histo_dr  = np.round(self.dtype(histo_dr),5)
        self.histo_inv_dr = self.dtype(1.0 / self.histo_dr)
        self.histo_n = np.int32((histo_max-histo_min)/histo_dr+0.5)

        self.box_dim = np.zeros(6,dtype=self.dtype)
        self.box_dim[0] = box_x
        self.box_dim[3] = box_x/2.0
        if box_y: 
            self.box_dim[1] = box_y
            self.box_dim[4] = box_y/2.0
        else: 
            self.box_dim[1] = self.box_dim[0]
            self.box_dim[4] = self.box_dim[3]
        if box_z: 
            self.box_dim[2] = box_z
            self.box_dim[5] = box_z/2.0
        else: 
            self.box_dim[2] = self.box_dim[0]
            self.box_dim[5] = self.box_dim[3]
           
        self.volume = self.box_dim[0] * self.box_dim[1] * self.box_dim[2]

        # remember volume for each frame for normalisation
        self.volume_list = []

        self.norm_volume = norm_volume
        
        # set particle numbers
        self.n1 = np.int32(sel1_n)
        self.n2 = np.int32(sel2_n)
        self.nmol1 = sel1_nmol
        self.nmol2 = sel2_nmol

        self.oneistwo = None

        # set frame counter to 0
        self.ctr = 0

        # initialize histogram array
        self.histogram = np.zeros(self.n1*self.histo_n*7,dtype=self.dtype)
        self.histogram_out = np.zeros(self.histo_n*7,dtype=np.float64)
        
        # print, if CUDA device will be used and initialize arrays
        if self.cuda and use_cuda:
            # initialize gpu arrays

            self.histo_gpu=cu.drv.mem_alloc(self.histogram.nbytes)
            cu.drv.memcpy_htod(self.histo_gpu, self.histogram)

            self.box_dim_gpu=cu.drv.mem_alloc(self.box_dim.nbytes)
            cu.drv.memcpy_htod(self.box_dim_gpu, self.box_dim)

            arr_sel1 = np.zeros((self.n1,3),dtype=np.float32)
            arr_sel2 = np.zeros((self.n2,3),dtype=np.float32)
            self.coor_sel1_gpu = cu.drv.mem_alloc(arr_sel1.nbytes)
            self.coor_sel2_gpu = cu.drv.mem_alloc(arr_sel2.nbytes)
            
            if mode_sel & 2 or mode_sel & 4 or mode_sel & 16 or mode_sel & 32:
                self.dip_sel1_gpu = cu.drv.mem_alloc(arr_sel1.nbytes)
            if mode_sel & 2 or mode_sel & 8 or mode_sel & 16 or mode_sel & 64:
                self.dip_sel2_gpu = cu.drv.mem_alloc(arr_sel2.nbytes)

            # set gpu block and grid
            block_x = cu.dev.get_attribute(cu.drv.device_attribute.MULTIPROCESSOR_COUNT) * 2
            block_y = cu.dev.get_attribute(cu.drv.device_attribute.MULTIPROCESSOR_COUNT) * 2
            block_z = 1
            grid_x = int(self.n1/block_x)
            if self.n1%block_x: grid_x+=1
            grid_y = int(self.n2/block_y)
            if self.n2%block_y: grid_y+=1
            self.block=(block_x,block_y,block_z)
            self.grid=(grid_x,grid_y)

            
    def update_box(self,box_x,box_y,box_z):
        """
        update_box(box_x,box_y,box_z)

        Update the box dimensions, when analysing NpT trajectories.
        """
        self.box_dim[0] = box_x
        self.box_dim[3] = box_x/2.0
        self.box_dim[1] = box_y
        self.box_dim[4] = box_y/2.0
        self.box_dim[2] = box_z
        self.box_dim[5] = box_z/2.0

        if self.cuda:
            cu.drv.memcpy_htod(self.box_dim_gpu, self.box_dim)

        self.volume = self.box_dim[0] * self.box_dim[1] * self.box_dim[2]

    def calcFrame(self,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] coor_sel1,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] coor_sel2,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] dip_sel1 = None,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] dip_sel2 = None,
                  pre_coor_sel1_gpu=None, pre_coor_sel2_gpu=None, 
                  pre_dip_sel1_gpu=None, pre_dip_sel2_gpu=None):
        """
        calcFrame(coor_sel1, coor_sel2, dip_sel1=None, dip_sel2=None,
                  pre_coor_sel1_gpu=None, pre_coor_sel2_gpu=None, 
                  pre_dip_sel1_gpu=None, pre_dip_sel2_gpu=None)

        Args:
             coor_sel1 .. numpy-array (dtype=numpy.float64, ndim=2, mode="c") containing the coordinates of the first selection
             coor_sel2 .. numpy-array containing the coordinates of the second selection
             dip_sel1  .. numpy-array containing the dipoles of the first selection (optional, only needed for 110, 101, 220 and 202)
             dip_sel2  .. numpy-array containing the dipoles of the second selection (optional, only needed for 110, 011, 220 and 022)

             pre-allocated arrays: PyCUDA GPU array objects obtained from drv.memalloc()

        Pass the data needed to calculate the RDF for the current frame and do the calculation.
        Optionally, pre-allocated arrays on the GPU can be used. This is useful, if they are reused for more than
        one rdf calculation.
        """
        
        # Check if dipole moments are needed and, if yes, were passed
        if dip_sel1 is None and ((<int> self.mode_sel) & 2 or (<int> self.mode_sel) & 4 or (<int> self.mode_sel) & 16 or (<int> self.mode_sel) & 32):
            raise TypeError("No dipole moment array for the fist selection was passed, although the requested gfunctions need it!")

        if dip_sel2 is None and ((<int> self.mode_sel) & 2 or (<int> self.mode_sel) & 8 or (<int> self.mode_sel) & 16 or (<int> self.mode_sel) & 64):
            raise TypeError("No dipole moment array for the second selection was passed, although the requested gfunctions need it!")

        if self.oneistwo is None:
            if (coor_sel1[0] == coor_sel2[0]).all() and self.nmol1 == self.nmol2:
                self.oneistwo = True
            else:
                self.oneistwo = False

        # Increment frame counter
        self.ctr+=1

        # add current volume to list
        self.volume_list.append(self.volume)

        if self.cuda:
            # allocate memory on cuda device and copy data to device
            if pre_coor_sel1_gpu:
                self.coor_sel1_gpu = pre_coor_sel1_gpu
            else:
                cu.drv.memcpy_htod(self.coor_sel1_gpu, coor_sel1.astype(np.float32))
            if pre_coor_sel2_gpu:
                self.coor_sel2_gpu = pre_coor_sel2_gpu
            else:
                cu.drv.memcpy_htod(self.coor_sel2_gpu, coor_sel2.astype(np.float32))
            # if no dipole information was given, just point at the coordinates instead to keep the gpu function happy
            if dip_sel1 is not None:
                if pre_dip_sel1_gpu:
                    self.dip_sel1_gpu = pre_dip_sel1_gpu
                else:
                    cu.drv.memcpy_htod(self.dip_sel1_gpu, dip_sel1.astype(np.float32))
            else:
                self.dip_sel1_gpu = self.coor_sel1_gpu
            if dip_sel2 is not None:
                if pre_dip_sel2_gpu:
                    self.dip_sel2_gpu = pre_dip_sel2_gpu
                else:
                    cu.drv.memcpy_htod(self.dip_sel2_gpu, dip_sel2.astype(np.float32))
            else:
                self.dip_sel2_gpu = self.coor_sel2_gpu

            cu.calcHistoGPU(self.coor_sel1_gpu, self.coor_sel2_gpu, self.dip_sel1_gpu, self.dip_sel2_gpu, 
                           self.histo_gpu, self.box_dim_gpu, self.n1, self.n2, self.histo_min, 
                           self.histo_max, self.histo_n, self.histo_inv_dr, self.mode_sel, self.block, self.grid)

        else:
            calcHisto(<double*> coor_sel1.data, <double*> coor_sel2.data, 
                      <double*> dip_sel1.data, <double*> dip_sel2.data, 
                      self.histogram, self.box_dim, 
                      self.n1, self.n2, self.histo_min, self.histo_max, 
                      self.histo_n, self.histo_inv_dr, self.mode_sel)
            
    def _addHisto(self,np.ndarray[np.float64_t,ndim=1] histo_out, np.ndarray[np.float64_t,ndim=1] histo):
        """
        Add the histograms of all core particles, after they were calculated in parallel.
        """
        
        cdef int histo_n = self.histo_n, n1=self.n1, i, j, k, ix, ix2 = self.n1*self.histo_n
        cdef double* histogram_out = <double*> histo_out.data
        cdef double* histogram = <double*> histo.data

        # add histograms
        for i in prange(histo_n, nogil=True):
            for j in range(n1):
                ix = histo_n * j
                for k in range(7):
                    histogram_out[k*histo_n+i]+=histogram[k*ix2+ix+i]


    def _normHisto(self):
        """
        Normalize the histogram.
        """

        if self.norm_volume:
            volume = self.norm_volume
        else:
            volume = 0.0
            for v in self.volume_list:
                volume+=v
            volume/=len(self.volume_list)

        cdef int i, j, ix, ix2 = self.n1 * self.histo_n
        PI = 3.14159265358979323846
        
        # norm histograms
        if self.oneistwo:
            rho = float(self.nmol1 * (self.nmol1 - 1)) / volume
        else:
            rho = float(self.nmol1 * self.nmol2) / volume

        for i in range(self.histo_n):
            r = self.histo_min + float(i) * self.histo_dr
            r_out = r + self.histo_dr
            slice_vol = 4.0/3.0 * PI * (r_out**3 - r**3)
            norm = rho*slice_vol*float(self.ctr)
            for j in range(7):
                self.histogram_out[j*self.histo_n+i]/=norm

    def scale(self, value):
        self.histogram_out[:] *= value
        self.histogram[:] *= value

    def write(self, filename="rdf"):
        """
        RDF.write(filename="rdf")

        Args:
             filename .. name of the file to be written to. Each type of rdf will be written to a separate file
                         with the ending specifying the type.
                         
        Norm the calculated histograms and write them to the specified file.
        """

        if self.cuda:
            cu.drv.memcpy_dtoh(self.histogram, self.histo_gpu)
            
        self._addHisto(self.histogram_out, self.histogram.astype(np.float64))
        self._normHisto()
        
        if self.mode_sel & 1:  f000 = open(filename+"_g000.dat",'w')
        if self.mode_sel & 2:  f110 = open(filename+"_g110.dat",'w')
        if self.mode_sel & 4:  f101 = open(filename+"_g101.dat",'w')
        if self.mode_sel & 8:  f011 = open(filename+"_g011.dat",'w')
        if self.mode_sel & 16: f220 = open(filename+"_g220.dat",'w')
        if self.mode_sel & 32: f202 = open(filename+"_g202.dat",'w')
        if self.mode_sel & 64: f022 = open(filename+"_g022.dat",'w')

        for i in range(self.histo_n):
            r = self.histo_min+i*self.histo_dr+self.histo_dr*0.5
            if self.mode_sel & 1:  f000.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[i]))
            if self.mode_sel & 2:  f110.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[self.histo_n+i]))
            if self.mode_sel & 4:  f101.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[2*self.histo_n+i]))
            if self.mode_sel & 8:  f011.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[3*self.histo_n+i]))
            if self.mode_sel & 16: f220.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[4*self.histo_n+i]))
            if self.mode_sel & 32: f202.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[5*self.histo_n+i]))
            if self.mode_sel & 64: f022.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[6*self.histo_n+i]))

    def resetArrays(self):
        """
        Reset histogram arrays to zero
        """
        self.histogram_out[:] = 0.0
        self.histogram[:] = 0.0

        
    def printInfo(self):
        """
        Print some information about the RDF container.
        """

        print("Modes: ", self.mode_sel)
        print("Frames read: ", self.ctr)

###################################################################################################

cdef void calcHistoVoronoi(double *core_xyz, double *surr_xyz, double *core_dip, double *surr_dip, 
                    np.ndarray[np.float64_t, ndim=1] np_dest, np.ndarray[np.float64_t,ndim=1] np_box_dim, 
                    int ncore, int nsurround, float histo_min, float histo_max, int histo_n, double invdx, int mode_sel, int nm_core, int nm_surround, int nshells, int * delaunay_matrix):

    #cdef int i, j, pos, ix1=ncore*histo_n, ix2, j_begin=0 # Change ix1, ix2
    cdef int i, j, pos, ix1=nshells*ncore*histo_n, ix2, j_begin=0 #ROLLBACK: ERASE THIS, UNCOMMENT ABOVE
    cdef double dx, dy, dz, dist, cpx, cpy, cpz, spx, spy, spz, slen, clen, incr, off_diag=1.0
    cdef double *dest = <double*> np_dest.data
    cdef double *box_dim = <double*> np_box_dim.data
    cdef int ds_pos1, ds_pos2, shell, ix3, ix4
    cdef int apr_core = ncore / nm_core
    cdef int apr_surround = nsurround / nm_surround


    #TODO: Change with nm_core, int nm_surround, int delaunay_matrix, nshells

    for i in prange(ncore, nogil=True):
        #ix2=i*histo_n
        ix2 = i*nshells*histo_n #ROLLBACK: ERASE THESE 2, UNCOMMENT ABOVE
        ds_pos1 = i/apr_core 

        if mode_sel & 2 or mode_sel & 4 or mode_sel & 16 or mode_sel & 32:
            cpx = core_dip[i*3]
            cpy = core_dip[i*3+1]
            cpz = core_dip[i*3+2]
            clen = sqrt(cpx*cpx+cpy*cpy+cpz*cpz)
        if ncore == nsurround:
            j_begin = i
        else:
            j_begin = 0
        for j in range(j_begin,nsurround):
            ds_pos2 = j/apr_surround # ROLLBACK: ERASE THESE 2
            shell = delaunay_matrix[ds_pos1*nsurround + ds_pos2] #Correctly shaped?
            shell = shell-1
            if shell <  0:       shell = nshells-1
            if shell >= nshells: shell = nshells-1

            if ncore == nsurround and i != j:
                off_diag = 2.0
            else:
                off_diag = 1.0
            if mode_sel & 2 or mode_sel & 8 or mode_sel & 16 or mode_sel & 64:
                spx = surr_dip[j*3]
                spy = surr_dip[j*3+1]
                spz = surr_dip[j*3+2]
                slen = sqrt(spx*spx+spy*spy+spz*spz)

            dx = surr_xyz[j*3]   - core_xyz[i*3]
            dy = surr_xyz[j*3+1] - core_xyz[i*3+1]
            dz = surr_xyz[j*3+2] - core_xyz[i*3+2]

            if fabs(dx) > box_dim[3]: dx = dx-((0.0 < dx) - (dx < 0.0))*box_dim[0]
            if fabs(dy) > box_dim[4]: dy = dy-((0.0 < dy) - (dy < 0.0))*box_dim[1]
            if fabs(dz) > box_dim[5]: dz = dz-((0.0 < dz) - (dz < 0.0))*box_dim[2]
            dist = sqrt(dx*dx+dy*dy+dz*dz)
            if dist<histo_min or dist>histo_max or floor(dist*invdx)==0: continue
            pos = <int> floor((dist-histo_min)*invdx)

            #pos -> idx3
            #shell = "new pos"

            """
            if mode_sel & 1: dest[ix2+pos]+=1.0*off_diag
            
            if mode_sel & 2 or mode_sel & 16:
                incr = (cpx*spx+cpy*spy+cpz*spz)/(clen*slen)
                if mode_sel & 2: dest[ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 16: dest[4*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 4 or mode_sel & 32:
                incr = (cpx*dx+cpy*dy+cpz*dz)/(clen*dist)
                if mode_sel & 4: dest[2*ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 32: dest[5*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 8 or mode_sel & 64:
                incr = (spx*dx+spy*dy+spz*dz)/(slen*dist)
                if mode_sel & 8: dest[3*ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 64: dest[6*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            """

            ix3 = pos*nshells
            ix4 = shell

            if mode_sel & 1: dest[ix2+ix3+ix4]+=1.0*off_diag
            
            if mode_sel & 2 or mode_sel & 16:
                incr = (cpx*spx+cpy*spy+cpz*spz)/(clen*slen)
                if mode_sel & 2: dest[ix1+ix2+ix3+ix4]+=incr*off_diag
                if mode_sel & 16: dest[4*ix1+ix2+ix3+ix4]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 4 or mode_sel & 32:
                incr = (cpx*dx+cpy*dy+cpz*dz)/(clen*dist)
                if mode_sel & 4: dest[2*ix1+ix2+ix3+ix4]+=incr*off_diag
                if mode_sel & 32: dest[5*ix1+ix2+ix3+ix4]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 8 or mode_sel & 64:
                incr = (spx*dx+spy*dy+spz*dz)/(slen*dist)
                if mode_sel & 8: dest[3*ix1+ix2+ix3+ix4]+=incr*off_diag
                if mode_sel & 64: dest[6*ix1+ix2+ix3+ix4]+=(1.5*incr*incr-0.5)*off_diag


cdef void calcHistoVoronoiNonSelf(double *core_xyz, double *surr_xyz, double *core_dip, double *surr_dip, 
                    np.ndarray[np.float64_t, ndim=1] np_dest, np.ndarray[np.float64_t,ndim=1] np_box_dim, 
                    int ncore, int nsurround, float histo_min, float histo_max, int histo_n, double invdx, int mode_sel, int nm_core, int nm_surround, int nshells, int * delaunay_matrix):

    #cdef int i, j, pos, ix1=ncore*histo_n, ix2, j_begin=0 # Change ix1, ix2
    cdef int i, j, pos, ix1=nshells*ncore*histo_n, ix2, j_begin=0 #ROLLBACK: ERASE THIS, UNCOMMENT ABOVE
    cdef double dx, dy, dz, dist, cpx, cpy, cpz, spx, spy, spz, slen, clen, incr, off_diag=1.0
    cdef double *dest = <double*> np_dest.data
    cdef double *box_dim = <double*> np_box_dim.data
    cdef int ds_pos1, ds_pos2, shell, ix3, ix4
    cdef int apr_core = ncore / nm_core
    cdef int apr_surround = nsurround / nm_surround


    #TODO: Change with nm_core, int nm_surround, int delaunay_matrix, nshells

    for i in prange(ncore, nogil=True):
        #ix2=i*histo_n
        ix2 = i*nshells*histo_n #ROLLBACK: ERASE THESE 2, UNCOMMENT ABOVE
        ds_pos1 = i/apr_core 

        if mode_sel & 2 or mode_sel & 4 or mode_sel & 16 or mode_sel & 32:
            cpx = core_dip[i*3]
            cpy = core_dip[i*3+1]
            cpz = core_dip[i*3+2]
            clen = sqrt(cpx*cpx+cpy*cpy+cpz*cpz)
        if ncore == nsurround:
            j_begin = i
        else:
            j_begin = 0
        for j in range(j_begin,nsurround):
            ds_pos2 = j/apr_surround # ROLLBACK: ERASE THESE 3
            if(ds_pos1 == ds_pos2): continue
            shell = delaunay_matrix[ds_pos1*nsurround + ds_pos2] #Correctly shaped?
            shell = shell-1
            if shell <  0:       shell = nshells-1
            if shell >= nshells: shell = nshells-1

            if ncore == nsurround and i != j:
                off_diag = 2.0
            else:
                off_diag = 1.0
            if mode_sel & 2 or mode_sel & 8 or mode_sel & 16 or mode_sel & 64:
                spx = surr_dip[j*3]
                spy = surr_dip[j*3+1]
                spz = surr_dip[j*3+2]
                slen = sqrt(spx*spx+spy*spy+spz*spz)

            dx = surr_xyz[j*3]   - core_xyz[i*3]
            dy = surr_xyz[j*3+1] - core_xyz[i*3+1]
            dz = surr_xyz[j*3+2] - core_xyz[i*3+2]

            if fabs(dx) > box_dim[3]: dx = dx-((0.0 < dx) - (dx < 0.0))*box_dim[0]
            if fabs(dy) > box_dim[4]: dy = dy-((0.0 < dy) - (dy < 0.0))*box_dim[1]
            if fabs(dz) > box_dim[5]: dz = dz-((0.0 < dz) - (dz < 0.0))*box_dim[2]
            dist = sqrt(dx*dx+dy*dy+dz*dz)
            if dist<histo_min or dist>histo_max or floor(dist*invdx)==0: continue
            pos = <int> floor((dist-histo_min)*invdx)

            #pos -> idx3
            #shell = "new pos"

            """
            if mode_sel & 1: dest[ix2+pos]+=1.0*off_diag
            
            if mode_sel & 2 or mode_sel & 16:
                incr = (cpx*spx+cpy*spy+cpz*spz)/(clen*slen)
                if mode_sel & 2: dest[ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 16: dest[4*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 4 or mode_sel & 32:
                incr = (cpx*dx+cpy*dy+cpz*dz)/(clen*dist)
                if mode_sel & 4: dest[2*ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 32: dest[5*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 8 or mode_sel & 64:
                incr = (spx*dx+spy*dy+spz*dz)/(slen*dist)
                if mode_sel & 8: dest[3*ix1+ix2+pos]+=incr*off_diag
                if mode_sel & 64: dest[6*ix1+ix2+pos]+=(1.5*incr*incr-0.5)*off_diag
            """

            ix3 = pos*nshells
            ix4 = shell

            if mode_sel & 1: dest[ix2+ix3+ix4]+=1.0*off_diag
            
            if mode_sel & 2 or mode_sel & 16:
                incr = (cpx*spx+cpy*spy+cpz*spz)/(clen*slen)
                if mode_sel & 2: dest[ix1+ix2+ix3+ix4]+=incr*off_diag
                if mode_sel & 16: dest[4*ix1+ix2+ix3+ix4]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 4 or mode_sel & 32:
                incr = (cpx*dx+cpy*dy+cpz*dz)/(clen*dist)
                if mode_sel & 4: dest[2*ix1+ix2+ix3+ix4]+=incr*off_diag
                if mode_sel & 32: dest[5*ix1+ix2+ix3+ix4]+=(1.5*incr*incr-0.5)*off_diag
            if mode_sel & 8 or mode_sel & 64:
                incr = (spx*dx+spy*dy+spz*dz)/(slen*dist)
                if mode_sel & 8: dest[3*ix1+ix2+ix3+ix4]+=incr*off_diag
                if mode_sel & 64: dest[6*ix1+ix2+ix3+ix4]+=(1.5*incr*incr-0.5)*off_diag


class RDF_voronoi(object):
    """
    class RDF

    A container object for calculating radial distribution functions.
    """

    def __init__(self, modes, histo_min, histo_max, histo_dr, sel1_n, sel2_n, sel1_nmol, sel2_nmol, box_x, nshells, box_y=None, box_z=None, use_cuda=False, norm_volume=None, exclude_self_mol=False):
        """
        RDF.__init__(modes, histo_min, histo_max, histo_dr, sel1_n, sel2_n, sel1_nmol, sel2_nmol, box_x, box_y=None, box_z=None, use_cuda=False)

        Creates a new rdf container.

        modes .. a list of gfunctions to calculate
                 e.g. ["all"] -> calculate all gfunctions
                      ["000","110","220"] -> calculate only g000, g110 and g220
        histo_min .. lower bound of histogram
        histo_max .. upper bound of histogram
        histo_dr  .. histogram bin width
        sel1_n    .. number of particles in first selection
        sel2_n    .. number of particles in second selection
        sel1_nmol .. number of molecules in first selection; needed for correct normalisation, if multiple atoms from the same molecule are selected
        sel2_nmol .. number of molecules in second selection
        box_x     .. box length, x dim
        box_y     .. box length, y dim (optional; if None, y=x)
        box_z     .. box length, z dim (optional; if None, z=x)
        use_cuda  .. whether to use cuda, if available (default=False)
        norm_volume .. volume to use for calculating the density during normalization (default = None)

        Usage example:
        my_rdf = RDF(["all"], 0.0, 50.0, 0.05, 200, 200, 200, 200, 25.0)
        """

        """
        if use_cuda and cu.cuda:
            self.cuda = True
        elif use_cuda and not cu.cuda:
            print "No suitable CUDA device was found to calculate radial distribution functions. Using CPU instead."
            self.cuda = False
        else:
            self.cuda = False

        if self.cuda:
            self.dtype = np.float32
        else:
            self.dtype = np.float64
        """
        self.dtype = np.float64
            
        # always initialize the histogram to 7 dimensions, but only fill the desired ones
        cdef int mode_sel = 0

        # select the gfunctions to calculate using bitwise operators
        for mode in modes:
            if mode == "all": mode_sel = mode_sel | 1 | 2 | 4 | 8 | 16 | 32 | 64
            if mode == "000": mode_sel = mode_sel | 1
            if mode == "110": mode_sel = mode_sel | 2
            if mode == "101": mode_sel = mode_sel | 4
            if mode == "011": mode_sel = mode_sel | 8
            if mode == "220": mode_sel = mode_sel | 16
            if mode == "202": mode_sel = mode_sel | 32
            if mode == "022": mode_sel = mode_sel | 64

        # remember the selection
        self.mode_sel = np.int32(mode_sel)
        
        # set histogram parameters
        self.histo_min = np.float32(histo_min)
        self.histo_max = np.float32(histo_max)
        self.histo_dr  = np.round(self.dtype(histo_dr),5)
        self.histo_inv_dr = self.dtype(1.0 / self.histo_dr)
        self.histo_n = np.int32((histo_max-histo_min)/histo_dr+0.5)
        self.nshells = np.int32(nshells)
        self.exclude_self_mol = exclude_self_mol

        self.box_dim = np.zeros(6,dtype=self.dtype)
        self.box_dim[0] = box_x
        self.box_dim[3] = box_x/2.0
        if box_y: 
            self.box_dim[1] = box_y
            self.box_dim[4] = box_y/2.0
        else: 
            self.box_dim[1] = self.box_dim[0]
            self.box_dim[4] = self.box_dim[3]
        if box_z: 
            self.box_dim[2] = box_z
            self.box_dim[5] = box_z/2.0
        else: 
            self.box_dim[2] = self.box_dim[0]
            self.box_dim[5] = self.box_dim[3]
           
        self.volume = self.box_dim[0] * self.box_dim[1] * self.box_dim[2]

        # remember volume for each frame for normalisation
        self.volume_list = []

        self.norm_volume = norm_volume
        
        # set particle numbers
        self.n1 = np.int32(sel1_n)
        self.n2 = np.int32(sel2_n)
        self.nmol1 = sel1_nmol
        self.nmol2 = sel2_nmol

        self.oneistwo = None

        # set frame counter to 0
        self.ctr = 0

        # initialize histogram array
        self.histogram = np.zeros(self.nshells*self.n1*self.histo_n*7,dtype=self.dtype)
        self.histogram_out = np.zeros(self.nshells*self.histo_n*7,dtype=np.float64)
        
        # print, if CUDA device will be used and initialize arrays
        """
        if self.cuda and use_cuda:
            # initialize gpu arrays

            self.histo_gpu=cu.drv.mem_alloc(self.histogram.nbytes)
            cu.drv.memcpy_htod(self.histo_gpu, self.histogram)

            self.box_dim_gpu=cu.drv.mem_alloc(self.box_dim.nbytes)
            cu.drv.memcpy_htod(self.box_dim_gpu, self.box_dim)

            arr_sel1 = np.zeros((self.n1,3),dtype=np.float32)
            arr_sel2 = np.zeros((self.n2,3),dtype=np.float32)
            self.coor_sel1_gpu = cu.drv.mem_alloc(arr_sel1.nbytes)
            self.coor_sel2_gpu = cu.drv.mem_alloc(arr_sel2.nbytes)
            
            if mode_sel & 2 or mode_sel & 4 or mode_sel & 16 or mode_sel & 32:
                self.dip_sel1_gpu = cu.drv.mem_alloc(arr_sel1.nbytes)
            if mode_sel & 2 or mode_sel & 8 or mode_sel & 16 or mode_sel & 64:
                self.dip_sel2_gpu = cu.drv.mem_alloc(arr_sel2.nbytes)

            # set gpu block and grid
            block_x = cu.dev.get_attribute(cu.drv.device_attribute.MULTIPROCESSOR_COUNT) * 2
            block_y = cu.dev.get_attribute(cu.drv.device_attribute.MULTIPROCESSOR_COUNT) * 2
            block_z = 1
            grid_x = int(self.n1/block_x)
            if self.n1%block_x: grid_x+=1
            grid_y = int(self.n2/block_y)
            if self.n2%block_y: grid_y+=1
            self.block=(block_x,block_y,block_z)
            self.grid=(grid_x,grid_y)
        """

            
    def update_box(self,box_x,box_y,box_z):
        """
        update_box(box_x,box_y,box_z)

        Update the box dimensions, when analysing NpT trajectories.
        """
        self.box_dim[0] = box_x
        self.box_dim[3] = box_x/2.0
        self.box_dim[1] = box_y
        self.box_dim[4] = box_y/2.0
        self.box_dim[2] = box_z
        self.box_dim[5] = box_z/2.0

        """
        if self.cuda:
            cu.drv.memcpy_htod(self.box_dim_gpu, self.box_dim)

        self.volume = self.box_dim[0] * self.box_dim[1] * self.box_dim[2]
        """

    def calcFrame(self,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] coor_sel1,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] coor_sel2,
                  np.ndarray[np.int32_t,  ndim=2,mode="c"] delaunay_matrix,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] dip_sel1 = None,
                  np.ndarray[np.float64_t,ndim=2,mode="c"] dip_sel2 = None,
                  pre_coor_sel1_gpu=None, pre_coor_sel2_gpu=None, 
                  pre_dip_sel1_gpu=None, pre_dip_sel2_gpu=None):
        """
        calcFrame(coor_sel1, coor_sel2, dip_sel1=None, dip_sel2=None,
                  pre_coor_sel1_gpu=None, pre_coor_sel2_gpu=None, 
                  pre_dip_sel1_gpu=None, pre_dip_sel2_gpu=None)

        Args:
             coor_sel1 .. numpy-array (dtype=numpy.float64, ndim=2, mode="c") containing the coordinates of the first selection
             coor_sel2 .. numpy-array containing the coordinates of the second selection
             dip_sel1  .. numpy-array containing the dipoles of the first selection (optional, only needed for 110, 101, 220 and 202)
             dip_sel2  .. numpy-array containing the dipoles of the second selection (optional, only needed for 110, 011, 220 and 022)

             pre-allocated arrays: PyCUDA GPU array objects obtained from drv.memalloc()

        Pass the data needed to calculate the RDF for the current frame and do the calculation.
        Optionally, pre-allocated arrays on the GPU can be used. This is useful, if they are reused for more than
        one rdf calculation.
        """
        
        # Check if dipole moments are needed and, if yes, were passed
        if dip_sel1 is None and ((<int> self.mode_sel) & 2 or (<int> self.mode_sel) & 4 or (<int> self.mode_sel) & 16 or (<int> self.mode_sel) & 32):
            raise TypeError("No dipole moment array for the first selection was passed, although the requested gfunctions need it!")

        if dip_sel2 is None and ((<int> self.mode_sel) & 2 or (<int> self.mode_sel) & 8 or (<int> self.mode_sel) & 16 or (<int> self.mode_sel) & 64):
            raise TypeError("No dipole moment array for the second selection was passed, although the requested gfunctions need it!")

        if self.oneistwo is None:
            if (coor_sel1[0] == coor_sel2[0]).all() and self.nmol1 == self.nmol2:
                self.oneistwo = True
            else:
                self.oneistwo = False

        # Increment frame counter
        self.ctr+=1

        # add current volume to list
        self.volume_list.append(self.volume)

        """
        if self.cuda:
            # allocate memory on cuda device and copy data to device
            if pre_coor_sel1_gpu:
                self.coor_sel1_gpu = pre_coor_sel1_gpu
            else:
                cu.drv.memcpy_htod(self.coor_sel1_gpu, coor_sel1.astype(np.float32))
            if pre_coor_sel2_gpu:
                self.coor_sel2_gpu = pre_coor_sel2_gpu
            else:
                cu.drv.memcpy_htod(self.coor_sel2_gpu, coor_sel2.astype(np.float32))
            # if no dipole information was given, just point at the coordinates instead to keep the gpu function happy
            if dip_sel1 != None:
                if pre_dip_sel1_gpu:
                    self.dip_sel1_gpu = pre_dip_sel1_gpu
                else:
                    cu.drv.memcpy_htod(self.dip_sel1_gpu, dip_sel1.astype(np.float32))
            else:
                self.dip_sel1_gpu = self.coor_sel1_gpu
            if dip_sel2 != None:
                if pre_dip_sel2_gpu:
                    self.dip_sel2_gpu = pre_dip_sel2_gpu
                else:
                    cu.drv.memcpy_htod(self.dip_sel2_gpu, dip_sel2.astype(np.float32))
            else:
                self.dip_sel2_gpu = self.coor_sel2_gpu

            cu.calcHistoGPU(self.coor_sel1_gpu, self.coor_sel2_gpu, self.dip_sel1_gpu, self.dip_sel2_gpu, 
                           self.histo_gpu, self.box_dim_gpu, self.n1, self.n2, self.histo_min, 
                           self.histo_max, self.histo_n, self.histo_inv_dr, self.mode_sel, self.block, self.grid)

        else:
        """
        if(self.exclude_self_mol):
            calcHistoVoronoiNonSelf(<double*> coor_sel1.data, <double*> coor_sel2.data, 
                  <double*> dip_sel1.data, <double*> dip_sel2.data, 
                  self.histogram, self.box_dim, 
                  self.n1, self.n2, self.histo_min, self.histo_max, 
                  self.histo_n, self.histo_inv_dr, self.mode_sel, self.nmol1, self.nmol2, self.nshells, <int*> delaunay_matrix.data)
        else:
            calcHistoVoronoi(<double*> coor_sel1.data, <double*> coor_sel2.data, 
                  <double*> dip_sel1.data, <double*> dip_sel2.data, 
                  self.histogram, self.box_dim, 
                  self.n1, self.n2, self.histo_min, self.histo_max, 
                  self.histo_n, self.histo_inv_dr, self.mode_sel, self.nmol1, self.nmol2, self.nshells, <int*> delaunay_matrix.data)
            
    def _addHisto(self,np.ndarray[np.float64_t,ndim=1] histo_out, np.ndarray[np.float64_t,ndim=1] histo):
        """
        Add the histograms of all core particles, after they were calculated in parallel.
        """
        
        #cdef int histo_n = self.histo_n, n1=self.n1, i, j, k, ix, ix2 = self.n1*self.histo_n
        cdef int histo_n = self.histo_n, n1=self.n1, i, j, k, ix, ix2 = self.n1*self.histo_n*self.nshells

        cdef double* histogram_out = <double*> histo_out.data
        cdef double* histogram = <double*> histo.data
        cdef int shell
        cdef int nshells = self.nshells

        # add histograms
        """
        for i in prange(histo_n, nogil=True):
            for j in range(n1):
                ix = histo_n * j
                for k in range(7):
                    histogram_out[k*histo_n+i]+=histogram[k*ix2+ix+i]
        """
        for shell in range(nshells):
            for i in prange(histo_n, nogil=True):
                for j in range(n1): #n_aufpunkte
                    ix = nshells * histo_n * j
                    for k in range(7):
                        histogram_out[k*histo_n*nshells+i*nshells+shell]+=histogram[k*ix2+ix+i*nshells+shell]

    def _normHisto(self):
        """
        Normalize the histogram.
        """

        if self.norm_volume:
            volume = self.norm_volume
        else:
            volume = 0.0
            for v in self.volume_list:
                volume+=v
            volume/=len(self.volume_list)

        cdef int i, j, ix, ix2 = self.n1 * self.histo_n, shell
        PI = 3.14159265358979323846
        
        # norm histograms
        if self.oneistwo:
            rho = float(self.nmol1 * (self.nmol1 - 1)) / volume
        else:
            rho = float(self.nmol1 * self.nmol2) / volume


        """
        for i in range(self.histo_n):
            r = self.histo_min + float(i) * self.histo_dr
            r_out = r + self.histo_dr
            slice_vol = 4.0/3.0 * PI * (r_out**3 - r**3)
            norm = rho*slice_vol*float(self.ctr)
            for j in range(7):
                self.histogram_out[j*self.histo_n+i]/=norm
        """
        for shell in range(self.nshells):
            for i in range(self.histo_n):
                r = self.histo_min + float(i) * self.histo_dr
                r_out = r + self.histo_dr
                slice_vol = 4.0/3.0 * PI * (r_out**3 - r**3)
                norm = rho*slice_vol*float(self.ctr)
                for j in range(7):
                    self.histogram_out[j*self.histo_n*self.nshells+i*self.nshells+shell]/=norm

    def scale(self, value):
        self.histogram_out[:] *= value
        self.histogram[:] *= value

    def write(self, filename="rdf"):
        """
        RDF.write(filename="rdf")

        Args:
             filename .. name of the file to be written to. Each type of rdf will be written to a separate file
                         with the ending specifying the type.
                         
        Norm the calculated histograms and write them to the specified file.
        """

        """
        if self.cuda:
            cu.drv.memcpy_dtoh(self.histogram, self.histo_gpu)
        """
            
        self._addHisto(self.histogram_out, self.histogram.astype(np.float64))
        self._normHisto()

        for shell in range(self.nshells):        
            if self.mode_sel & 1:  f000 = open(filename+"_shell"+str(shell+1)+"_g000.dat",'w')
            if self.mode_sel & 2:  f110 = open(filename+"_shell"+str(shell+1)+"_g110.dat",'w')
            if self.mode_sel & 4:  f101 = open(filename+"_shell"+str(shell+1)+"_g101.dat",'w')
            if self.mode_sel & 8:  f011 = open(filename+"_shell"+str(shell+1)+"_g011.dat",'w')
            if self.mode_sel & 16: f220 = open(filename+"_shell"+str(shell+1)+"_g220.dat",'w')
            if self.mode_sel & 32: f202 = open(filename+"_shell"+str(shell+1)+"_g202.dat",'w')
            if self.mode_sel & 64: f022 = open(filename+"_shell"+str(shell+1)+"_g022.dat",'w')

            for i in range(self.histo_n):
                r = self.histo_min+i*self.histo_dr+self.histo_dr*0.5
                if self.mode_sel & 1:  f000.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[0*self.histo_n*self.nshells+i*self.nshells+shell]))
                if self.mode_sel & 2:  f110.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[1*self.histo_n*self.nshells+i*self.nshells+shell]))
                if self.mode_sel & 4:  f101.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[2*self.histo_n*self.nshells+i*self.nshells+shell]))
                if self.mode_sel & 8:  f011.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[3*self.histo_n*self.nshells+i*self.nshells+shell]))
                if self.mode_sel & 16: f220.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[4*self.histo_n*self.nshells+i*self.nshells+shell]))
                if self.mode_sel & 32: f202.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[5*self.histo_n*self.nshells+i*self.nshells+shell]))
                if self.mode_sel & 64: f022.write("%5.5f\t%5.5f\n" % (r,self.histogram_out[6*self.histo_n*self.nshells+i*self.nshells+shell]))

    def resetArrays(self):
        """
        Reset histogram arrays to zero
        """
        self.histogram_out[:] = 0.0
        self.histogram[:] = 0.0

        
    def printInfo(self):
        """
        Print some information about the RDF container.
        """

        print("Modes: ", self.mode_sel)
        print("Frames read: ", self.ctr)
