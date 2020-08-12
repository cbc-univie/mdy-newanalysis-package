from __future__ import print_function
import MDAnalysis
import numpy as np
from newanalysis.correl import correlateParallel, correlate
from newanalysis.functions import atomsPerResidue, residueFirstAtom
#from newanalysis.helpers import dipByResidue, velcomByResidue                                       #old nomenclature (see alternative code below)
from newanalysis.functions import centerOfMassByResidue, dipoleMomentByResidue, velcomByResidue      #newer nomenclature
import os
import time

###########################################################################################################################
# Correlation functions
###########################################################################################################################
def correlXYZ(data, result):
    data = np.ascontiguousarray(data.T)
    for i in range(3):
        result += correlate(data[i], data[i], ltc=1)

def crosscorrelXYZ(data1, data2, result, both=True):
    data1 = np.ascontiguousarray(data1.T)
    data2 = np.ascontiguousarray(data2.T)
    for i in range(3):
        result += correlate(data1[i], data2[i], ltc=1)
        if both:
            result += correlate(data2[i], data1[i], ltc=1)

###########################################################################################################################
# Trajectory
###########################################################################################################################
skip=10
base="../../data/emim_dca_equilibrium/"
psf=base+'emim_dca.psf'
#Check PSF:
#if np.array_equal(MDAnalysis.Universe(psf).atoms.masses, MDAnalysis.Universe(psf).atoms.masses.astype(bool)):
#    print("Used wrong PSF format for S0 state (masses unreadable!)")
#    sys.exit()


u  = MDAnalysis.Universe(psf,base+"emim_dca_fine.dcd")
uv = MDAnalysis.Universe(psf,base+"emim_dca_fine_vel.dcd")

boxl = np.float64(round(u.coord.dimensions[0],4))
dt=round(u.trajectory.dt,4)

n = int(u.trajectory.n_frames/skip)
if u.trajectory.n_frames%skip != 0:
    n+=1
    
###########################################################################################################################
# molecular species
###########################################################################################################################
sel_cat1  = u.select_atoms("resname EMIM")
selv_cat1 = uv.select_atoms("resname EMIM")
ncat1     = sel_cat1.n_residues
mass_cat1 = sel_cat1.masses
charge_cat1 = sel_cat1.charges
com_cat1  = np.zeros((n,ncat1,3),dtype=np.float64)
mdcat1    = np.zeros((n,3),dtype=np.float64)
apr_cat1 = atomsPerResidue(sel_cat1)
rfa_cat1 = residueFirstAtom(sel_cat1)
print("Number EMIM   = ",ncat1)

sel_an1  = u.select_atoms("resname DCA")
selv_an1 = uv.select_atoms("resname DCA")
nan1     = sel_an1.n_residues
mass_an1 = sel_an1.masses
charge_an1 = sel_an1.charges
com_an1  = np.zeros((n,nan1, 3),dtype=np.float64)
mdan1    = np.zeros((n,3),dtype=np.float64)
apr_an1 = atomsPerResidue(sel_an1)
rfa_an1 = residueFirstAtom(sel_an1)
print("Number DCA  = ",nan1)
 
md  = np.zeros((n,3),dtype=np.float64)
j   = np.zeros((n,3),dtype=np.float64)
mdj_cat = np.zeros(n,dtype=np.float64)
mdj_an = np.zeros(n,dtype=np.float64)
jj  = np.zeros(n,dtype=np.float64)



###########################################################################################################################
# Analysis
###########################################################################################################################
ctr=0
start=time.time()
print("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d" % (ts.frame,u.trajectory.n_frames), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    uv.trajectory[ts.frame-1]
    
    # efficiently calculate center-of-mass coordinates
    coor_cat1 = np.ascontiguousarray(sel_cat1.positions,dtype='double')
    coor_an1 = np.ascontiguousarray(sel_an1.positions,dtype='double')

#   altervative code, where functions are directly called,
#    comment in: from newanalysis.helpers import ... (top of file)
#    comment out: from newanalysis.functions import ...(top of file)
#    com_an1  = sel_an1.center_of_mass(compound='residues')
#    com_cat1 = sel_cat1.center_of_mass(compound='residues')
#    mdcat1[ctr] += np.sum(dipByResidue(coor_cat1,charge_cat1,mass_cat1,ncat1,apr_cat1,rfa_cat1,com_cat1),axis=0)
#    mdan1[ctr] += np.sum(dipByResidue(coor_an1,charge_an1,mass_an1,nan1,apr_an1,rfa_an1,com_an1),axis=0)
#    md[ctr]     += mdcat1[ctr]+mdan1[ctr]
#    vels_cat1=np.ascontiguousarray(selv_cat1.positions*MDAnalysis.units.timeUnit_factor['AKMA'],dtype='double')
#    vels_an1= np.ascontiguousarray(selv_an1.positions*MDAnalysis.units.timeUnit_factor['AKMA'],dtype='double')
#    j[ctr] += np.sum(velcomByResidue(vels_cat1,mass_cat1,ncat1,apr_cat1,rfa_cat1))
#    j[ctr] -= np.sum(velcomByResidue(vels_an1,mass_an1,nan1,apr_an1,rfa_an1))
    
#    Code where functions can be called as usual
#    comment out: from newanalysis.helpers import ... (top of file)
#    comment in: from newanalysis.functions import ...(top of file)
    com_an1  = centerOfMassByResidue(sel_an1,coor=coor_an1,masses=mass_an1,apr=apr_an1,rfa=rfa_an1)
    com_cat1 = centerOfMassByResidue(sel_cat1,coor=coor_cat1,masses=mass_cat1,apr=apr_cat1,rfa=rfa_cat1)
    mdcat1[ctr] += np.sum(dipoleMomentByResidue(sel_cat1,coor=coor_cat1,charges=charge_cat1,masses=mass_cat1,com=com_cat1,apr=apr_cat1,rfa=rfa_cat1),axis=0)
    mdan1[ctr]  += np.sum(dipoleMomentByResidue(sel_an1,coor=coor_an1,charges=charge_an1,masses=mass_an1,com=com_an1,apr=apr_an1,rfa=rfa_an1),axis=0)
    md[ctr]     += mdcat1[ctr]+mdan1[ctr]
    j[ctr] += np.sum(velcomByResidue(selv_cat1,masses=mass_cat1,apr=apr_cat1,rfa=rfa_cat1))
    j[ctr] -= np.sum(velcomByResidue(selv_an1,masses=mass_an1,apr=apr_an1,rfa=rfa_an1))

    ctr+=1

print ("<MD(0) * J(t)> ...")
crosscorrelXYZ(mdcat1, j, mdj_cat)
crosscorrelXYZ(mdan1, j, mdj_an)

print ("<J(0) * J(t)> ...")
correlXYZ(j, jj)

fjj  = open ('jj.dat','w')
fmdj_cat = open ('mdj_emim.dat','w')
fmdj_an  = open ('mdj_dca.dat','w')
for i in range(ctr):
    fjj.write("%5.5f\t%5.5f\n"   % (i*skip*dt, jj[i]))
    fmdj_cat.write("%5.5f\t%5.5f\n"  % (i*skip*dt, mdj_cat[i]))
    fmdj_an.write("%5.5f\t%5.5f\n"  % (i*skip*dt, mdj_an[i]))
fjj.close()
fmdj_cat.close()
fmdj_an.close()

