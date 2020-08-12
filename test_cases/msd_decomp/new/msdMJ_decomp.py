from __future__ import print_function
import MDAnalysis
import numpy as np
from newanalysis.diffusion import msdCOM, msdMJ, unfoldTraj, msdMJdecomp
from newanalysis.correl import correlate
import os
import time
import sys

########################################################################################################################################
def correlXYZ(data, result, ltc=0):
    data = np.ascontiguousarray(data.T)
    for i in range(3):
        tmp = correlate(data[i], data[i], ltc=ltc)
        for j in range(len(data[i])):
            result[j] += tmp[j]

           
########################################################################################################################################
# trajectories
########################################################################################################################################
base='../../data/emim_dca_equilibrium/'
psf=base+'emim_dca.psf'
#Check PSF:
if np.array_equal(MDAnalysis.Universe(psf).atoms.masses, MDAnalysis.Universe(psf).atoms.masses.astype(bool)):
    print("Used wrong PSF format for S0 state (masses unreadable!)")
    sys.exit()

skip = 20
u=MDAnalysis.Universe(psf,"../../data/emim_dca_equilibrium/emim_dca.dcd")

boxl=round(u.coord.dimensions[0],4)
dt=round(u.trajectory.dt,4)
n = int(u.trajectory.n_frames/skip)
if u.trajectory.n_frames%skip != 0:
    n+=1


########################################################################################################################################
# molecular species
########################################################################################################################################
sel_cat  = u.select_atoms("resname EMIM")
sel_an  = u.select_atoms("resname DCA")


ncat = sel_cat.n_residues
nan = sel_an.n_residues


com_cat = np.zeros((int(n),int(ncat),3),dtype=np.float64)
com_an  = np.zeros((int(n),int(nan),3),dtype=np.float64)


########################################################################################################################################
# analysis
########################################################################################################################################
ctr=0
start=time.time()
print("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d" % (ts.frame,u.trajectory.n_frames), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    com_cat[ctr] = sel_cat.center_of_mass(compound='residues')
    com_an[ctr] = sel_an.center_of_mass(compound='residues')

    ctr+=1

print("unfolding coordinates ..")
unfoldTraj(com_cat,boxl)
unfoldTraj(com_an, boxl)

# Classical mean-squared displacement of coordinates
print("calculating < Delta r^2(t) > ...")
msdcat = msdCOM(com_cat)
filename = 'msd_emim.dat'
f1=open(filename,'w')
for i in range(len(msdcat)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdcat[i]))
f1.close()

msdan = msdCOM(com_an)
filename = 'msd_dca.dat'
f1=open(filename,'w')
for i in range(len(msdan)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdan[i]))
f1.close()


# mean-squared displacement of collective translational dipole moment
print("calculating msdMJ ..")
msdmj = msdMJ(com_cat,com_an)
filename = 'msdMJ.dat'
f1=open(filename,'w')
for i in range(len(msdmj)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj[i]))
f1.close()

# decomposition of mean-squared displacement of collective translational dipole moment
msdmj_cat = msdMJdecomp(com_cat, com_an, com_cat, 1)
filename = 'msdMJ_cat.dat'
f1=open(filename,'w')
for i in range(len(msdmj_cat)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj_cat[i]))
f1.close()

msdmj_an = msdMJdecomp(com_cat, com_an, com_an, -1)
filename = 'msdMJ_an.dat'
f1=open(filename,'w')
for i in range(len(msdmj_an)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj_an[i]))
f1.close()

