from __future__ import print_function
import MDAnalysis
import numpy as np
from newanalysis.diffusion import msdCOM, unfoldTraj, msdMJ
import time

###########################################################################################################################
# Trajectory
###########################################################################################################################
base='../../data/emim_dca_equilibrium/'
psf=base+'emim_dca.psf'
#Check PSF:
if np.array_equal(MDAnalysis.Universe(psf).atoms.masses, MDAnalysis.Universe(psf).atoms.masses.astype(bool)):
    print("Used wrong PSF format for S0 state (masses unreadable!)")
    sys.exit()


skip=20
u=MDAnalysis.Universe(psf,"../../data/emim_dca_equilibrium/emim_dca.dcd")
boxl = np.float64(round(u.coord.dimensions[0],4))
dt=round(u.trajectory.dt,4)

n = int(u.trajectory.n_frames/skip)
if u.trajectory.n_frames%skip != 0:
    n+=1

###########################################################################################################################
# molecular species
###########################################################################################################################
sel_cat1  = u.select_atoms("resname EMIM")
ncat1     = sel_cat1.n_residues
com_cat1  = np.zeros((n,ncat1,3),dtype=np.float64)
print("Number EMIM   = ",ncat1)

sel_an1  = u.select_atoms("resname DCA")
nan1     = sel_an1.n_residues
com_an1  = np.zeros((n,nan1, 3),dtype=np.float64)
print("Number DCA   = ",nan1)


###########################################################################################################################
# Analysis
###########################################################################################################################
ctr=0
start=time.time()
print("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d" % (ts.frame,len(u.trajectory)), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    com_an1[ctr] = sel_an1.center_of_mass(compound='residues')
    com_cat1[ctr]= sel_cat1.center_of_mass(compound='residues')
    ctr+=1

print("unfolding coordinates ..")
unfoldTraj(com_cat1,boxl)
unfoldTraj(com_an1, boxl)

print("calculating msd ..")
msd_an1  = msdCOM(com_an1)
msd_cat1 = msdCOM(com_cat1)

fan1=open("msd_an.dat",'w')
fcat1=open("msd_cat.dat",'w')

for i in range(ctr):
    fan1.write("%5.5f\t%5.5f\n"  % (i*skip*dt, msd_an1[i]))
    fcat1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msd_cat1[i]))
    
print("calculating msdMJ ..")
msdmj = msdMJ(com_cat1,com_an1)
f1=open("msdMJ.dat",'w')
for i in range(ctr):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, msdmj[i]))
f1.close()
