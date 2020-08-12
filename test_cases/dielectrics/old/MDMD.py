from __future__ import print_function
import MDAnalysis
import numpy as np
from MDAnalysis.newanalysis.correl import correlateParallel
import os
import time

###########################################################################################################################
# Trajectory
###########################################################################################################################
skip=20
u=MDAnalysis.Universe("../../data/emim_dca_equilibrium/emim_dca.psf","../../data/emim_dca_equilibrium/emim_dca.dcd")

boxl = np.float64(round(u.coord.dimensions[0],4))
dt=round(u.trajectory.dt,4)

n = int(u.trajectory.numframes/skip)
if u.trajectory.numframes%skip != 0:
    n+=1

###########################################################################################################################
# molecular species
###########################################################################################################################
sel_cat1  = u.selectAtoms("resname EMIM")
ncat1     = sel_cat1.numberOfResidues()
mass_cat1 = sel_cat1.masses()
charge_cat1 = sel_cat1.charges()
com_cat1  = np.zeros((n,ncat1,3),dtype=np.float64)
mdcat1    = np.zeros((n,3),dtype=np.float64)
print("Number EMIM   = ",ncat1)

sel_an1  = u.selectAtoms("resname DCA")
nan1     = sel_an1.numberOfResidues()
mass_an1 = sel_an1.masses()
charge_an1 = sel_an1.charges()
com_an1  = np.zeros((n,nan1, 3),dtype=np.float64)
mdan1    = np.zeros((n,3),dtype=np.float64)
print("Number DCA    = ",nan1)


###########################################################################################################################
# Analysis
###########################################################################################################################
md = np.zeros((n,3),dtype=np.float64)

ctr=0
start=time.time()
print("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d" % (ts.frame,u.trajectory.numframes), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    
    # efficiently calculate center-of-mass coordinates
    coor_an1  = sel_an1.get_positions()
    coor_cat1 = sel_cat1.get_positions()
    
    com_an1   = sel_an1.centerOfMassByResidue(coor=coor_an1,masses=mass_an1)
    com_cat1  = sel_cat1.centerOfMassByResidue(coor=coor_cat1,masses=mass_cat1)

    mdcat1[ctr] += np.sum(sel_cat1.dipoleMomentByResidue(coor=coor_cat1,charges=charge_cat1,masses=mass_cat1,com=com_cat1),axis=0)
    mdan1[ctr]  += np.sum(sel_an1.dipoleMomentByResidue(coor=coor_an1,charges=charge_an1,masses=mass_an1,com=com_an1),axis=0)
    md[ctr]     += mdcat1[ctr]+mdan1[ctr]
    ctr+=1
    
print("calculating <MD(0)MD(t)>")

# total
md = np.ascontiguousarray(md.T)
md0mdt = np.zeros(n)
correlateParallel(md,md,md0mdt,ltc=1)

f1 = open("md0mdt.dat",'w')
for i in range(len(md0mdt)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, md0mdt[i]))
f1.close()

# cations
mdcat1 = np.ascontiguousarray(mdcat1.T)
md0mdt = np.zeros(n)
correlateParallel(mdcat1,md,md0mdt,ltc=1)
f1 = open("md0mdt_emim.dat",'w')
for i in range(len(md0mdt)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, md0mdt[i]))
f1.close()


# anions
mdan1 = np.ascontiguousarray(mdan1.T)
md0mdt = np.zeros(n)
correlateParallel(mdan1,md,md0mdt,ltc=1)
f1 = open("md0mdt_dca.dat",'w')
for i in range(len(md0mdt)):
    f1.write("%5.5f\t%5.5f\n" % (i*skip*dt, md0mdt[i]))
f1.close()


