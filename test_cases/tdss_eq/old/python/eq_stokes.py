from __future__ import print_function
import MDAnalysis
import numpy as np
import os
import time
from MDAnalysis.newanalysis.helpers import calcEnergyAtomic
from MDAnalysis.newanalysis.unfold import minDistCenterBox
from sys import exit
from copy import deepcopy
import h5py
import cython
from datetime import timedelta
from MDAnalysis.newanalysis.correl import correlate

base='../../../data/1mq_swm4_equilibrium/'
psf_s0=base+'mqs0_swm4_1000.psf'
psf_s1=base+'mqs1_swm4_1000.psf'


skip = 1
u0=MDAnalysis.Universe(psf_s0)
u1=MDAnalysis.Universe(psf_s1,base+"1mq_swm4.dcd")

dq=u1.MQS1.charges()-u0.MQS0.charges()

for i in range(len(u1.MQS1.atoms)):
    u1.MQS1.atoms[i].charge = dq[i]

sel=u1.selectAtoms('all')
mqs1=u1.selectAtoms('resname MQS1')

charges=u1.residues.charges()
masses=u1.residues.masses()
apr=sel.atomsPerResidue()
rfa=sel.residueFirstAtom()

boxl=round(u1.coord.dimensions[0],4)
dt=round(u1.trajectory.dt,4)

isolute=0
nmol=1001

n = int((u1.trajectory.numframes)/skip)
if u1.trajectory.numframes%skip != 0:
    n+=1
    
epa=np.zeros((n))

print("Calculating energy function...")
ctr=0
start=time.time()
print("")
for ts in u1.trajectory[::skip]:
    print("\033[1AFrame %d of %d (%4.1f%%)" % (ts.frame,u1.trajectory.numframes,ts.frame/u1.trajectory.numframes*100),
          "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
          "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/ts.frame * (u1.trajectory.numframes-ts.frame)))[:7]))
    coor=np.ascontiguousarray(sel.get_positions())
    coms=sel.centerOfMassByResidue(coor=coor,masses=masses)
    com_mq = mqs1.centerOfMass()
    minDistCenterBox(com_mq, coms, coor, boxl, apr, rfa)
    tmp=calcEnergyAtomic(coor,charges,apr,rfa,isolute,nmol)
    epa[ctr]+=tmp.sum()  
    ctr+=1

print("")
print("Correlating data...")

times=np.zeros((n))
for i in range(n):
    times[i]=i*dt*skip
    
ycorr_total=0
    
tmp=sum(epa)/n
epa-=tmp


ycorr_total=correlate(np.array(epa,dtype=np.float64),np.array(epa,dtype=np.float64))
for i in range(len(ycorr_total)):
    ycorr_total[-(i+1)]*=i+1

norm=ycorr_total[0]

f1=open("shift_total.dat",'w')
f1.write("#          t[ps]                   S%total                   S%raw                 \n")
f1.close()
f1=open("shift_total.dat",'ab')
np.savetxt(f1,np.c_[times,ycorr_total/norm,ycorr_total/n])
