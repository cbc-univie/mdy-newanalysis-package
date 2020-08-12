import os
import MDAnalysis
from newanalysis.helpers import calcEnergyAtomic
from newanalysis.unfold import minDistCenterBox
from newanalysis.correl import correlate
from newanalysis.functions import atomsPerResidue, residueFirstAtom
import numpy as np
from copy import deepcopy
from sys import exit
import time
from datetime import timedelta

base='../../data/1mq_swm4_equilibrium/'
psf_s0=base+'mqs0_swm4_1000.psf'
psf_s1=base+'mqs1_swm4_1000.psf'
#Check S0 PSF:
if np.array_equal(MDAnalysis.Universe(psf_s0).atoms.masses, MDAnalysis.Universe(psf_s0).atoms.masses.astype(bool)):
    print("Used wrong PSF format for S0 state (masses unreadable!)")
    sys.exit()
#Check S1 PSF:
if np.array_equal(MDAnalysis.Universe(psf_s1).atoms.masses, MDAnalysis.Universe(psf_s1).atoms.masses.astype(bool)):
    print("Used wrong PSF format for S1 state (masses unreadable!)")
    sys.exit()


skip = 1
u0=MDAnalysis.Universe(psf_s0)
u1=MDAnalysis.Universe(psf_s1,base+"1mq_swm4.dcd")

dq=u1.select_atoms("resname MQS1").charges-u0.select_atoms("resname MQS0").charges
mqs1=u1.select_atoms('resname MQS1')
sel=u1.select_atoms('all')
for i in range(mqs1.n_atoms):
    mqs1.atoms[i].charge=dq[i]

charges=sel.charges
masses=sel.masses

apr = atomsPerResidue(sel)
rfa= residueFirstAtom(sel)


boxl=round(u1.coord.dimensions[0],4)
dt=round(u1.trajectory.dt,4)


isolute=0
nmol=1001

n = int((u1.trajectory.n_frames)/skip)
if u1.trajectory.n_frames%skip != 0:
    n+=1
    
epa=np.zeros((n))

print("Calculating energy function...")
ctr=0
start=time.time()
print("")
for ts in u1.trajectory[::skip]:
    print("\033[1AFrame %d of %d (%4.1f%%)" % ((ts.frame+1),u1.trajectory.n_frames,(ts.frame+1)/u1.trajectory.n_frames*100),
          "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
          "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/(ts.frame+1) * (u1.trajectory.n_frames-(ts.frame+1))))[:7]))
    coor=np.ascontiguousarray(sel.positions,dtype='double')
    coms=sel.center_of_mass(compound='residues')
    com_mq = mqs1.center_of_mass()
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
