import os
import MDAnalysis
from MDAnalysis.newanalysis.voro import calcTessellationParallel
import numpy as np
from copy import deepcopy
from sys import exit
import time
import sys
import h5py
from datetime import timedelta

base='../../data/1mq_swm4_nonequilibrium/'

u0=MDAnalysis.Universe(base+"mqs0_swm4_1000.psf")
u1=MDAnalysis.Universe(base+"mqs1_swm4_1000.psf",base+"1/1mq_swm4_1.dcd",base+"1/1mq_swm4_2.dcd",base+"1/1mq_swm4_3.dcd",base+"1/1mq_swm4_4.dcd")

dq=u1.MQS1.charges()-u0.MQS0.charges()

for i in range(len(u1.MQS1.atoms)):
    u1.MQS1.atoms[i].charge = dq[i]

sel=u1.selectAtoms('all')
mqs1=u1.selectAtoms('resname MQS1')
wat=u1.selectAtoms('resname SWM4')

charges=u1.residues.charges()
masses=u1.residues.masses()

isolute=0
nmol=1001
boxl=round(u1.coord.dimensions[0],4)


times = np.concatenate((np.arange(0.0,0.101,0.001),
                        np.arange(0.11,1.01,0.01),
                        np.arange(1.1,10.1,0.1),
                        np.arange(10.5,50.5,0.5)))

framesets = [(0,360,1)]

tlen = times.shape[0]

nshells = 20
ncpu = 4
f2c = sel.f2c()

corelist = [mqs1.universe.atoms.f2c(i) for i in mqs1.indices()]
corelist = np.array(list(set(corelist)),dtype=np.int32)
surroundlist = [wat.universe.atoms.f2c(i) for i in wat.indices()]
surroundlist = np.array(list(set(surroundlist)),dtype=np.int32)

nat = sel.numberOfAtoms()
nmol = sel.numberOfResidues()
nwat = wat.numberOfResidues()

list=np.arange(1,51)
folders = ["voro_{number}".format(number=i) for i in list]

ctr3=0

for f in folders:
    print("Entering folder "+f+" .. with dcds from"+base+str(ctr3+1))

    h5file = h5py.File(f+"/neq_delaunay.hdf5",'w')
    h5file.create_dataset("delaunay",(tlen,nwat),dtype='int8')
    dataset = h5file['delaunay']
    ctr2=1

    coor_ts = np.zeros((ncpu, sel.numberOfAtoms(), 3))
    ds_ts = np.zeros((ncpu, nwat), dtype=np.int8)
    fctr = 1
    wctr = 0

    index=str(list[ctr3])
    u10=MDAnalysis.Universe(base+"mqs1_swm4_1000.psf", base+"crd/1mq-swm4_cold_"+index+".crd")
    sel=u10.selectAtoms('all')
    tmp=sel.get_positions()
    for i in range(ncpu):
        coor_ts[i]=tmp
    calcTessellationParallel(coor_ts, f2c, corelist, surroundlist, ds_ts, boxl, nat, nmol, nshells)
    dataset[0]=ds_ts[0]
    
    u1=MDAnalysis.Universe(base+"mqs1_swm4_1000.psf",base+str(ctr3+1)+"/1mq_swm4_1.dcd",base+str(ctr3+1)+"/1mq_swm4_2.dcd",base+str(ctr3+1)+"/1mq_swm4_3.dcd",base+str(ctr3+1)+"/1mq_swm4_4.dcd")
    u1.trajectory[0]
    sel=u1.selectAtoms('all')

    
    start = time.time()
 #   print("")

    for fs in framesets:
        for ts in u1.trajectory[fs[0]:fs[1]:fs[2]]:
  #          print("\033[1AFrame %d of %d (%4.1f%%)" % (ctr2+1,tlen,(ctr2+1)/tlen*100),
   #               "      Elapsed time: %s" % str(timedelta(seconds=(time.time()-start)))[:7],
    #              "      Est. time left: %s" % (str(timedelta(seconds=(time.time()-start)/(ctr2+1) * (tlen-(ctr2+1))))[:7]))
        
            coor_ts[fctr-1] = sel.get_positions()

            if fctr % ncpu == 0:
                fctr = 1
                calcTessellationParallel(coor_ts, f2c, corelist, surroundlist, ds_ts, boxl, nat, nmol, nshells)
                for i in range(ncpu):
                    dataset[wctr+1] = ds_ts[i]
                    wctr += 1
            elif ctr2 == tlen:
                calcTessellationParallel(coor_ts, f2c, corelist, surroundlist, ds_ts, boxl, nat, nmol, nshells)
                for i in range(fctr):
                    dataset[wctr+1] = ds_ts[i]
                    wctr += 1
            else:
                fctr += 1
            
            ctr2+=1
    ctr3+=1
