from __future__ import print_function
import MDAnalysis
import numpy as np
from newanalysis.functions import calcTessellation
import os
import time

base='../../data/1mq_swm4_equilibrium/'
psf=base+'mqs0_swm4_1000.psf'

#Check PSF:
if np.array_equal(MDAnalysis.Universe(psf).atoms.masses, MDAnalysis.Universe(psf).atoms.masses.astype(bool)):
    print("Used wrong PSF format (masses unreadable!)")
    sys.exit()


skip = 20
u=MDAnalysis.Universe(psf,base+"1mq_swm4.dcd")

sel_solute=u.select_atoms("resname MQS0")
sel=u.select_atoms("resname SWM4")

nsolute=1
nwat = sel.n_residues

u.trajectory[0]
shells = calcTessellation(sel,maxshell=3,core_sel=sel_solute,volumes=None,face_areas=None)
print(shells)
