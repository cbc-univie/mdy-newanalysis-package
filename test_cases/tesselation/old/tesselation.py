from __future__ import print_function
import MDAnalysis
import numpy as np
from MDAnalysis.newanalysis.gfunction import RDF
import os
import time

base='../../data/1mq_swm4_equilibrium/'
psf=base+'mqs0_swm4_1000.psf'

skip = 20
u=MDAnalysis.Universe(psf,base+"1mq_swm4.dcd")

sel_solute=u.selectAtoms("resname MQS0")
sel=u.selectAtoms("resname SWM4")

nsolute=1
nwat = sel.numberOfResidues()

u.trajectory[0]
shells = sel.calcTessellation(maxshell=3,core_sel=sel_solute,volumes=None,face_areas=None)
print(shells)
