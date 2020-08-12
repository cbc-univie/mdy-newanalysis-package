from __future__ import print_function
import MDAnalysis
import numpy as np
from newanalysis.gfunction import RDF
from newanalysis.helpers import dipByResidue
#from newanalysis.functions import centerOfMassByResidue, dipoleMomentByResidue
from newanalysis.functions import atomsPerResidue, residueFirstAtom
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

histo_min=0.0
histo_dx=0.05
histo_invdx=1.0/histo_dx
histo_max=20.0
boxl = np.float64(round(u.coord.dimensions[0],4))

sol_wat = RDF(["all"],histo_min,histo_max,histo_dx,nsolute,nwat,nsolute,nwat,boxl)

charge_solute=sel_solute.charges
charge_wat=sel.charges

apr_solute = atomsPerResidue(sel_solute)
rfa_solute = residueFirstAtom(sel_solute)
apr_wat = atomsPerResidue(sel)
rfa_wat = residueFirstAtom(sel)

    
mass_solute=sel_solute.masses
mass_wat=sel.masses

start=time.time()
print("")

for ts in u.trajectory[::skip]:
    print("\033[1AFrame %d of %d" % (ts.frame,u.trajectory.n_frames), "\tElapsed time: %.2f hours" % ((time.time()-start)/3600))
    
    # efficiently calculate center-of-mass coordinates and dipole moments
    coor_wat = np.ascontiguousarray(sel.positions,dtype='double')
    coor_solute = np.ascontiguousarray(sel_solute.positions,dtype='double')
    com_wat=sel.center_of_mass(compound='residues')
    com_solute=sel_solute.center_of_mass(compound='residues')
    dip_wat=dipByResidue(coor_wat,charge_wat,mass_wat,nwat,apr_wat,rfa_wat,com_wat)
    dip_solute=dipByResidue(coor_solute,charge_solute,mass_solute,nsolute,apr_solute,rfa_solute,com_solute)

    #Alternatively:
#    com_wat  = centerOfMassByResidue(sel,coor=coor_wat,masses=mass_wat,apr=apr_wat,rfa=rfa_wat)
#    com_solute  = centerOfMassByResidue(sel_solute,coor=coor_solute,masses=mass_solute,apr=apr_solute,rfa=rfa_solute)
#    dip_wat= dipoleMomentByResidue(sel,coor=coor_wat,charges=charge_wat,masses=mass_wat,com=com_wat,apr=apr_wat,rfa=rfa_wat)
#    dip_solute=dipoleMomentByResidue(sel_solute,coor=coor_solute,charges=charge_solute,masses=mass_solute,com=com_solute,apr=apr_solute,rfa=rfa_solute)

    sol_wat.calcFrame(com_solute,com_wat,dip_solute,dip_wat)

print("Passed time: ",time.time()-start)
    
sol_wat.write("rdf")
