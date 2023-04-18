# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; encoding: utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

import numpy as np
cimport numpy as np

from scipy.special import sph_harm

from cython.parallel cimport prange, parallel
cimport cython

from libc.math cimport fabs, sqrt, floor, pow, sin, cos
from libc.stdlib cimport malloc, free

from libc.stdlib cimport malloc, free

cdef extern from "BertholdHorn.h":
    void GetRotation(double *R_ptr, int n, double *APoints_ptr, double *BPoints_ptr, int debug)

cdef int sgn2bin(double val):
     return (<double> 0.0 < val)

@cython.boundscheck(False)
def SphereinCube(double A, int numpoints, double gfunc, double deltaA):
    cdef int i, j, k, sumr
    cdef double vol,x,y,z

    sumr=0
    for i in prange(numpoints,nogil=True):
        x=-A/2+(i+0.5)*deltaA
        for j in range(numpoints):
            y=-A/2+(j+0.5)*deltaA
            for k in range(numpoints):
                z=-A/2+(k+0.5)*deltaA

                if x*x+y*y+z*z <= gfunc**2:
                    sumr+=1
    return(sumr)
 
@cython.boundscheck(False)
def velcomByResidue(double [:,:] vels, double [:] masses, int nres, int [:] apr, int [:] rfa):
    """
    velcomByResidue(vels,masses,nres,atoms_per_residue)

    Calculates the center-of-mass velocities for a given selection.

    NOTE:
        Don't call this function directly, use the AtomGroup interface instead!
        E.g. for an atom selection 'sel', call 

        velcom = sel.velcomByResidue()    
    """

    cdef double [:,:] velcom = np.zeros((nres,3))
    cdef int i, j, k, actr
    cdef double tot_mass

    for i in prange(nres, nogil=True):
        tot_mass = 0.0
        for j in range(apr[i]):
            actr = rfa[i] + j
            for k in range(3):
                velcom[i,k] += vels[actr,k] * masses[actr]
            tot_mass += masses[actr]
        for k in range(3):
            velcom[i,k] /= tot_mass

    return np.asarray(velcom)

@cython.boundscheck(False)
def comByResidue(double [:,:] coor, double [:] masses, int nres, int [:] apr, int [:] rfa):
    """
    comByResidue(coor,masses,nres,atoms_per_residue)

    Calculates the center-of-mass coordinates for a given selection.

    NOTE:
        Don't call this function directly, use the AtomGroup interface instead!
        E.g. for an atom selection 'sel', call 

        com = sel.comByResidue()    
    """

    cdef double [:,:] com = np.zeros((nres,3))
    cdef int i, j, k, actr
    cdef double tot_mass

    for i in prange(nres, nogil=True):
        tot_mass = 0.0
        for j in range(apr[i]):
            actr = rfa[i] + j
            for k in range(3):
                com[i,k] += coor[actr,k] * masses[actr]
            tot_mass += masses[actr]
        for k in range(3):
            com[i,k] /= tot_mass

    return np.asarray(com)

@cython.boundscheck(False)
def dipByResidue(double [:,:] coor, double [:] charges, double [:] masses, int nres, int [:] apr, int [:] rfa, double [:,:] com):
    """
    dipByResidue(coor,charges,masses,nresidues,atoms_per_residue,com)

    Calculates the molecular dipole moments, each referenced to the respective center of mass, 
    for a given selection.

    NOTE:
        Don't call this function directly, use the AtomGroup interface instead!
        E.g. for an atom selection 'sel', call 

        dip = sel.dipByResidue()    
    """
    cdef double [:,:] dip = np.zeros((nres,3))
    cdef int i, j, k, actr

    for i in prange(nres, nogil=True):
        for j in range(apr[i]):
            actr = rfa[i] + j
            for k in range(3):
                dip[i,k] += (coor[actr,k] - com[i,k]) * charges[actr]
        
    return np.asarray(dip)

@cython.boundscheck(False)
def dipoleMomentNeutralSelection(double [:,:] coor, double [:] charges):
    
    cdef double [:] dip = np.zeros(3)
    cdef int i, natoms = coor.shape[0]

    for i in range(natoms):
        dip[0] += coor[i,0] * charges[i]
        dip[1] += coor[i,1] * charges[i]
        dip[2] += coor[i,2] * charges[i]
        
    return np.asarray(dip)


@cython.boundscheck(False)
def collectiveDipoleMomentWaterShells(double [:,:] coor, double [:] charges, int [:] ds, int nshells):
    
    cdef int i,j,shell
    cdef int natoms = coor.shape[0]
    cdef int nmol = len(ds)
    cdef double [:,:] dip = np.zeros((nshells,3))

    for shell in prange(nshells,nogil=True):
        for i in range(nmol):
            if ds[i] == shell+1:
                for j in range(3):
                    dip[shell,0] += coor[i*3+j,0] * charges[i*3+j]
                    dip[shell,1] += coor[i*3+j,1] * charges[i*3+j]
                    dip[shell,2] += coor[i*3+j,2] * charges[i*3+j]
    return np.asarray(dip)



@cython.boundscheck(False)
def atomicCurrent(double [:,:] vel, double [:] charges, double [:,:] result, int ctr):
    cdef int i, natoms=len(vel)
    
    for i in range(natoms):
        result[ctr,0] += vel[i,0] * charges[i]
        result[ctr,1] += vel[i,1] * charges[i]
        result[ctr,2] += vel[i,2] * charges[i]

@cython.boundscheck(False)
def centerOrientBox(double [:,:] com, double [:,:] coor, double boxl, int isolute,
                    int [:] apr, int [:] rfa, double [:,:] coorA, double [:,:] coorB):
    """
    centerOrientBox(com, xyz, boxlength, isolute, atoms_per_residue, residue_first_atom, coorA, coorB)

    Centers the box to the center of mass of a given molecule and rotates the box so that two sets of coordinates
    are as identical as possible.

    Args:
        com                .. centers of mass of all molecules in the box
        xyz                .. coordinates of all atoms
        boxlength          .. edge length of the cubic box
        isolute            .. index of the solute / center molecule (count starts with 0)
        atoms_per_residue  .. array showing how many atoms each residue contains
        residue_first_atom .. index of the first atom of each residue
        coorA              .. template coordinates
        coorB              .. corresponding coordinates of the current frame
    """
    cdef double[3] shift
    cdef double[3] comslt
    cdef int i, j, k, nmol = com.shape[0]
    cdef double [:,:] R = np.zeros((3,3))
    cdef double tmpx, tmpy, tmpz

    for j in range(3):
        comslt[j] = com[isolute,j]
    
    for i in range(nmol):
        for j in range(3):
            com[i,j] -= comslt[j]
            shift[j] = boxl * floor(com[i,j] / boxl + 0.5)
            com[i,j] -= shift[j]

        for k in range(rfa[i], rfa[i] + apr[i]):
            for j in range(3):
                coor[k,j] -= comslt[j] - shift[j]

    GetRotation(&R[0,0], coorA.shape[0], &coorA[0,0], &coorB[0,0], 0)

    for i in prange(coor.shape[0], nogil=True):
        tmpx = R[0,0] * coor[i,0] + R[0,1] * coor[i,1] + R[0,2] * coor[i,2]
        tmpy = R[1,0] * coor[i,0] + R[1,1] * coor[i,1] + R[1,2] * coor[i,2]
        tmpz = R[2,0] * coor[i,0] + R[2,1] * coor[i,1] + R[2,2] * coor[i,2]
        coor[i,0] = tmpx
        coor[i,1] = tmpy
        coor[i,2] = tmpz

@cython.boundscheck(False)
def centerToPointOrientBox(double [:,:] com, double [:,:] coor, double [:] center, double boxl,
                           int [:] apr, int [:] rfa, double [:,:] coorA, double [:,:] coorB):
    """
    centerOrientToPointBox(com, xyz, center, boxlength, atoms_per_residue, residue_first_atom, coorA, coorB)

    Centers the box to the center of mass of a given molecule and rotates the box so that two sets of coordinates
    are as identical as possible.

    Args:
        com                .. centers of mass of all molecules in the box
        xyz                .. coordinates of all atoms
        boxlength          .. edge length of the cubic box
        atoms_per_residue  .. array showing how many atoms each residue contains
        residue_first_atom .. index of the first atom of each residue
        coorA              .. template coordinates
        coorB              .. corresponding coordinates of the current frame
    """
    cdef double[3] shift
    cdef int i, j, k, nmol = com.shape[0]
    cdef double [:,:] R = np.zeros((3,3))
    cdef double tmpx, tmpy, tmpz

    for i in range(nmol):
        for j in range(3):
            com[i,j] -= center[j]
            shift[j] = boxl * floor(com[i,j] / boxl + 0.5)
            com[i,j] -= shift[j]

        for k in range(rfa[i], rfa[i] + apr[i]):
            for j in range(3):
                coor[k,j] -= center[j] - shift[j]

    GetRotation(&R[0,0], coorA.shape[0], &coorA[0,0], &coorB[0,0], 0)

    for i in prange(coor.shape[0], nogil=True):
        tmpx = R[0,0] * coor[i,0] + R[0,1] * coor[i,1] + R[0,2] * coor[i,2]
        tmpy = R[1,0] * coor[i,0] + R[1,1] * coor[i,1] + R[1,2] * coor[i,2]
        tmpz = R[2,0] * coor[i,0] + R[2,1] * coor[i,1] + R[2,2] * coor[i,2]
        coor[i,0] = tmpx
        coor[i,1] = tmpy
        coor[i,2] = tmpz
        
@cython.boundscheck(False)
def calcEnergyAA(double [:,:] coor, double [:] charges, int [:] apr, int [:] rfa, int isolute, int nmol):
    """
    energy = calcEnergyAA(xyz, charges, atoms_per_residue, residue_first_atom, resnum_solute, nmolecules)

    Calculates the solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i, j, k, idx, idx2
    cdef double [:] epa = np.zeros(apr[isolute])
    cdef double dx, dy, dz, r

    energy = 0.0

    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        # loop over solvent molecules
        for j in range(nmol):
            if j == isolute:
                continue
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                epa[i] += charges[idx] * charges[idx2] / r

    for i in range(apr[isolute]):
        energy += epa[i]

    return energy * 1390.02

@cython.boundscheck(False)
def calcEnergyMuA(double [:,:] coor, double [:] charges, int [:] apr, int [:] rfa, int isolute, int nmol, double [:] mu_solute):
    """
    energy = calcEnergyMuA(xyz, charges, atoms_per_residue, residue_first_atom, resnum_solute, nmolecules,mu_solute)

    Calculates the solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int j, k, idx2
    cdef double dx, dy, dz, r, energy

    energy = 0.0

    # loop over solvent molecules
    for j in range(nmol):
        if j == isolute:
            continue
        for k in range(apr[j]):
            idx2 = rfa[j] + k
            dx = coor[idx2,0]
            dy = coor[idx2,1]
            dz = coor[idx2,2]
            r  = sqrt(dx*dx + dy*dy + dz*dz)

            energy+=charges[idx2]*(mu_solute[0]*dx+mu_solute[1]*dy+mu_solute[2]*dz)/(r*r*r)
    
    return energy * 1390.02

@cython.boundscheck(False)
def calcEnergyAApermind(double [:,:] coor, double[:,:] coms, double [:] charges, int [:] apr, int [:] rfa, int isolute, int first, int last, double[:] drude):
    """
    energy = calcEnergyAA(xyz, coms, charges, atoms_per_residue, residue_first_atom, resnum_solute, resnum_from, resnum_to, drude_list)

    Calculates the permanent and induced part of the solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i, j, k, idx, idx2, histo_error
    cdef double [:] epa_ind = np.zeros(apr[isolute])
    cdef double [:] epa_perm = np.zeros(apr[isolute])
    cdef double dx, dy, dz, r, qq, energy_perm, energy_ind, qq_diff

    energy_perm = 0.0
    energy_ind = 0.0

    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        # loop over solvent molecules
        for j in range(first,last+1):
            if j == isolute:
                continue
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                qq = charges[idx] * charges[idx2] / r

                if drude[k]==1:
                    epa_ind[i]+=qq
                else:
                    epa_perm[i]+=qq
                    if k<apr[j]-1 and drude[k+1]==1:
                        qq_diff=charges[idx]*charges[idx2+1] / r
                        epa_ind[i] -= qq_diff
                        epa_perm[i]+= qq_diff
                        
    for i in range(apr[isolute]):
        energy_perm += epa_perm[i]
        energy_ind  += epa_ind[i]

    return energy_perm * 1390.02 , energy_ind * 1390.02

@cython.boundscheck(False)
def calcEnergyAAhisto(double [:,:] coor,  double[:,:] coms, double [:] charges, int [:] apr, int [:] rfa, int isolute, int nmol, double histo_min, double histo_max, int histo_bins, int[:] trehalose, int[:] oxyquinol):
    """
    energy = calcEnergyAAhisto(xyz, coms, charges, atoms_per_residue, residue_first_atom, resnum_solute, nmolecules, min_histo, max_histo, bins_hist,array indices, array indices 2)

    Calculates the solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i, j, k, idx, idx2, histo_error
    cdef double [:] epa = np.zeros(apr[isolute])
    cdef double [:,:] histo=np.zeros((histo_bins,histo_bins))
    cdef double [:,:] count=np.zeros((histo_bins,histo_bins))
    cdef double [:] store=np.zeros(nmol)
    cdef double dx, dy, dz, r, rtre, tmp, tmp_ener, histo_width
    cdef double [:] squared_r_tre = np.zeros(trehalose.shape[0])
    cdef double [:] squared_r_oxy = np.zeros(oxyquinol.shape[0])
    
    energy = 0.0
    histo_error=0
    histo_width=(histo_max-histo_min)/histo_bins
    
    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        # loop over solvent molecules

        for j in range(nmol):
            if j == isolute:
                continue
            tmp_ener=0.0
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                tmp=charges[idx] * charges[idx2] / r
                tmp_ener+=tmp
                epa[i] += tmp
                
            store[j]+=tmp_ener

            
    for j in range(nmol):
        if j==isolute:
            continue
        for k in range(trehalose.shape[0]):
            squared_r_tre[k]=(coor[trehalose[k],0]-coms[j,0])**2+(coor[trehalose[k],1]-coms[j,1])**2+(coor[trehalose[k],2]-coms[j,2])**2
            if k>=1:
                if squared_r_tre[k]<rtre**2:
                    rtre=sqrt(squared_r_tre[k])
            else:
                rtre=sqrt(squared_r_tre[k])
        for k in range(oxyquinol.shape[0]):
            squared_r_oxy[k]=(coor[oxyquinol[k],0]-coms[j,0])**2+(coor[oxyquinol[k],1]-coms[j,1])**2+(coor[oxyquinol[k],2]-coms[j,2])**2
            if k>=1:
                if squared_r_oxy[k]<roxy**2:
                    roxy=sqrt(squared_r_oxy[k])
            else:
                roxy=sqrt(squared_r_oxy[k])
                
        if rtre < histo_min or roxy < histo_min:
            histo_error=1
        elif rtre >= histo_max or roxy>=histo_max:
            histo_error=2
        else:
            histo[int((rtre-histo_min)/histo_width),int((roxy-histo_min)/histo_width)]+=store[j]
            count[int((rtre-histo_min)/histo_width),int((roxy-histo_min)/histo_width)]+=1

    for i in range(apr[isolute]):
        energy += epa[i]

    return energy * 1390.02, np.asarray(histo)[:,:] * 1390.02 , histo_error, np.asarray(count)[:,:]

@cython.boundscheck(False)
def calcEnergyAAhisto1(double [:,:] coor,  double[:,:] coms, double [:] charges, int [:] apr, int [:] rfa, int isolute, int nmol, double histo_min, double histo_max, int histo_bins, int[:] trehalose):
    """
    energy = calcEnergyAAhisto(xyz, coms, charges, atoms_per_residue, residue_first_atom, resnum_solute, nmolecules, min_histo, max_histo, bins_hist,array indices)

    Calculates the solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i, j, k, idx, idx2, histo_error
    cdef double [:] epa = np.zeros(apr[isolute])
    cdef double [:] histo=np.zeros((histo_bins))
    cdef double [:] count=np.zeros((histo_bins))
    cdef double [:] store=np.zeros(nmol)
    cdef double dx, dy, dz, r, rtre, tmp, tmp_ener, histo_width
    cdef double [:] squared_r_tre = np.zeros(trehalose.shape[0])
    
    energy = 0.0
    histo_error=0
    histo_width=(histo_max-histo_min)/histo_bins
    
    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        # loop over solvent molecules

        for j in range(nmol):
            if j == isolute:
                continue
            tmp_ener=0.0
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                tmp=charges[idx] * charges[idx2] / r
                tmp_ener+=tmp
                epa[i] += tmp
                
            store[j]+=tmp_ener

            
    for j in range(nmol):
        if j==isolute:
            continue
        for k in range(trehalose.shape[0]):
            squared_r_tre[k]=(coor[trehalose[k],0]-coms[j,0])**2+(coor[trehalose[k],1]-coms[j,1])**2+(coor[trehalose[k],2]-coms[j,2])**2
            if k>=1:
                if squared_r_tre[k]<rtre**2:
                    rtre=sqrt(squared_r_tre[k])
            else:
                rtre=sqrt(squared_r_tre[k])
                
        if rtre < histo_min:
            histo_error=1
        elif rtre >= histo_max:
            histo_error=2
        else:
            histo[int((rtre-histo_min)/histo_width)]+=store[j]
            count[int((rtre-histo_min)/histo_width)]+=1

    for i in range(apr[isolute]):
        energy += epa[i]

    return energy * 1390.02, np.asarray(histo)[:] * 1390.02 , histo_error, np.asarray(count)[:]

@cython.boundscheck(False)
def calcEnergyAApermindhisto(double [:,:] coor, double[:,:] coms, double [:] charges, int [:] apr, int [:] rfa, int isolute, int first, int last, double[:] drude,double histo_min, double histo_max, int histo_bins,):
    """
    energy = calcEnergyAA(xyz, coms, charges, atoms_per_residue, residue_first_atom, resnum_solute, resnum_from, resnum_to, drude_list, min_histo, max_histo, bins_histo)

    Calculates the permanent and induced part of the solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i, j, k, idx, idx2, histo_error
    cdef double [:] epa_ind = np.zeros(apr[isolute])
    cdef double [:] epa_perm = np.zeros(apr[isolute])
    cdef double [:] histo_perm=np.zeros(histo_bins), histo_ind=np.zeros(histo_bins)
    cdef double dx, dy, dz, r, qq, energy_perm, energy_ind, histo_width, tmp_perm, tmp_ind, qq_diff

    energy_perm = 0.0
    energy_ind = 0.0
    histo_error=0
    histo_width=(histo_max-histo_min)/histo_bins

    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        # loop over solvent molecules
        for j in range(first,last+1):
            if j == isolute:
                continue
            tmp_perm=0.0
            tmp_ind=0.0
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                qq = charges[idx] * charges[idx2] / r

                if drude[k]==1:
                    epa_ind[i]+=qq
                    tmp_ind   +=qq
                else:
                    epa_perm[i]+=qq
                    tmp_perm   +=qq
                    if k<apr[j]-1 and drude[k+1]==1:
                        qq_diff=charges[idx]*charges[idx2+1] / r
                        epa_ind[i] -= qq_diff
                        tmp_ind    = tmp_ind-qq_diff
                        epa_perm[i]+= qq_diff
                        tmp_perm   = tmp_perm+qq_diff
                        
            r=sqrt(coms[j,0]*coms[j,0]+coms[j,1]*coms[j,1]+coms[j,2]*coms[j,2])
            if r < histo_min:
                histo_error=1
            elif r >= histo_max:
                histo_error=2
            else:
                histo_perm[int((r-histo_min)/histo_width)]+=tmp_perm
                histo_ind[int((r-histo_min)/histo_width)] +=tmp_ind
                    
    for i in range(apr[isolute]):
        energy_perm += epa_perm[i]
        energy_ind  += epa_ind[i]

    return energy_perm * 1390.02 , energy_ind * 1390.02, np.asarray(histo_perm)[:] * 1390.02, np.asarray(histo_ind)[:] * 1390.02 , histo_error

@cython.boundscheck(False)
def calcEnergyAtomic(double [:,:] coor, double [:] charges, int [:] apr, int [:] rfa, int isolute, int nmol, int exclude_begin=-1, int exclude_end=-1):
    """
    energy = calcEnergyAtomic(xyz, charges, atoms_per_residue, residue_first_atom, resnum_solute, nmolecules)

    Calculates the atom-resolved solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i, j, k, idx, idx2
    cdef double [:] epa = np.zeros(apr[isolute])
    cdef double dx, dy, dz, r

    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        # loop over solvent molecules
        for j in range(nmol):
            if j == isolute or (j >= exclude_begin and j < exclude_end):
                continue    
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                epa[i] += charges[idx] * charges[idx2] / r

    return np.asarray(epa)[:] * 1390.02

@cython.boundscheck(False)
def calcEnergyDouble(double [:,:] coor, double [:] charges1,  int [:] apr, int [:] rfa, int isolute, int nmol, int start, int end):
    """
    energy = calcEnergyDouble(xyz, charges_1, charges_2, atoms_per_residue, residue_first_atom, resnum_solute, nmolecules,start_res,end_res)

    Calculates the atom-resolved solvation energy of a solute in any solvent for a specific coordinate set 
    
    """
    cdef int i, j, k, idx, idx2
    cdef double [:] epa1 = np.zeros(apr[isolute])
    cdef double dx, dy, dz, r

    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        if charges1[idx]==0:
            continue
        # loop over solvent molecules
        for j in range(nmol):
            if j == isolute or j < start or j > end:
                continue    
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                epa1[i] += charges1[idx] * charges1[idx2] / r

    return np.asarray(epa1)[:] * 1390.02 

@cython.boundscheck(False)
def calcDipDipEnergyAtomic(double [:,:] coms, double [:,:] dipol,  int isolute, int nmol, int exclude_begin=-1, int exclude_end=-1):
    """
    energy = calcDipDipEnergyAtomic(coms, dipol, resnum_solute, nmolecules)

    Calculates the atom-resolved solvation dipol-dipol energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i
    cdef double edip, r

    # loop over solvent molecules
    for i in range(nmol):
        if i == isolute or (i >= exclude_begin and i < exclude_end):
            continue
        r=sqrt(coms[i,0]*coms[i,0]+coms[i,1]*coms[i,1]+coms[i,2]*coms[i,2])
        edip+=(dipol[0,0]*dipol[i,0]+dipol[0,1]*dipol[i,1]+dipol[0,2]*dipol[i,2])/(r*r*r)-3/(r*r*r*r*r)*(dipol[0,0]*coms[i,0]+dipol[0,1]*coms[i,1]+dipol[0,2]*coms[i,2])*(dipol[i,0]*coms[i,0]+dipol[i,1]*coms[i,1]+dipol[i,2]*coms[i,2])

    return edip * 1390.02


@cython.boundscheck(False)
def calcEnergyAtomicVoro(double [:,:] coor, double [:] charges, int [:] apr, int [:] rfa, int isolute, int nmol, char [:] ds, int maxshell):
    """
    energy = calcEnergyAtomic(xyz, charges, atoms_per_residue, residue_first_atom, resnum_solute, nmolecules, delaunay_shell)

    Calculates the atom+shell-resolved solvation energy of a solute in any solvent for a specific coordinate set.
    
    """
    cdef int i, j, k, idx, idx2
    cdef double [:,:] epa = np.zeros((apr[isolute],maxshell+1))
    cdef double dx, dy, dz, r

    # loop over solute atoms
#    for i in prange(apr[isolute], nogil=True):
    for i in range(apr[isolute]):
        idx = rfa[isolute] + i
        # loop over solvent molecules
        for j in range(nmol):
            if j == isolute:
                continue
            for k in range(apr[j]):
                idx2 = rfa[j] + k
                dx = coor[idx,0] - coor[idx2,0]
                dy = coor[idx,1] - coor[idx2,1]
                dz = coor[idx,2] - coor[idx2,2]
                r  = sqrt(dx*dx + dy*dy + dz*dz)
                if ds[j-1] > maxshell:
                    epa[i,maxshell] += charges[idx] * charges[idx2] / r
                else:
                    epa[i,ds[j-1]-1] += charges[idx] * charges[idx2] / r

    return np.asarray(epa)[:,:] * 1390.02

@cython.boundscheck(False)
def calcEnergyAASep(np.ndarray[np.float64_t,ndim=2,mode="c"] xyz, 
                 np.ndarray[np.float64_t,ndim=1] charges,
                 np.ndarray[np.int32_t,ndim=1,mode="c"] atoms_per_residue,
                 np.ndarray[np.int32_t,ndim=1,mode="c"] residue_first_atom,
                 np.ndarray[np.int32_t,ndim=2,mode="c"] ds,
                 isolute, nmolecules, cat_first, cat_last, an_first, an_last,
                 np.ndarray[np.int32_t, ndim=1] octants=None):
    """
    Calculates the solvation energy of a solute in an ionic liquid for a given frame.

    The data is returned in three decompositions:
        -) Voronoi shell-resolved

        -) x/y/z axis-resolved

        -) octant-resolved for the first shell

    :Example:

    energy, energy_xyz, energy_octant = calcEnergyAA(xyz, charges, atoms_per_residue, residue_first_atom, \
                                                     ds, octants, isolute, nmolecules, \
                                                     cat_first, cat_last, an_first, an_last, octants=None)
    """

    cdef double *cxyz    = <double *> xyz.data
    cdef double *q       = <double *> charges.data
    cdef int* apr        = <int *> atoms_per_residue.data
    cdef int* rfa        = <int *> residue_first_atom.data
    cdef int* coct = NULL
    if octants!=None:
        coct = <int *> octants.data
    cdef int i, j, k, idx, idx2, idx3, nmol=<int>nmolecules, islt=<int>isolute
    cdef np.ndarray[np.float64_t,ndim=2] energy_per_atom = np.zeros((apr[islt],8),dtype=np.float64)
    cdef double* epa = <double *> energy_per_atom.data
    cdef np.ndarray[np.float64_t,ndim=2] energy_xyz_per_atom = np.zeros((apr[islt],3),dtype=np.float64)
    cdef double* epa_xyz = <double *> energy_xyz_per_atom.data
    cdef np.ndarray[np.float64_t,ndim=2] energy_octant_per_atom = np.zeros((apr[islt],8),dtype=np.float64)
    cdef double* epa_oct = <double *> energy_octant_per_atom.data
    cdef double dx, dy, dz, r, qqr
    cdef int x, y, z, idx4, c1=cat_first, c2=cat_last, a1=an_first, a2=an_last
    cdef int *cds = <int *> ds.data
    cdef np.ndarray[np.float64_t,ndim=1] energy = np.zeros(8,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] energy_xyz = np.zeros(3,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=1] energy_octant = np.zeros(8,dtype=np.float64)

#    for i in prange(apr[islt],nogil=True):
    for i in range(apr[islt]):
        idx=rfa[islt]+i
        # cations
        for j in range(c1,c2+1):
            idx3=cds[islt*nmol+j]-1
            for k in range(apr[j]):
                idx2=rfa[j]+k
                dx=cxyz[idx*3]-cxyz[idx2*3]
                dy=cxyz[idx*3+1]-cxyz[idx2*3+1]
                dz=cxyz[idx*3+2]-cxyz[idx2*3+2]
                r=sqrt(dx*dx+dy*dy+dz*dz)
                qqr=q[idx]*q[idx2]/r
                epa[i*8+idx3]+=qqr
                epa_xyz[i*3]+=qqr*dx*dx/(r*r)
                epa_xyz[i*3+1]+=qqr*dy*dy/(r*r)
                epa_xyz[i*3+2]+=qqr*dz*dz/(r*r)
                if idx3 == 0 and coct != NULL:
                    epa_oct[i*8+coct[j]]+=qqr
                
        # anions
        for j in range(a1,a2+1):
            idx3=cds[islt*nmol+j]-1
            for k in range(apr[j]):
                idx2=rfa[j]+k
                dx=cxyz[idx*3]-cxyz[idx2*3]
                dy=cxyz[idx*3+1]-cxyz[idx2*3+1]
                dz=cxyz[idx*3+2]-cxyz[idx2*3+2]
                r=sqrt(dx*dx+dy*dy+dz*dz)
                qqr=q[idx]*q[idx2]/r
                epa[i*8+4+idx3]+=qqr
                epa_xyz[i*3]+=qqr*dx*dx/(r*r)
                epa_xyz[i*3+1]+=qqr*dy*dy/(r*r)
                epa_xyz[i*3+2]+=qqr*dz*dz/(r*r)
                if idx3 == 0 and coct != NULL:
                    epa_oct[i*8+coct[j]]+=qqr

    for i in range(apr[islt]):
        for j in range(8):
            energy[j]+=epa[i*8+j]
            energy_octant[j]+=epa_oct[i*8+j]
        for j in range(3):
            energy_xyz[j]+=epa_xyz[i*3+j]

    return energy*1390.02, energy_xyz*1390.02, energy_octant*1390.02

@cython.boundscheck(False)
def findDS(char [:,:,:] ds, double [:,:] cn, int n1, int n2, int shell, int t):

    cdef int i, j

    for i in prange(n1, nogil=True):
        for j in range(n2):
            if ds[t, i, j] == shell:
                cn[t, i] += 1.0


@cython.boundscheck(False)
def dipTenCorrel(np.ndarray[np.float64_t,ndim=2,mode='c'] coo_center,
                 np.ndarray[np.float64_t,ndim=2,mode='c'] coo_H,
                 char [:,:] H_indices,
                 np.ndarray[np.float64_t,ndim=3,mode='c'] dipT0,
                 np.ndarray[np.float64_t,ndim=3,mode='c'] corrsubmean,
                 np.int32_t timectr):

    cdef int i, center, H_shell, ctr
    cdef int len_center = len(coo_center)
    cdef int len_H = len(coo_H)
    cdef double rvec0, rvec1, rvec2
    cdef double dipTt0, dipTt1, dipTt2, dipTt3, dipTt4, dipTt5
    cdef double r2, f1, f2, f2_0, f2_1, f2_2
    cdef double *centers = <double *> coo_center.data
    cdef double *hs      = <double *> coo_H.data

    for center in prange(len_center,nogil=True):
        ctr = 0
        for i in range(len_H):
            if H_indices[center,i] > 0:
                H_shell = H_indices[center,i]
                rvec0 = hs[3*i] -   centers[3*center]
                rvec1 = hs[3*i+1] - centers[3*center+1]
                rvec2 = hs[3*i+2] - centers[3*center+2]
                r2 = rvec0*rvec0+rvec1*rvec1+rvec2*rvec2
                f1 = pow(r2,-1.5)
                f2 = 3.0 * f1 / r2
                f2_0 = f2 * rvec0
                f2_1 = f2 * rvec1
                f2_2 = f2 * rvec2
                
                dipTt0 = (f2_0 * rvec0 - f1)
                dipTt1 = (f2_1 * rvec1 - f1) 
                dipTt2 = (f2_2 * rvec2 - f1)
                dipTt3 = (f2_0 * rvec1)
                dipTt4 = (f2_0 * rvec2)
                dipTt5 = (f2_1 * rvec2)
                
                corrsubmean[center,H_shell,timectr] = corrsubmean[center,H_shell,timectr] + dipT0[center,ctr,0] * dipTt0 + dipT0[center,ctr,1] * dipTt1 + dipT0[center,ctr,2] * dipTt2 + dipT0[center,ctr,3] * dipTt3 + dipT0[center,ctr,4] * dipTt4 + dipT0[center,ctr,5] * dipTt5

                ctr = ctr + 1

@cython.boundscheck(False)
def dipTenInit(np.ndarray[np.float64_t,ndim=2,mode='c'] coo_center,
               np.ndarray[np.float64_t,ndim=2,mode='c'] coo_H,
               char [:,:] H_indices,
               np.ndarray[np.float64_t,ndim=3,mode='c'] dipT0,
               np.float64_t nshells2):
    cdef int i, center, ctr
    cdef int len_center = len(coo_center)
    cdef int len_H = len(coo_H)
    cdef double rvec0, rvec1, rvec2
    cdef double dipTt0, dipTt1, dipTt2, dipTt3, dipTt4, dipTt5
    cdef double r2, f1, f2, f2_0, f2_1, f2_2
    cdef double *centers = <double *> coo_center.data
    cdef double *hs      = <double *> coo_H.data
    
    for center in prange(len_center,nogil=True):
        ctr = 0
        for i in range(len_H):
            rvec0 = hs[3*i]   - centers[3*center]
            rvec1 = hs[3*i+1] - centers[3*center+1]
            rvec2 = hs[3*i+2] - centers[3*center+2]
            r2 = rvec0*rvec0+rvec1*rvec1+rvec2*rvec2
            
            if r2 <= nshells2:
                H_indices[center,i] = int(floor(r2**0.5))
                f1 = pow(r2,-1.5)
                f2 = 3.0 * f1 / r2
                f2_0 = f2 * rvec0
                f2_1 = f2 * rvec1
                f2_2 = f2 * rvec2
                
                dipT0[center,ctr,0] = (f2_0 * rvec0 - f1) /6 # /6 for averaging the elements in the correlation
                dipT0[center,ctr,1] = (f2_1 * rvec1 - f1) /6
                dipT0[center,ctr,2] = (f2_2 * rvec2 - f1) /6
                dipT0[center,ctr,3] = (f2_0 * rvec1) /3 # *2 because off-diag element are taken twice, when summing over all elements in the correlation
                dipT0[center,ctr,4] = (f2_0 * rvec2) /3
                dipT0[center,ctr,5] = (f2_1 * rvec2) /3
                
                ctr = ctr + 1


@cython.boundscheck(False)
def dipTen(np.ndarray[np.float64_t,ndim=1] rv,
           np.float64_t r2):
    """
    dipTen(coo1,coo2) -> returns dip tensor
    Calculates the dipole T-tensor (see Daniel's NMR code).
    """
    cdef double* c_rv = <double *> rv.data
    cdef np.ndarray[np.float64_t,ndim=1] dipt = np.empty(6,dtype=np.float64)
   
    cdef double f1 = pow(r2,-1.5)
    cdef double f2 = 3.0 * f1 / r2
    cdef double f2_0 = f2 * c_rv[0]
    cdef double f2_1 = f2 * c_rv[1]
    cdef double f2_2 = f2 * c_rv[2]
        
    dipt[0] = f2_0 * c_rv[0] - f1
    dipt[1] = f2_1 * c_rv[1] - f1
    dipt[2] = f2_2 * c_rv[2] - f1
    dipt[3] = f2_0 * c_rv[1]
    dipt[4] = f2_0 * c_rv[2]
    dipt[5] = f2_1 * c_rv[2]
    
    return dipt

@cython.boundscheck(False)
def NQRself(np.ndarray[np.float64_t,ndim=2] py_xyz):
    
    cdef int nwat = len(py_xyz)/3 # number of water molecules
    cdef double *xyz = <double *> py_xyz.data
    cdef double H1x,H1y,H1z,H2x,H2y,H2z,n2,n3
    cdef double Exx = -2.3067236 # -2.0486*1.126
    cdef double Eyy =  2.0364836 # 1.8086*1.126
    cdef double Ezz =  0.27024   # 0.2400*1.126
    cdef int i, k, ind
    cdef np.ndarray[np.float64_t,ndim=1] py_dipt = np.zeros(nwat*6,dtype=np.float64)
    cdef double *dipt = <double *> py_dipt.data

    with nogil, parallel():
        B = <double *> malloc(sizeof(double) * 9)
        T = <double *> malloc(sizeof(double) * 9)
        for i in prange(0,nwat*9,9): # loop over all coordinates of atoms of water molecules
            for k in range(9):
                B[k] = 0.0
                T[k] = 0.0
            
            H1x = xyz[i+3]-xyz[i]
            H1y = xyz[i+4]-xyz[i+1]
            H1z = xyz[i+5]-xyz[i+2]
            H2x = xyz[i+6]-xyz[i]
            H2y = xyz[i+7]-xyz[i+1]
            H2z = xyz[i+8]-xyz[i+2]

            B[3] = (H1x-H2x)
            B[4] = (H1y-H2y)
            B[5] = (H1z-H2z)
            B[6] = (H1x+H2x)
            B[7] = (H1y+H2y)
            B[8] = (H1z+H2z)
            
            n2 = (B[3]*B[3]+B[4]*B[4]+B[5]*B[5])**.5 # normalization Bvec_2
            n3 = (B[6]*B[6]+B[7]*B[7]+B[8]*B[8])**.5 # normalization Bvec_3

            B[3] /= n2
            B[4] /= n2
            B[5] /= n2
            B[6] /= n3
            B[7] /= n3
            B[8] /= n3

            B[0] = B[4]*B[8]-B[5]*B[7] # (Bvec_1,Bvec_2,Bvec_3) are orthonormal -> vector product
            B[1] = B[5]*B[6]-B[3]*B[8]
            B[2] = B[3]*B[7]-B[4]*B[6]
            
            # B dot T
            T[0] = B[0]*Exx
            T[1] = B[1]*Exx
            T[2] = B[2]*Exx
            T[3] = B[3]*Eyy
            T[4] = B[4]*Eyy
            T[5] = B[5]*Eyy
            T[6] = B[6]*Ezz
            T[7] = B[7]*Ezz
            T[8] = B[8]*Ezz

            ind = (i/9)*6
            
            # (B dot T) dot B.T
            dipt[ind]   = T[0]*B[0]+T[3]*B[3]+T[6]*B[6] # 11 .... here B.T is taken directly from B
            dipt[ind+1] = T[1]*B[1]+T[4]*B[4]+T[7]*B[7] # 22
            dipt[ind+2] = T[2]*B[2]+T[5]*B[5]+T[8]*B[8] # 33
            dipt[ind+3] = T[1]*B[0]+T[4]*B[3]+T[7]*B[6] # 21
            dipt[ind+4] = T[2]*B[0]+T[5]*B[3]+T[8]*B[6] # 31
            dipt[ind+5] = T[2]*B[1]+T[5]*B[4]+T[8]*B[7] # 32 .... matrix is symmetric
            
        free(B)
        free(T)
        
    return py_dipt

@cython.boundscheck(False)
def waterRotationMatrix(double [:,:] coor):
    cdef int nwat = len(coor)/3 # number of water molecules
    cdef double H1x,H1y,H1z,H2x,H2y,H2z,n2,n3
    cdef int i, k, ind
    cdef double [:,:,:] B_out = np.zeros((nwat,3,3)) # rotation matrix to return from parallel code
    
    with nogil, parallel():
        B = <double *> malloc(sizeof(double) * 9)
        for i in prange(0,nwat*3,3): # loop over all coordinates of atoms of water molecules
            for k in range(9):
                B[k] = 0.0

            H1x = coor[i+1,0]-coor[i,0] # rel. coordinates of H1 with respect to O
            H1y = coor[i+1,1]-coor[i,1]
            H1z = coor[i+1,2]-coor[i,2]
            
            H2x = coor[i+2,0]-coor[i,0] # rel. coordinates of H2 with respect to O
            H2y = coor[i+2,1]-coor[i,1]
            H2z = coor[i+2,2]-coor[i,2]

            B[3] = (H1x-H2x)
            B[4] = (H1y-H2y)
            B[5] = (H1z-H2z)
            B[6] = (H1x+H2x)
            B[7] = (H1y+H2y)
            B[8] = (H1z+H2z)
            
            n2 = (B[3]*B[3]+B[4]*B[4]+B[5]*B[5])**.5 # normalization Bvec_2
            n3 = (B[6]*B[6]+B[7]*B[7]+B[8]*B[8])**.5 # normalization Bvec_3

            B[3] /= n2
            B[4] /= n2
            B[5] /= n2
            B[6] /= n3
            B[7] /= n3
            B[8] /= n3

            B[0] = B[4]*B[8]-B[5]*B[7] # (Bvec_1,Bvec_2,Bvec_3) are orthonormal -> vector product
            B[1] = B[5]*B[6]-B[3]*B[8]
            B[2] = B[3]*B[7]-B[4]*B[6]

            ind =i/3
            
            B_out[ind,0,0] = B[0]
            B_out[ind,0,1] = B[1]
            B_out[ind,0,2] = B[2]
            B_out[ind,1,0] = B[3]
            B_out[ind,1,1] = B[4]
            B_out[ind,1,2] = B[5]
            B_out[ind,2,0] = B[6]
            B_out[ind,2,1] = B[7]
            B_out[ind,2,2] = B[8]
            
        free(B)
        
    return B_out
    
@cython.boundscheck(False)
def NQRselfAndB(np.ndarray[np.float64_t,ndim=2] py_xyz):
    
    cdef int nwat = len(py_xyz)/3 # number of water molecules
    cdef double *xyz = <double *> py_xyz.data
    cdef double H1x,H1y,H1z,H2x,H2y,H2z,n2,n3
    cdef double Exx = -2.3067236 # -2.0486*1.126
    cdef double Eyy =  2.0364836 # 1.8086*1.126
    cdef double Ezz =  0.27024   # 0.2400*1.126
    cdef int i, k, ind
    
    cdef np.ndarray[np.float64_t,ndim=1] py_dipt = np.zeros(nwat*6,dtype=np.float64)
    cdef double *dipt = <double *> py_dipt.data
    cdef np.ndarray[np.float64_t,ndim=1] py_B1 = np.zeros(nwat*3,dtype=np.float64)
    cdef double *B1 = <double *> py_B1.data
    cdef np.ndarray[np.float64_t,ndim=1] py_B2 = np.zeros(nwat*3,dtype=np.float64)
    cdef double *B2 = <double *> py_B2.data
    cdef np.ndarray[np.float64_t,ndim=1] py_B3 = np.zeros(nwat*3,dtype=np.float64)
    cdef double *B3 = <double *> py_B3.data
    
    with nogil, parallel():
        B = <double *> malloc(sizeof(double) * 9)
        T = <double *> malloc(sizeof(double) * 9)
        for i in prange(0,nwat*9,9): # loop over all coordinates of atoms of water molecules
            for k in range(9):
                B[k] = 0.0
                T[k] = 0.0
            
            H1x = xyz[i+3]-xyz[i]
            H1y = xyz[i+4]-xyz[i+1]
            H1z = xyz[i+5]-xyz[i+2]
            H2x = xyz[i+6]-xyz[i]
            H2y = xyz[i+7]-xyz[i+1]
            H2z = xyz[i+8]-xyz[i+2]

            B[3] = (H1x-H2x)
            B[4] = (H1y-H2y)
            B[5] = (H1z-H2z)
            B[6] = (H1x+H2x)
            B[7] = (H1y+H2y)
            B[8] = (H1z+H2z)
            
            n2 = (B[3]*B[3]+B[4]*B[4]+B[5]*B[5])**.5 # normalization Bvec_2
            n3 = (B[6]*B[6]+B[7]*B[7]+B[8]*B[8])**.5 # normalization Bvec_3

            B[3] /= n2
            B[4] /= n2
            B[5] /= n2
            B[6] /= n3
            B[7] /= n3
            B[8] /= n3

            B[0] = B[4]*B[8]-B[5]*B[7] # (Bvec_1,Bvec_2,Bvec_3) are orthonormal -> vector product
            B[1] = B[5]*B[6]-B[3]*B[8]
            B[2] = B[3]*B[7]-B[4]*B[6]
            
            # B dot T
            T[0] = B[0]*Exx
            T[1] = B[1]*Exx
            T[2] = B[2]*Exx
            T[3] = B[3]*Eyy
            T[4] = B[4]*Eyy
            T[5] = B[5]*Eyy
            T[6] = B[6]*Ezz
            T[7] = B[7]*Ezz
            T[8] = B[8]*Ezz

            ind = (i/9)*6
            
            # (B dot T) dot B.T
            dipt[ind]   = T[0]*B[0]+T[3]*B[3]+T[6]*B[6] # 11 .... here B.T is taken directly from B
            dipt[ind+1] = T[1]*B[1]+T[4]*B[4]+T[7]*B[7] # 22
            dipt[ind+2] = T[2]*B[2]+T[5]*B[5]+T[8]*B[8] # 33
            dipt[ind+3] = T[1]*B[0]+T[4]*B[3]+T[7]*B[6] # 21
            dipt[ind+4] = T[2]*B[0]+T[5]*B[3]+T[8]*B[6] # 31
            dipt[ind+5] = T[2]*B[1]+T[5]*B[4]+T[8]*B[7] # 32 .... matrix is symmetric

            ind = (i/9)*3
            
            B1[ind]   = B[0]
            B1[ind+1] = B[1]
            B1[ind+2] = B[2]
            B2[ind]   = B[3]
            B2[ind+1] = B[4]
            B2[ind+2] = B[5]
            B3[ind]   = B[6]
            B3[ind+1] = B[7]
            B3[ind+2] = B[8]
            
        free(B)
        free(T)
        
    return py_dipt,py_B1,py_B2,py_B3

@cython.boundscheck(False)
def NQRdipTen(np.ndarray[np.float64_t,ndim=2] py_xyz,
              np.ndarray[np.float64_t,ndim=1] py_charges,
              aufpunkt):
    
    cdef int n = 3*len(py_xyz)
    cdef int auf = <int> 3*aufpunkt # x,y,z for each atom -> *3
    cdef int auf3 = auf+3
    cdef int auf6 = auf+6
    cdef double *xyz = <double *> py_xyz.data
    cdef double *charges = <double *> py_charges.data     
    cdef double x,y,z, r2, f1,f2, f2_x,f2_y,f2_z, auf_x1,auf_y1,auf_z1,auf_x2,auf_y2,auf_z2
    cdef double tyH = 1.63
    cdef double tzH = 1.158921913
    cdef int i, j, k
#    cdef np.ndarray[np.float64_t,ndim=1] pyB = np.zeros(9,dtype=np.float64) # transformation matrix of aufpunkt wat
    cdef np.ndarray[np.float64_t,ndim=1] pyE = np.asarray([-2.0487,1.8086,0.2400],dtype=np.float64) # EFG matrix from Boykin
#    cdef np.ndarray[np.float64_t,ndim=1] pyT = np.zeros(9,dtype=np.float64) # temp matrix for mult
    #cdef np.ndarray[np.float64_t,ndim=1] py_dipt = np.zeros(15,dtype=np.float64)
    cdef np.ndarray[np.float64_t,ndim=2] py_dipt = np.zeros((n/3,15),dtype=np.float64)

    cdef double *dipt = <double *> py_dipt.data
#    cdef double *B = <double *> malloc(sizeof(double) * 9)
#    cdef double *T = <double *> malloc(sizeof(double) * 9)
    cdef double *E = <double *> pyE.data

    with nogil, parallel():
        B = <double *> malloc(sizeof(double) * 9)
        T = <double *> malloc(sizeof(double) * 9)
        for i in prange(0,n,3): # dip-dip T for to all atoms on every other molecule
            j = i/3 * 15
            for k in range(9):
                B[k] = 0.0
                T[k] = 0.0
            if i!=auf and i!=auf3 and i!=auf6:
                x = xyz[i]
                y = xyz[i+1]
                z = xyz[i+2]
                r2 = x*x+y*y+z*z
                f1 = pow(r2,-1.5) * charges[i/3]
                f2 = 3.0 * f1 / r2
                f2_x = f2 * x
                f2_y = f2 * y
                f2_z = f2 * z
                dipt[j] += f2_x * x - f1
                dipt[j+1] += f2_y * y - f1
                dipt[j+2] += f2_z * z - f1
                dipt[j+3] += f2_x * y
                dipt[j+4] += f2_x * z
                dipt[j+5] += f2_y * z
            elif i == auf:
                auf_x1 = xyz[auf+3]
                auf_y1 = xyz[auf+4]
                auf_z1 = xyz[auf+5]
                auf_x2 = xyz[auf+6]
                auf_y2 = xyz[auf+7]
                auf_z2 = xyz[auf+8]

                B[3] = (auf_x1-auf_x2)/tyH
                B[4] = (auf_y1-auf_y2)/tyH
                B[5] = (auf_z1-auf_z2)/tyH
                B[6] = (auf_x1+auf_x2)/tzH
                B[7] = (auf_y1+auf_y2)/tzH
                B[8] = (auf_z1+auf_z2)/tzH
                B[0] = B[4]*B[8]-B[5]*B[7] # (Bvec_1,Bvec_2,Bvec_3) are orthonormal -> vector product
                B[1] = B[5]*B[6]-B[3]*B[8]
                B[2] = B[3]*B[7]-B[4]*B[6]

                T[0] = B[0]*E[0]
                T[1] = B[1]*E[0]
                T[2] = B[2]*E[0]
                T[3] = B[3]*E[1]
                T[4] = B[4]*E[1]
                T[5] = B[5]*E[1]
                T[6] = B[6]*E[2]
                T[7] = B[7]*E[2]
                T[8] = B[8]*E[2]

                dipt[j+6] = T[0]*B[0]+T[3]*B[3]+T[6]*B[6]
                dipt[j+7] = T[1]*B[0]+T[4]*B[3]+T[7]*B[6]
                dipt[j+8] = T[2]*B[0]+T[5]*B[3]+T[8]*B[6]
                dipt[j+9] = T[0]*B[1]+T[3]*B[4]+T[6]*B[7]
                dipt[j+10] = T[1]*B[1]+T[4]*B[4]+T[7]*B[7]
                dipt[j+11] = T[2]*B[1]+T[5]*B[4]+T[8]*B[7]
                dipt[j+12] = T[0]*B[2]+T[3]*B[5]+T[6]*B[8]
                dipt[j+13] = T[1]*B[2]+T[4]*B[5]+T[7]*B[8]
                dipt[j+14] = T[2]*B[2]+T[5]*B[5]+T[8]*B[8]
            else:
                continue
        free(B)
        free(T)

    for i in range(1,n/3):
        for j in range(15):
            dipt[j] += dipt[i*15+j]
    
    return py_dipt[0]



    # # field gradient of aufpunkt water
    # # transformation of body fixed field gradient from Boykin (os 24.4.2015)

    # auf_x1 = xyz[auf+3]
    # auf_y1 = xyz[auf+4]
    # auf_z1 = xyz[auf+5]
    # auf_x2 = xyz[auf+6]
    # auf_y2 = xyz[auf+7]
    # auf_z2 = xyz[auf+8]

    # B[3] = (auf_x1-auf_x2)/tyH
    # B[4] = (auf_y1-auf_y2)/tyH
    # B[5] = (auf_z1-auf_z2)/tyH

    # B[6] = (auf_x1+auf_x2)/tzH
    # B[7] = (auf_y1+auf_y2)/tzH
    # B[8] = (auf_z1+auf_z2)/tzH

    # B[0] = B[4]*B[8]-B[5]*B[7]
    # B[1] = B[5]*B[6]-B[3]*B[8]
    # B[2] = B[3]*B[7]-B[4]*B[6]

    # T[0] = B[0]*E[0]
    # T[1] = B[1]*E[0]
    # T[2] = B[2]*E[0]
    # T[3] = B[3]*E[1]
    # T[4] = B[4]*E[1]
    # T[5] = B[5]*E[1]
    # T[6] = B[6]*E[2]
    # T[7] = B[7]*E[2]
    # T[8] = B[8]*E[2]
    
    # dipt[6] = T[0]*B[0]+T[3]*B[3]+T[6]*B[6]
    # dipt[7] = T[1]*B[0]+T[4]*B[3]+T[7]*B[6]
    # dipt[8] = T[2]*B[0]+T[5]*B[3]+T[8]*B[6]
    # dipt[9] = T[0]*B[1]+T[3]*B[4]+T[6]*B[7]
    # dipt[10] = T[1]*B[1]+T[4]*B[4]+T[7]*B[7]
    # dipt[11] = T[2]*B[1]+T[5]*B[4]+T[8]*B[7]
    # dipt[12] = T[0]*B[2]+T[3]*B[5]+T[6]*B[8]
    # dipt[13] = T[1]*B[2]+T[4]*B[5]+T[7]*B[8]
    # dipt[14] = T[2]*B[2]+T[5]*B[5]+T[8]*B[8]
    
    # for i in prange(auf+9,n,3, nogil=True): # dip-dip T for to all atoms on every other molecule 
    #     x = xyz[i]
    #     y = xyz[i+1]
    #     z = xyz[i+2]
    #     r2 = x*x+y*y+z*z
    #     f1 = pow(r2,-1.5) * charges[i/3]
    #     f2 = 3.0 * f1 / r2
    #     f2_x = f2 * x
    #     f2_y = f2 * y
    #     f2_z = f2 * z
    #     dipt[0] += f2_x * x - f1
    #     dipt[1] += f2_y * y - f1
    #     dipt[2] += f2_z * z - f1
    #     dipt[3] += f2_x * y
    #     dipt[4] += f2_x * z
    #     dipt[5] += f2_y * z

    # return dipt


# @cython.boundscheck(False)
# def NQRdipTen(np.ndarray[np.float64_t,ndim=2] py_xyz,
#               np.ndarray[np.float64_t,ndim=1] py_charges):
    
#     cdef int n = 3*len(py_xyz)
#     cdef double *xyz = <double *> py_xyz.data
#     cdef double *charges = <double *> py_charges.data     
#     cdef double x,y,z, r2, f1,f2, f2_x,f2_y,f2_z
#     cdef int i
#     cdef np.ndarray[np.float64_t,ndim=1] dipt = np.zeros(6,dtype=np.float64)

#     for i in prange(0,n,3, nogil=True):
        
#         x = xyz[i]
#         y = xyz[i+1]
#         z = xyz[i+2]
        
#         if x > 0.1 or y > 0.1 or z > 0.1:

#             r2 = x*x+y*y+z*z

#             f1 = pow(r2,-1.5) * charges[i/3]
#             f2 = 3.0 * f1 / r2
#             f2_x = f2 * x
#             f2_y = f2 * y
#             f2_z = f2 * z

#             dipt[0] += f2_x * x - f1
#             dipt[1] += f2_y * y - f1
#             dipt[2] += f2_z * z - f1
#             dipt[3] += f2_x * y
#             dipt[4] += f2_x * z
#             dipt[5] += f2_y * z

#     return dipt


# def dipTen(np.ndarray[np.float64_t,ndim=1] rv,
#            np.float64_t r2):


#         # np.ndarray[np.float64_t,ndim=1] coo1,
#         #    np.ndarray[np.float64_t,ndim=1] coo2,
#         #    np.float64_t r2):
#     """
#     dipTen(coo1,coo2) -> returns dip tensor

#     Calculates the dipole T-tensor (see Daniel's NMR code).
#     """
# #    cdef double* c_coo1 = <double *> coo1.data
# #    cdef double* c_coo2 = <double *> coo2.data
#     cdef double* c_rv = <double *> rv.data

# #    cdef double rv[3]
# #    cdef double r2 = 0.0
#     cdef double f1 = 0.0, f2 = 0.0

# #    cdef int i = 0
# #
# #    for i in range(3):
# #        rv[i] = c_coo2[i] - c_coo1[i]
# #        r2+=rv[i]*rv[i]
#     f1 = pow(r2,-1.5)
#     f2 = 3.0 * f1 / r2

#     cdef np.ndarray[np.float64_t,ndim=1] dipt = np.empty(6,dtype=np.float64)

#     dipt[0]=f2*c_rv[0]*c_rv[0]-f1
#     dipt[1]=f2*c_rv[1]*c_rv[1]-f1     # minus 1 in definition of dipole dipole tensor
#     dipt[2]=f2*c_rv[2]*c_rv[2]-f1
#     dipt[3]=f2*c_rv[0]*c_rv[1]        # not times 2 for off-diag (see "The intermol. NOE is strongly influenced by dynamics", G^2 must be 1)
#     dipt[4]=f2*c_rv[0]*c_rv[2]
#     dipt[5]=f2*c_rv[1]*c_rv[2]
#     return dipt


def calcOctant(np.ndarray[np.float64_t,ndim=2,mode="c"] com,
               isolute, nmolecules):
    """
    calcOctant(com, isolute, nmolecules)

    In a box, that is centered on a specific residue and rotated to its body-fixed frame, this determines in which
    octant around this molecule each other molecule is centered. For speed and memory efficiency, this information
    is encoded as follows:

    x y z sign as bits in a binary number (0 for negative sign, 1 for positive sign),
    3-bit number gives 8 dec. numbers -> 8 array indices needed

    x y z
    0 0 0 -> 0
    0 0 1 -> 1
    0 1 0 -> 2
    0 1 1 -> 3
    1 0 0 -> 4
    1 0 1 -> 5
    1 1 0 -> 6
    1 1 1 -> 7

    """
    cdef double *ccom    = <double *> com.data
    cdef int nmol=<int>nmolecules, islt=<int>isolute, j
    cdef np.ndarray[np.int32_t, ndim=1] octants = np.zeros(nmol,dtype=np.int32)
    cdef int* coct = <int *> octants.data
    cdef int x, y, z

    for j in range(nmol):
        if j == islt:
            continue
        x=sgn2bin(ccom[j*3])*4
        y=sgn2bin(ccom[j*3+1])*2
        z=sgn2bin(ccom[j*3+2])
        coct[j]=x|y|z
    
    return octants

@cython.boundscheck(False)
def sumMDCage(double [:,:,:,:] mdcage_ts,
              double [:,:] dip_wat,
              int nres_wat, char [:] ds, int maxshell, int rep, int frame):

    cdef int w, shell, i
    for w in range(nres_wat):
        shell = ds[w]-1
        if shell < maxshell:
            for i in range(3):
                mdcage_ts[rep,shell,frame,i] += dip_wat[w,i]
        else:
            for i in range(3):
                mdcage_ts[rep,maxshell,frame,i] += dip_wat[w,i]

@cython.boundscheck(False)
def sumMDCageSingle(double [:,:,:] mdcage_ts, double [:,:] dip_wat,
                    char [:] ds, int maxshell, int frame):
    
    cdef int w, shell, i, nres_wat = dip_wat.shape[0]

    for w in range(nres_wat):
        shell = ds[w]-1
        if shell < maxshell:
            for i in range(3):
                mdcage_ts[shell,frame,i] += dip_wat[w,i]
        else:
            for i in range(3):
                mdcage_ts[maxshell,frame,i] += dip_wat[w,i]
                
@cython.boundscheck(False)
def calcAngularMomentum(double [:,:] coor, double [:,:] vel, double [:] masses, int natoms):

    cdef np.ndarray[np.float64_t, ndim=1] L = np.zeros(3)
    cdef double *cL = <double *> L.data
    cdef int i

    for i in range(natoms):
        cL[0] += masses[i] * (coor[i,1] * vel[i,2] - coor[i,2] * vel[i,1])
        cL[1] += masses[i] * (coor[i,2] * vel[i,0] - coor[i,0] * vel[i,2])
        cL[2] += masses[i] * (coor[i,0] * vel[i,1] - coor[i,1] * vel[i,0])
    
    return L

@cython.boundscheck(False)
def calcInertiaTensor(double [:,:] coor, double [:] masses, int natoms):

    cdef np.ndarray[np.float64_t, ndim=2] I = np.zeros((3,3))
    cdef double *cI = <double *> I.data
    cdef int i
    cdef double x, y, z

    for i in range(natoms):
        x = coor[i,0]
        y = coor[i,1]
        z = coor[i,2]
        cI[0] += masses[i] * (y*y+z*z)
        cI[1] -= masses[i] * x*y
        cI[2] -= masses[i] * x*z
        cI[4] += masses[i] * (x*x+z*z)
        cI[5] -= masses[i] * y*z
        cI[8] += masses[i] * (x*x+y*y)

    cI[3] = cI[1]
    cI[6] = cI[2]
    cI[7] = cI[5]
        
    return I


@cython.boundscheck(False)
def calcResidenceTimeseries(char [:,:] ds, int nshells):

    cdef int nmol    = len(ds)
    cdef int nt      = len(ds[0])
    
    cdef char [:,:,:] ts = np.zeros((nshells,nmol,nt),dtype=np.int8)

    cdef int i,j,ind

    with nogil, parallel():
        for i in prange(nmol):
            for j in range(nt):
                ts[ds[i,j]-1, i, j] += 1
    
    return np.asarray(ts)

@cython.boundscheck(False)
def calcAngularDisplacement(double [:,:] wts, double dt):

    cdef int tn = wts.shape[0]
    cdef int i, j, k
    cdef double [:,:] integral = np.zeros((tn,3))
    cdef double [:,:] msd = np.zeros((tn,3))
    cdef int [:] ctr = np.zeros(tn, dtype=np.int32)
    
    # calculate integral
    for i in range(1,tn):
        for j in range(3):
            k = i-1
            integral[i,j] = integral[k,j] + (wts[k,j] + wts[i,j]) * 0.5 * dt

    # calculate angular displacement
    for i in prange(tn,nogil=True,schedule=dynamic):
        for j in range(tn-i):
            for k in range(3):
                msd[i,k] += pow(integral[j,k] - integral[j+i,k], 2)
            ctr[i] += 1

    for i in range(tn):
        for j in range(3):
            msd[i,j] /= ctr[i]

    return msd

@cython.boundscheck(False)
def findNearestAtom(double [:,:] coor_core, double [:,:] coor_surr, double [:] mindist,
                    double [:] mindist2, double [:,:] minvec, int [:] next_id):

    cdef int n_core = coor_core.shape[0]
    cdef int n_surr = coor_surr.shape[0]
    cdef int i, j

    cdef double tmp_x, tmp_y, tmp_z, dist2
    
    for i in prange(n_core, nogil=True):
        for j in range(n_surr):
            tmp_x = coor_surr[j,0] - coor_core[i,0]
            tmp_y = coor_surr[j,1] - coor_core[i,1]
            tmp_z = coor_surr[j,2] - coor_core[i,2]

            dist2 = tmp_x*tmp_x + tmp_y*tmp_y + tmp_z*tmp_z

            if dist2 < mindist2[i]:
                mindist2[i] = dist2
                next_id[i] = j

        mindist[i] = sqrt(mindist2[i])
        for j in range(3):
            minvec[i,j] = coor_surr[next_id[i],j] - coor_core[i,j]

@cython.boundscheck(False)
def checkHBond(double [:,:] coor_surr, double [:,:] coor_oh2, int nres_surr, double maxdist):
    # this function is designed only for water hydrogen bonds!
    cdef int sites_per_res = coor_surr.shape[0] / nres_surr
    cdef int nsurr = coor_surr.shape[0] / sites_per_res
    cdef int nwat = coor_oh2.shape[0] / 3
    cdef int i, j, k, l, idx, idx2, idx3
    
    cdef char [:] hbond = np.zeros(nwat, dtype=np.int8)

    cdef double dx, dy, dz, dx2, dy2, dz2, dot, dot2, dist, dist2, cosine

    # loop over water molecules
    for i in prange(nwat, nogil=True):
        if hbond[i] == 0:
            # loop over h atoms
            for j in range(2):
                if hbond[i] == 0:
                    idx = i*3+1+j
                    idx3 = i*3
                    # loop over surrounding molecules
                    for k in range(nsurr):
                        if hbond[i] == 0:
                            # loop over oxygen atoms
                            for l in range(sites_per_res):
                                idx2 = k*sites_per_res+l
                                dx = coor_surr[idx2,0] - coor_oh2[idx,0]
                                dy = coor_surr[idx2,1] - coor_oh2[idx,1]
                                dz = coor_surr[idx2,2] - coor_oh2[idx,2]
                                dot = dx*dx + dy*dy + dz*dz
                                dist = sqrt(dot)
                                if dist < maxdist:
                                    dx2 = coor_oh2[idx3,0] - coor_oh2[idx,0]
                                    dy2 = coor_oh2[idx3,1] - coor_oh2[idx,1]
                                    dz2 = coor_oh2[idx3,2] - coor_oh2[idx,2]
                                    dot2 = dx2*dx2 + dy2*dy2 + dz2*dz2
                                    dist2 = sqrt(dot2)
                                    cosine = (dx * dx2 + dy * dy2 + dz * dz2) / (dist * dist2)
                                    if cosine < -0.95:
                                        hbond[i] = 1

    return np.asarray(hbond)

@cython.boundscheck(False)
def sphHarmMatrix(double [:,:] coor, np.complex128_t [:,:] y_mat, int lmax):
    cdef double r, pol, azi
    cdef int nat = coor.shape[0], i, l1, m1, m1r, l2, m2, m2r

    for i in range(nat):
        r = sqrt(coor[i,0]*coor[i,0] + coor[i,1]*coor[i,1] + coor[i,2]*coor[i,2])
        pol = np.arccos(coor[i,2] / r)
        azi = np.arctan2(coor[i,1] , coor[i,0])

        for l1 in range(lmax+1):
            for m1 in range(2*l1+1):
                m1r = m1 - l1
                for l2 in range(lmax+1):
                    for m2 in range(2*l2+1):
                        m2r = m2 - l2
                        y_mat[2*l1+m1,2*l2+m2] += sph_harm(m1r,l1,azi,pol) * np.conjugate(sph_harm(m2r,l2,azi,pol))

@cython.boundscheck(False)
def calcFourierLaplaceTransform(double [:] data_x, double [:] data_y, double w):
    cdef double wt, wt1
    cdef double dt = data_x[1] - data_x[0]
    cdef double laplace_re = 0.0
    cdef double laplace_im = 0.0
    cdef int i
    cdef int n = data_x.shape[0]-1
    cdef double [:,:] transform = np.zeros((n,2))

    for i in prange(n, nogil=True):
        wt = w*data_x[i]
        wt1 = w*data_x[i+1]

        transform[i,0] = (cos(wt)*data_y[i] + cos(wt1)*data_y[i+1])*0.5*dt
        transform[i,1] = (sin(wt)*data_y[i] + sin(wt1)*data_y[i+1])*0.5*dt

    for i in range(n):
        laplace_re += transform[i,0]
        laplace_im += transform[i,1]

    return float(laplace_re), float(laplace_im)

@cython.boundscheck(False)
def calcRotationMatrix(double [:,:] coorA, double [:,:] coorB):
    """
    calcRotationMatrix(coor,coorA,coorB)

    Applies the Berthold-Horn-algorithm to a set of coordinates xyz, inplace.

    Args:
        coor               .. coordinates of all atoms
        coorA              .. template coordinates
        coorB              .. corresponding coordinates of the current frame
    """
    cdef double [:,:] R = np.zeros((3,3))

    GetRotation(&R[0,0], coorA.shape[0], &coorA[0,0], &coorB[0,0], 0)

    return np.asarray(R)

@cython.boundscheck(False)
def applyRotationMatrix(double [:,:] coor, double [:,:] R):
    """
    calcRotationMatrix(coor,coorA,coorB)

    Applies the Berthold-Horn-algorithm to a set of coordinates xyz, inplace.

    Args:
        coor               .. coordinates of all atoms
        coorA              .. template coordinates
        coorB              .. corresponding coordinates of the current frame
    """
    cdef int N = coor.shape[0]
    cdef double tmpx, tmpy, tmpz
    cdef int i    

    for i in prange(N, nogil=True):
        tmpx = R[0,0] * coor[i,0] + R[0,1] * coor[i,1] + R[0,2] * coor[i,2]
        tmpy = R[1,0] * coor[i,0] + R[1,1] * coor[i,1] + R[1,2] * coor[i,2]
        tmpz = R[2,0] * coor[i,0] + R[2,1] * coor[i,1] + R[2,2] * coor[i,2]
        coor[i,0] = tmpx
        coor[i,1] = tmpy
        coor[i,2] = tmpz
