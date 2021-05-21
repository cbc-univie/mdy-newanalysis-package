import numpy
import sys

def f2c(sel,i=None,atomctr=0):
    if i!=None:
        try:
            box_f2c = sel.universe.atoms._cache['f2c']
        except KeyError:
            fine2coarse=numpy.empty(sel.universe.atoms.n_atoms,dtype=numpy.int32)
            running=0
            presname=""
            presnum=1
            ctr=0
            for a in sel.universe.atoms:
                if a.resname not in presname or presnum != a.resnum:
                    running+=1
                fine2coarse[ctr]=running-1
                presname=a.resname
                presnum=a.resnum
                ctr+=1
            sel.universe.atoms._cache['f2c']=fine2coarse
            box_f2c = sel.universe.atoms._cache['f2c']
        return box_f2c[i]
    else:
        fine2coarse=numpy.empty(sel.universe.atoms.n_atoms,dtype=numpy.int32)
        running=0
        presname=""
        presnum=1
        ctr=0
        for a in sel.universe.atoms.atoms:
            if a.resname not in presname or presnum != a.resnum:
                running+=1
            fine2coarse[ctr]=running-1
            presname=a.resname
            presnum=a.resnum
            ctr+=1
        box_f2c = fine2coarse
        sel_f2c = numpy.zeros(sel.n_atoms,dtype=numpy.int32)
        indices = sel.indices
        for j in range(sel.n_atoms):
            sel_f2c[i] = box_f2c[indices[i]]
        return sel_f2c


def calcTessellation(sel,maxshell=3,core_sel=None,volumes=None,face_areas=None):
    """
    Calculates the Voronoi tessellation of the whole box with the current AtomGroup as the core selection and returns the Delaunay distance matrix.
    
    Arguments:
    \tmaxshell:\tMaximum shell to calculate (default is 3)
    \tcore_sel:\tAtomGroup object of core selection (default is self)
    \tvolumes:\t1-dim numpy.float64 array for cell volume calculation (only together with face_areas!)
    \tface_areas:\t2-dim numpy.float64 array for surface area calculation (only together with volumes!)
    """
    from newanalysis.voro import calcTessellation
    if core_sel is None:
        core_sel = sel

    corelist=[f2c(sel.universe.atoms,i,atomctr) for atomctr,i in enumerate(core_sel.indices)]
    corelist_unique=numpy.array(list(set(corelist)),dtype=numpy.int32)

    return calcTessellation(sel.universe.atoms.positions.astype('float64'),sel.universe.coord.dimensions[0],f2c(sel.universe.atoms),sel.universe.atoms.n_atoms,sel.universe.atoms.n_residues,maxshell,corelist_unique,volumes,face_areas)

def atomsPerResidue(sel):
    apr=numpy.zeros(sel.n_residues,dtype=numpy.int32)
    curnum=sel.atoms[0].resid
    prvnum=curnum
    curname=sel.atoms[0].resname
    prvname=curname
    idx=0
    for i in range(sel.n_atoms):
        prvnum=curnum
        curnum=sel.atoms[i].resid
        prvname=curname
        curname=sel.atoms[i].resname
        if prvnum != curnum or prvname != curname:
            idx+=1
        apr[idx]+=1
    return apr


def residueFirstAtom(sel):
    rfa=numpy.zeros(sel.n_residues,dtype=numpy.int32)
    apr=atomsPerResidue(sel)
    ctr=0
    for i in range(sel.n_residues):
        rfa[i]=ctr
        ctr+=apr[i]
    return rfa

def centerOfMassByResidue(sel,**kwargs):
    """Returns an array of the centers of mass of all residues in the current AtomGroup"""
    from newanalysis.helpers import comByResidue
    if not "coor" in kwargs:
        coor=sel.positions.astype('float64')
    else:
        coor=kwargs["coor"]
    if not "masses" in kwargs:
        masses = numpy.ascontiguousarray(sel.masses)
    else:
        masses = kwargs["masses"]
    if not "apr" in kwargs:
        print("Warning: apr is now calculated each timestep (very slow)! Give apr as an argument to speed up code")
        apr=atomsPerResidue(sel)
    else:
        apr = kwargs["apr"]
    if not "rfa" in kwargs:
        print("Warning: rfa is now calculated each timestep (very slow)! Give rfa as an argument to speed up code")
        rfa=residueFirstAtom(sel)
    else:
        rfa = kwargs["rfa"]
    numres = sel.n_residues

    return comByResidue(coor,masses,numres,apr,rfa)

def dipoleMoment(sel):
    """Dipole moment of the selection."""
    return numpy.sum((sel.positions.astype('float64')-sel.center_of_mass())*sel.charges[:,numpy.newaxis],axis=0)

def dipoleMomentByResidue(sel,**kwargs):
    """Returns an array of the dipole moments of all residues in the current AtomGroup"""
    from newanalysis.helpers import dipByResidue
    if not "charges" in kwargs:
        charges=numpy.ascontiguousarray(sel.charges)
    else:
        charges = kwargs["charges"]
    if not "masses" in kwargs:
        masses=numpy.ascontiguousarray(sel.masses)
    else:
        masses=kwargs["masses"]
    if not "coor" in kwargs:
        coor=sel.positions.astype('float64')
    else:
        coor=kwargs["coor"]
    if not "com" in kwargs:
        com = centerOfMassByResidue(sel,masses=masses, coor=coor)
    else:
        com = kwargs["com"]
    if not "apr" in kwargs:
        print("Warning: apr is now calculated each timestep (very slow)! Give apr as an argument to speed up code")
        apr=atomsPerResidue(sel)
    else:
        apr = kwargs["apr"]
    if not "rfa" in kwargs:
        print("Warning: rfa is now calculated each timestep (very slow)! Give rfa as an argument to speed up code")
        rfa=residueFirstAtom(sel)
    else:
        rfa = kwargs["rfa"]
    numres = sel.n_residues

    return dipByResidue(coor,charges,masses,numres,apr,rfa,com)

def velcomByResidue(sel,**kwargs):
    """
    Gives array of center-of-mass velocites of all residues in the current AtomGroup.
    """
    from newanalysis.helpers import velcomByResidue
    if not "vels" in kwargs:
        vels = get_velocities(sel)
    else:
        vels = kwargs["vels"]
    if not "masses" in kwargs:
        masses=numpy.ascontiguousarray(sel.masses)
    else:
        masses=kwargs["masses"]
    if not "apr" in kwargs:
        print("Warning: apr is now calculated each timestep (very slow)! Give apr as an argument to speed up code")
        apr=atomsPerResidue(sel)
    else:
        apr = kwargs["apr"]
    if not "rfa" in kwargs:
        print("Warning: rfa is now calculated each timestep (very slow)! Give rfa as an argument to speed up code")
        rfa=residueFirstAtom(sel)
    else:
        rfa = kwargs["rfa"]
    numres = sel.n_residues

    return velcomByResidue(vels,masses,numres,apr,rfa)

def get_velocities(sel):
    from MDAnalysis.units import timeUnit_factor
    vel=numpy.ascontiguousarray(sel.positions*timeUnit_factor['AKMA'],dtype='double')
    return vel

