<p align="center">
 <a href="https://github.com/cbc-univie/mdy-newanalysis-package/blob/master/docs/_static/newanalysis_logo.png" target="_blank" rel="noopener noreferrer">
  <img src="https://github.com/cbc-univie/mdy-newanalysis-package/blob/master/docs/_static/newanalysis_logo.png" alt="newanalysis Logo" width="400"/>
 </a>
</p>

### Installation

```python
git clone git@github.com:cbc-univie/mdy-newanalysis-package.git
cd mdy-newanalysis-package
pip install .
```

### Info

If you want to install the package you have to connect an ssh key to your github account.
Here are the instructions on how to generate ssh keys and link them to your github account:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

The python package "newanalysis" contains all our homebrew functions for MDAnalysis,
that formerly were embedded into the MDAnalysis package and are now available as a separate python module.
Be aware that the API syntax of MDAnalysis has changed compared to the old version we have used for a long time.

 ### CONTENTS OF THIS REPO (newanalysis_package)

- charmm_utilities:
  -- scripts for CHARMM

- newanalysis:
  -- the actual source for the newanalysis package
  

**Test Cases are moved to new repo: mdy-newanalysis-package-testdata**

- test_cases
  -- test cases, contain scripts for the new version (this package) in the new/ directories,
     as well as scripts for the old version of MDAnalysis (0.8.2dev, where newanalysis was
     embedded)
  -- please note that the trajectories were shortened considerably to reduce file size, so the
     results you will get for the test cases are obviously garbage




### HOW TO INSTALL NEWANALYSIS

1) add new channel via
conda config --add channels conda-forge

2) make a new conda environment that has MDAnalysis via
conda create -n new_mdanalysis_environment python numpy scipy mdanalysis cython h5py ipython

3) activate that environment via
conda activate new_mdanalysis_environment

4) go into the newanalysis_source folder

5) build and install the modules via
python3 setup.py build
python3 setup.py install

6) Esther says : In case of failure, look up the module where it failed, and delete the respective cpp file:
- src/helpers/correl.cpp
- src/helpers/diffusion.cpp
- src/helpers/helpers.cpp
- src/helpers/unfold.cpp
- src/voro/voro.cpp
- src/gfunction/gfunction.cpp
Try to build again, if still not working, contact me.



### HOW TO IMPORT INTO YOUR PYTHON SCRIPTS

- To be able to import these modules, change syntax in python script:

OLD:
from MDAnalysis.newanalysis.helpers import calcEnergyAtomic
from MDAnalysis.newanalysis.correl import correlate

NEW:
from newanalysis.helpers import calcEnergyAtomic
from newanalysis.correl import correlate



 ### CHANGES COMPARED TO THE OLD MDANALYSIS VERSIONS

- use new CHARMM psf type (default in Versions >= 40, otherwise specify "psf xplor" instead of "psf" in the write statement)

- if you want to read CHARMM velocity files: reformat with the script charmm_utilities/transform_VELtoDCD.inp
and make sure to convert the to the correct units!

- make sure to change syntax, e.g.
      OLD						NEW
      u.trajectory.numframes		       		u.trajectory.n_frames
      u.selectAtoms(...)		       		u.select_atoms(...)
      sel.numberOfResidues()		       		sel.n_residues
      sel.masses()					sel.masses
      sel.charges()			      		sel.charges
      sel.get_positions()		      		sel.positions
      sel.centerOfMassByResidue(coor,masses)   		sel.center_of_mass(compound='residues')

- some previously available functions within an atom selection are now only accessible directly (and need to be imported from helpers), e.g.
      OLD                                      	     	NEW
      sel.dipoleMomentByResidue(coor,charges,masses)    from newanalysis.functions import dipoleMomentByResidue
                                                        dipoleMomentByResidue(sel,coor,charges,masses) 
      sel.velcomByResidue(mass)                         from newanalysis.functions import velcomByResidue
                                                        velcomByResidue(sel,vels,mass)
      same for atomsPerResidue(), residueFirstAtom(), centerOfMassByResidue(), dipoleMoment()
