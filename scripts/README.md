# Python Scripts for Computing Minimum Lattice Thermal Conductivity

Here, we include:

* An implementation of minimum lattice thermal conductivity: `minikappa.py`
* A modified version of Phonopy (2.17.1) source code to compute off-diagonal group velocities
  * To use this script, replace the original Phonopy files with the modified ones found in phonopy_files
    - `api_phonopy.py` &rarr; `from phonopy import Phonopy`
    - `mesh.py` &rarr; `from phonopy.phonon.mesh import Mesh`
    - `group_velocity.py` &rarr; `from phonopy.phonon.group_velocity import GroupVelocity`
* A script to compute minimum lattice thermal conductivity using data in the example folder
  * To run the example, go to `example` and run `python get_minikappa.py` or `python ../minikappa.py`
  * You will likely want to update the input parameters for your specific compounds
  * Only a structure file (e.g., `POSCAR-unitcell`) and harmonic force constants (e.g., `FORCE_CONSTANTS`) are required to compute $\kappa_{L}^{min}$
    - Alternatively, you can use a `phonopy.yaml` file and harmonic force constants
