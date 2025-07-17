import itertools
from pathlib import Path

import numpy as np
import phonopy
from phonopy import Phonopy

"""
author: @yixia
author: @dgaines2
A python class to compute kL_min

The calculation of kL_min relies on modified versions of:
    - api_phonopy.py -> from phonopy import Phonopy
    - mesh.py -> from phonopy.phonon.mesh import Mesh
    - group_velocity.py -> from phonopy.phonon.group_velocity import GroupVelocity
for Phonopy version 2.17.1
"""


class LibraryModificationRequired(Exception):
    """Raised when the required library modifications have not been made."""
    pass


class MinikappaManager:
    def __init__(
        self,
        phonon,
        mesh=25.0,
        temperatures=[300.0, 600.0, 900.0],
        tau_factors=[2.0],
    ):
        """
        Args:
            phonon (phonopy.Phonopy)
            mesh (float | 1x3 array[int]): q-point mesh density
            temperatures (array[float]) in Kelvin
            tau_factors (array[float])
        """
        self.phonon = phonon
        self.mesh = mesh
        self.temperatures = temperatures
        self.tau_factors = tau_factors

    def get_minikappa(self, verbose=True):
        def vprint(message, verbose=True):
            if verbose:
                print(message)

        # Mesh
        vprint(f"Running phonon mesh...\n", verbose)
        try:
            self.phonon.run_mesh(
                self.mesh,
                with_eigenvectors=False,
                is_gamma_center=True,
                with_full_group_velocities=True,
                is_time_reversal=False,
                is_mesh_symmetry=False,
            )
        except TypeError as e:
            if "full_group_velocities" in str(e):
                raise LibraryModificationRequired(
                    "Phonopy must be modified to calculate off-diagonal group "
                    "velocities. Be sure to make modifications to Phonopy 2.17.1. "
                    "See more details in the README here: "
                    "https://github.com/yimavxia/Minikappa/tree/main/scripts"
                ) from e
            else:
                raise
        mesh_dict = self.phonon.get_mesh_dict()
        qpoints = mesh_dict["qpoints"]
        freqs = mesh_dict["frequencies"]
        gvfull = mesh_dict["group_velocities_full"]

        # Units
        hbar = 1.054571726470000e-022
        kB = 1.380648813000000e-023
        pi = np.pi

        primcell = self.phonon.primitive.cell.T
        volpc = np.abs(np.dot(np.cross(primcell[1], primcell[2]), primcell[0])) / 1000.0
        gvfull = gvfull / 10.0
        freqs = freqs * 2 * pi
        nband = len(freqs[0])
        freqcf = 0.1

        # kappa
        nqpt = len(qpoints)
        nband = len(freqs[0])
        results = {}
        for temperature in self.temperatures:
            results[temperature] = {}
            for tau_factor in self.tau_factors:
                results[temperature][tau_factor] = {}
                vprint(
                    f"Calculating minikappa... T={temperature}K, tau={tau_factor}",
                    verbose,
                )
                kappaband = np.zeros(
                    (nband, nband, 3, 3), dtype=np.complex128, order="C"
                )
                for iq, i, j, k, kp in itertools.product(
                    range(nqpt),
                    range(nband),
                    range(nband),
                    range(3),
                    range(3),
                ):
                    omega1 = freqs[iq, i]
                    omega2 = freqs[iq, j]
                    if omega1 > freqcf and omega2 > freqcf:
                        if omega1 / 2 / pi > 0:
                            Gamma1 = freqs[iq, i] / 2 / pi * tau_factor
                        else:
                            Gamma1 = 1e10
                        if omega2 / 2 / pi > 0:
                            Gamma2 = freqs[iq, j] / 2 / pi * tau_factor
                        else:
                            Gamma2 = 1e10
                        fBE1 = 1.0 / (np.exp(hbar * omega1 / kB / temperature) - 1.0)
                        fBE2 = 1.0 / (np.exp(hbar * omega2 / kB / temperature) - 1.0)
                        tmpv = (gvfull[iq, i, j, k] * gvfull[iq, j, i, kp]).real
                        kappaband[i,j,k,kp] += (omega1+omega2)/2 * \
                                            (fBE1*(fBE1+1)*omega1+fBE2*(fBE2+1)*omega2) * tmpv \
                                            / (4*(omega1-omega2)**2+(Gamma1+Gamma2)**2) \
                                            * (Gamma1+Gamma2)

                # conversion
                kappaband = (kappaband * 1e21 * hbar**2) / (
                    kB * temperature * temperature * volpc * nqpt
                )
                kappaD = np.zeros((3, 3), dtype=np.complex128, order="C")
                kappaOD = np.zeros((3, 3), dtype=np.complex128, order="C")
                kappaF = np.zeros((3, 3), dtype=np.complex128, order="C")
                for i in range(nband):
                    for j in range(nband):
                        kappaF += kappaband[i, j]
                        if i == j:
                            kappaD += kappaband[i, j]
                        else:
                            kappaOD += kappaband[i, j]
                kappaD = kappaD.real
                kappaOD = kappaOD.real
                kappaF = kappaF.real
                results[temperature][tau_factor]["D"] = kappaD
                results[temperature][tau_factor]["OD"] = kappaOD
                results[temperature][tau_factor]["F"] = kappaF

                output_filename = f"minikappa-{temperature}-{tau_factor}.dat"
                vprint(f"Writing to {output_filename}", verbose)
                with open(output_filename, "w+") as fw:
                    for kappa_matrix in [kappaD, kappaOD, kappaF]:
                        kappa_matrix = np.round(
                            kappa_matrix.reshape(9, 1).flatten(),
                            decimals=8,
                        )
                        fw.write(
                            "".join([f"{num:>14.8f}" for num in kappa_matrix]) + "\n"
                        )

                vprint(
                    "Diagonal part of minimum thermal conductivity: "
                    f"{kappaD[0, 0]:.3f}",
                    verbose,
                )
                vprint(
                    "Off-diagonal part of minimum thermal conductivity: "
                    f"{kappaOD[0, 0]:.3f}",
                    verbose,
                )
                vprint(
                    "Total minimum thermal conductivity: " f"{kappaF[0, 0]:.3f}",
                    verbose,
                )
                vprint("", verbose)
        return results

    @classmethod
    def from_phonopy_yaml(
        cls,
        yaml_path="phonopy.yaml",
        force_constants_filename="FORCE_CONSTANTS",
        kwargs=None,
    ):
        """
        Args:
            yaml_path (str): path to phonopy.yaml
            force_constants_filename (str): path to harmonic force constants file
            kwargs (optional, dict): dictionary with mesh, temperatures, or tau
                factors
        """
        if not Path(yaml_path).exists():
            raise Exception(f"{yaml_path} not found")
        if not Path(force_constants_filename).exists():
            raise Exception(f"{force_constants_filename} not found")
        if kwargs is None:
            kwargs = {}

        phonon = phonopy.load(
            yaml_path,
            force_constants_filename=force_constants_filename,
            is_symmetry=False,
        )
        return cls(phonon, **kwargs)

    @classmethod
    def from_data(
        cls,
        poscar_path,
        supercell_matrix,
        primitive_matrix,
        force_constants_filename="FORCE_CONSTANTS",
        kwargs=None,
    ):
        """
        Args:
            poscar_path (str): path to POSCAR file
            supercell_matrix (3x3 array[int]): supercell matrix
            primitive_matrix (3x3 array[float]): primitive matrix
            force_constants_filename (str): path to harmonic force constants file
            kwargs (optional, dict): dictionary with mesh, temperatures, or tau
                factors
        """
        if kwargs is None:
            kwargs = {}

        phonon = phonopy.load(
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            unitcell_filename=poscar_path,
            force_constants_filename=force_constants_filename,
            is_symmetry=False,
        )
        return cls(phonon, **kwargs)


def read_minikappa_file(fpath, verbose=False):
    minikappa_output = np.loadtxt(fpath)
    kappaD, kappaOD, kappaF = minikappa_output.reshape(3, 3, 3)
    if verbose:
        print(
            f"Diagonal part of minimum thermal conductivity: {kappaD[0, 0]:.3f}",
        )
        print(
            f"Off-diagonal part of minimum thermal conductivity: {kappaOD[0, 0]:.3f}",
        )
        print(
            f"Total minimum thermal conductivity: {kappaF[0, 0]:.3f}",
        )
    return kappaD, kappaOD, kappaF


if __name__ == "__main__":
    """
    Here's an example of using from_data to calculate kL_min
    """
    minikappa_manager = MinikappaManager.from_data(
        poscar_path="POSCAR-unitcell",
        supercell_matrix=np.eye(3) * 4,
        primitive_matrix=np.eye(3),
        kwargs={
            "mesh": [12, 12, 12],
            "temperatures": [600.0],
        },
    )
    results = minikappa_manager.get_minikappa(verbose=True)

    """
    Here's an example of using from_phonopy_yaml to calculate kL_min
    """
    # minikappa_manager = MinikappaManager.from_phonopy_yaml(
    #     kwargs={
    #         "mesh": 25.0,
    #         "temperatures": [300.0, 600.0, 900.0],
    #         "tau_factors": [1.0, 2.0],
    #     },
    # )
