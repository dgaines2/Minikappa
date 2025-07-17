import sys

sys.path.append("..")
import numpy as np
from minikappa import MinikappaManager

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
print(results)
