import pandas as pd
import numpy as np
import csv


import numpy as np


def acceleration_cartesian(v, xi, eta, zeta):
    """
    Compute the Cartesian acceleration vector given components along
    an orthogonal (not necessarily orthonormal) basis defined by a direction vector v.

    Parameters
    ----------
    v : array-like, shape (3,)
        Reference direction vector.
    xi : float
        Acceleration along v.
    eta : float
        Acceleration along horizontal perpendicular axis (to the right when values are +).
    zeta : float
        Acceleration along perpendicular vertical axis (to the top when values are +).

    Returns
    -------
    a_cartesian : np.ndarray, shape (3,)
        Cartesian acceleration vector.
    basis : tuple of np.ndarray
        The three orthogonal basis vectors (v_dir, eta_dir, zeta_dir).
    """

    v = np.array(v, dtype=float)
    v_dir = v / np.linalg.norm(v)

    # Define world vertical (z-axis)
    z_axis = np.array([0.0, 0.0, 1.0])

    # Compute horizontal perpendicular direction (eta_dir)
    eta_dir = np.cross(z_axis, v_dir)
    if np.linalg.norm(eta_dir) < 1e-8:
        # v is parallel to z-axis; pick arbitrary horizontal axis
        eta_dir = np.array([1.0, 0.0, 0.0])
    else:
        eta_dir = eta_dir / np.linalg.norm(eta_dir)

    # Compute the third orthogonal direction (zeta_dir)
    zeta_dir = np.cross(eta_dir, v_dir)

    # Combine the three components
    a_cartesian = xi * v_dir - eta * eta_dir - zeta * zeta_dir
    return a_cartesian, (v_dir, eta_dir, zeta_dir)


v = np.array([1, 1, 0])

xi, eta, zeta = [1, 0, 0], [0, 1, 0], [0, 0, 1]

for i in range(3):
    cartesian_acceleration, orthogonal_basis = acceleration_cartesian(
        v, xi[i], eta[i], zeta[i]
    )

    print(cartesian_acceleration)
