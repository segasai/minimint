import os
import pathlib
import tempfile
import numpy as np


def get_data_path():
    path = os.environ.get('MINIMINT_DATA_PATH')
    if path is not None:
        return path
    path = os.path.join(str(pathlib.Path(__file__).parent.absolute()), 'data')
    os.makedirs(path, exist_ok=True)
    return path


def tail_head(fin, nskip, nout):
    """
    Read nout lines from fin after skipping nskip lines
    and put output in the temporary file. Return filename
    """
    fp = open(fin, 'r')
    fpout = tempfile.NamedTemporaryFile(delete=False, mode='w')
    i = -1
    for ll in fp:
        i += 1
        if i < nskip:
            continue
        print(ll, file=fpout)
        if i == (nskip + nout):
            break
    fp.close()
    fpout.close()
    return fpout.name


def _get_cubic_coeffs(x, x_grid, j):
    """
    Get cubic interpolation weights and indices for target x on x_grid.
    Uses a Catmull-Rom spline approach to ensure C1 continuity (continuous
    first derivatives) at grid boundaries.
    
    j is the index such that x_grid[j] <= x < x_grid[j+1].
    """
    n = len(x_grid)
    # We need a 4-point neighborhood: j-1, j, j+1, j+2.
    # We clip indices to the grid boundaries to handle edges.
    i0 = np.clip(j - 1, 0, n - 1)
    i1 = j
    i2 = np.clip(j + 1, 0, n - 1)
    i3 = np.clip(j + 2, 0, n - 1)

    x0 = x_grid[i0]
    x1 = x_grid[i1]
    x2 = x_grid[i2]
    x3 = x_grid[i3]

    # Normalized coordinate within the central interval [x1, x2]
    h = x2 - x1
    h_safe = np.where(h > 0, h, 1.0)
    t = (x - x1) / h_safe

    # Hermite basis functions
    t2 = t * t
    t3 = t2 * t
    h00 = 1 - 3 * t2 + 2 * t3
    h10 = t - 2 * t2 + t3
    h01 = 3 * t2 - 2 * t3
    h11 = t3 - t2

    # Finite difference estimates for derivatives at x1 and x2
    dx1 = x2 - x0
    dx1 = np.where(dx1 > 0, dx1, 1.0)
    dx2 = x3 - x1
    dx2 = np.where(dx2 > 0, dx2, 1.0)

    # Combine Hermite basis with derivative weights to get final 4-point weights
    w0 = -h / dx1 * h10
    w1 = h00 - h / dx2 * h11
    w2 = h01 + h / dx1 * h10
    w3 = h / dx2 * h11

    return (np.stack([w0, w1, w2, w3], axis=-1),
            np.stack([i0, i1, i2, i3], axis=-1))


def _get_cubic_coeffs_deriv(x, x_grid, j):
    """
    Get the derivative of the cubic interpolation weights with respect to x.
    Used for Newton-Raphson refinement of the EEP phase.
    """
    n = len(x_grid)
    i0 = np.clip(j - 1, 0, n - 1)
    i1 = j
    i2 = np.clip(j + 1, 0, n - 1)
    i3 = np.clip(j + 2, 0, n - 1)

    x0 = x_grid[i0]
    x1 = x_grid[i1]
    x2 = x_grid[i2]
    x3 = x_grid[i3]

    h = x2 - x1
    h_safe = np.where(h > 0, h, 1.0)
    t = (x - x1) / h_safe

    # Derivatives of the Hermite basis functions
    dt = 1.0 / h_safe
    dh00 = (-6 * t + 6 * t * t) * dt
    dh10 = (1 - 4 * t + 3 * t * t) * dt
    dh01 = (6 * t - 6 * t * t) * dt
    dh11 = (3 * t * t - 2 * t) * dt

    dx1 = x2 - x0
    dx1 = np.where(dx1 > 0, dx1, 1.0)
    dx2 = x3 - x1
    dx2 = np.where(dx2 > 0, dx2, 1.0)

    dw0 = -h / dx1 * dh10
    dw1 = dh00 - h / dx2 * dh11
    dw2 = dh01 + h / dx1 * dh10
    dw3 = h / dx2 * dh11

    return np.stack([dw0, dw1, dw2, dw3], axis=-1)


def _interpolator_bicubic(grid, wf, ifehs, wm, imasses, ieep):
    """
    Perform bicubic interpolation over the first two dimensions (metallicity, mass).
    """
    res = np.zeros(wf.shape[0])
    for i in range(4):
        w_i = wf[:, i]
        idx_i = ifehs[:, i]
        for j in range(4):
            res += w_i * wm[:, j] * grid[idx_i, imasses[:, j], ieep]
    return res


def _interpolator_tricubic(grid, wf, ifehs, wm, imasses, we, ieeps):
    """
    Perform tricubic interpolation over all three dimensions (metallicity, mass, EEP).
    """
    res = np.zeros(wf.shape[0])
    for i in range(4):
        w_i = wf[:, i]
        idx_i = ifehs[:, i]
        for j in range(4):
            w_ij = w_i * wm[:, j]
            idx_j = imasses[:, j]
            for k in range(4):
                res += w_ij * we[:, k] * grid[idx_i, idx_j, ieeps[:, k]]
    return res
