import os
import pathlib
import tempfile
import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback path
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # pragma: no cover - fallback path
        def _wrap(func):
            return func
        return _wrap


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
    j is the index such that x_grid[j] <= x < x_grid[j+1].
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

    t2 = t * t
    t3 = t2 * t
    h00 = 1 - 3 * t2 + 2 * t3
    h10 = t - 2 * t2 + t3
    h01 = 3 * t2 - 2 * t3
    h11 = t3 - t2

    dx1 = x2 - x0
    dx1 = np.where(dx1 > 0, dx1, 1.0)
    dx2 = x3 - x1
    dx2 = np.where(dx2 > 0, dx2, 1.0)

    w0 = -h / dx1 * h10
    w1 = h00 - h / dx2 * h11
    w2 = h01 + h / dx1 * h10
    w3 = h / dx2 * h11

    return (np.stack([w0, w1, w2, w3], axis=-1),
            np.stack([i0, i1, i2, i3], axis=-1))


def _get_cubic_coeffs_deriv(x, x_grid, j):
    """
    Derivative of cubic interpolation weights with respect to x.
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
    Perform bicubic interpolation over the first two dimensions
    (metallicity, mass) at fixed ieep.
    """
    if HAS_NUMBA:
        return _interpolator_bicubic_numba(grid, wf, ifehs, wm, imasses,
                                           np.asarray(ieep, dtype=np.int64))
    res = np.zeros(wf.shape[0], dtype=np.float64)
    ieep = np.asarray(ieep, dtype=np.int64)
    for i in range(4):
        w_i = wf[:, i]
        idx_i = ifehs[:, i]
        for j in range(4):
            res += w_i * wm[:, j] * grid[idx_i, imasses[:, j], ieep]
    return res


def _interpolator_tricubic(grid, wf, ifehs, wm, imasses, we, ieeps):
    """
    Perform tricubic interpolation over (metallicity, mass, EEP).
    """
    if HAS_NUMBA:
        return _interpolator_tricubic_numba(grid, wf, ifehs, wm, imasses, we,
                                            np.asarray(ieeps, dtype=np.int64))
    res = np.zeros(wf.shape[0], dtype=np.float64)
    ieeps = np.asarray(ieeps, dtype=np.int64)
    for i in range(4):
        w_i = wf[:, i]
        idx_i = ifehs[:, i]
        for j in range(4):
            w_ij = w_i * wm[:, j]
            idx_j = imasses[:, j]
            for k in range(4):
                res += w_ij * we[:, k] * grid[idx_i, idx_j, ieeps[:, k]]
    return res


@njit(cache=True)
def _interpolator_bicubic_numba(grid, wf, ifehs, wm, imasses, ieep):
    n = wf.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        acc = 0.0
        e = ieep[t]
        for i in range(4):
            wi = wf[t, i]
            ii = ifehs[t, i]
            for j in range(4):
                acc += wi * wm[t, j] * grid[ii, imasses[t, j], e]
        out[t] = acc
    return out


@njit(cache=True)
def _interpolator_tricubic_numba(grid, wf, ifehs, wm, imasses, we, ieeps):
    n = wf.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        acc = 0.0
        for i in range(4):
            wi = wf[t, i]
            ii = ifehs[t, i]
            for j in range(4):
                wij = wi * wm[t, j]
                jj = imasses[t, j]
                for k in range(4):
                    acc += wij * we[t, k] * grid[ii, jj, ieeps[t, k]]
        out[t] = acc
    return out


def steffen_interp(y_m1, y_0, y_1, y_2, t):
    """
    Vectorized piecewise-monotonic cubic interpolation (Steffen 1990).
    y_m1, y_0, y_1, y_2 are the values at x=-1, 0, 1, 2.
    t is the fractional distance between 0 and 1.
    """
    s_0 = y_1 - y_0
    s_m1 = np.where(np.isfinite(y_m1), y_0 - y_m1, s_0)
    s_1 = np.where(np.isfinite(y_2), y_2 - y_1, s_0)

    p_0 = (s_m1 + s_0) / 2.0
    p_1 = (s_0 + s_1) / 2.0

    y_prime_0 = np.where(s_m1 * s_0 > 0, np.sign(s_0) * np.minimum(np.abs(p_0), np.minimum(2*np.abs(s_m1), 2*np.abs(s_0))), 0.0)
    y_prime_1 = np.where(s_0 * s_1 > 0, np.sign(s_0) * np.minimum(np.abs(p_1), np.minimum(2*np.abs(s_0), 2*np.abs(s_1))), 0.0)

    # Evaluate the cubic polynomial
    return y_0 + y_prime_0 * t + (3*s_0 - 2*y_prime_0 - y_prime_1) * t**2 + (y_prime_0 + y_prime_1 - 2*s_0) * t**3

def solve_steffen_t(y_m1, y_0, y_1, y_2, target_y):
    """
    Find t in [0, 1] such that steffen_interp(..., t) == target_y.
    """
    s_0 = y_1 - y_0
    s_m1 = np.where(np.isfinite(y_m1), y_0 - y_m1, s_0)
    s_1 = np.where(np.isfinite(y_2), y_2 - y_1, s_0)

    p_0 = (s_m1 + s_0) / 2.0
    p_1 = (s_0 + s_1) / 2.0

    y_prime_0 = np.where(s_m1 * s_0 > 0, np.sign(s_0) * np.minimum(np.abs(p_0), np.minimum(2*np.abs(s_m1), 2*np.abs(s_0))), 0.0)
    y_prime_1 = np.where(s_0 * s_1 > 0, np.sign(s_0) * np.minimum(np.abs(p_1), np.minimum(2*np.abs(s_0), 2*np.abs(s_1))), 0.0)

    a = y_prime_0 + y_prime_1 - 2*s_0
    b = 3*s_0 - 2*y_prime_0 - y_prime_1
    c = y_prime_0
    
    # Linear initial guess
    t = (target_y - y_0) / s_0
    
    # Newton-Raphson iteration
    for _ in range(3):
        ft = a*t**3 + b*t**2 + c*t + y_0 - target_y
        dft = 3*a*t**2 + 2*b*t + c
        t = t - ft / np.where(dft != 0, dft, 1e-10) # protect against div by zero
        t = np.clip(t, 0.0, 1.0)
    
    return t
