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
