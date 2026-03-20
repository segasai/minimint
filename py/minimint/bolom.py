import glob
import itertools
import re
import os
import astropy.table as atpy
import numpy as np
from .utils import get_data_path, tail_head, _get_cubic_coeffs

try:
    from numba import njit
    HAS_NUMBA = True
except Exception:  # pragma: no cover - fallback path
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # pragma: no cover - fallback path
        def _wrap(func):
            return func
        return _wrap

POINTS_NPY = 'bolom_points.npy'
FILT_NPY = 'filt_%s.npy'


def _interpolator_4cubic(grid, ws, idxs):
    """
    Perform 4D cubic interpolation for bolometric corrections.
    """
    if HAS_NUMBA:
        return _interpolator_4cubic_numba(grid, ws[0], idxs[0], ws[1], idxs[1],
                                          ws[2], idxs[2], ws[3], idxs[3])
    res = np.zeros(ws[0].shape[0])
    for i in range(4):
        w_i = ws[0][:, i]
        idx_i = idxs[0][:, i]
        for j in range(4):
            w_ij = w_i * ws[1][:, j]
            idx_j = idxs[1][:, j]
            for k in range(4):
                w_ijk = w_ij * ws[2][:, k]
                idx_k = idxs[2][:, k]
                for l in range(4):
                    w_ijkl = w_ijk * ws[3][:, l]
                    idx_l = idxs[3][:, l]
                    res += w_ijkl * grid[idx_i, idx_j, idx_k, idx_l]
    return res


@njit(cache=True)
def _interpolator_4cubic_numba(grid, w0, i0, w1, i1, w2, i2, w3, i3):
    n = w0.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for t in range(n):
        acc = 0.0
        for a in range(4):
            wa = w0[t, a]
            ia = i0[t, a]
            for b in range(4):
                wab = wa * w1[t, b]
                ib = i1[t, b]
                for c in range(4):
                    wabc = wab * w2[t, c]
                    ic = i2[t, c]
                    for d in range(4):
                        acc += wabc * w3[t, d] * grid[ia, ib, ic, i3[t, d]]
        out[t] = acc
    return out


def read_bolom(filt, iprefix):
    """
    Read the bolometric corrections files for
    a given filter system.

    Parameters:
    -----------

    filt: string
        Filter system/group like UBVRIplus or WISE
    iprefix: string
        Location of the bc correction files
    """
    fs = sorted(glob.glob('%s/*%s' % (iprefix, filt)))
    if len(fs) == 0:
        raise RuntimeError(
            'Filter system %s bolometric correction not found in %s' %
            (filt, iprefix))
    tmpfile = tail_head(fs[0], 5, 10)
    tab0 = atpy.Table().read(tmpfile, format='ascii.fast_commented_header')
    os.unlink(tmpfile)
    tabs = []
    for f in fs:
        curt = atpy.Table().read(f, format='ascii')
        for i, k in enumerate(list(curt.columns)):
            curt.rename_column(k, list(tab0.columns)[i])
        tabs.append(curt)

    tabs = atpy.vstack(tabs)
    return tabs


class BCInterpolator:

    def __init__(self, prefix, filts):
        filts = set(filts)
        vec = np.load(prefix + '/' + POINTS_NPY)
        ndim = 4
        self.ndim = ndim
        uids = [np.unique(vec[i, :], return_inverse=True) for i in range(ndim)]
        self.uvecs = [uids[_][0] for _ in range(ndim)]
        self.uids = [uids[_][1] for _ in range(ndim)]
        size = [len(self.uvecs[_]) for _ in range(ndim)]
        self.filts = filts
        self.dats = {}

        self.box_list = []
        for a in itertools.product(*[[0, 1]] * self.ndim):
            self.box_list.append((a))
        self.box_list = np.array(self.box_list)

        for f in filts:
            curd = np.zeros(size) - np.nan
            curd[tuple(self.uids)] = np.load(prefix + '/' + FILT_NPY % (f, ))
            self.dats[f] = curd

    def __call__(self, p):
        """
        Return bolometric corrections given the stellar parameters
        The input is an array shaped Nx4
        where the 4 dimensions corresponds to
        logteff, logg ,feh, A_V
        and N for the number of stars
        """
        res = {}
        bad = np.zeros(p.shape[0], dtype=bool)
        ws = []
        idxs = []
        for i in range(self.ndim):
            pos = np.searchsorted(self.uvecs[i], p[:, i], 'right') - 1
            bad = bad | (pos < 0) | (pos >= (len(self.uvecs[i]) - 1))
            pos_clipped = np.clip(pos, 0, len(self.uvecs[i]) - 2)
            w, idx = _get_cubic_coeffs(p[:, i], self.uvecs[i], pos_clipped)
            ws.append(w)
            idxs.append(idx)
        for f in self.filts:
            curres = _interpolator_4cubic(self.dats[f], ws, idxs)
            res[f] = curres
            res[f][bad] = np.nan
        return res


def list_filters(path=None):
    """
    Return the list of photometric filters for which the isochrones
    can be constructed
    """
    if path is None:
        path = get_data_path()

    fs = glob.glob(os.path.join(path, FILT_NPY % '*'))
    filts = []
    for f in fs:
        filts.append(
            re.match(FILT_NPY % '(.*)',
                     f.split(os.path.sep)[-1]).group(1))
    return filts


def prepare(iprefix,
            oprefix,
            filters=('SDSSugriz', 'SkyMapper', 'UBVRIplus', 'DECam', 'WISE',
                     'GALEX')):
    cols_ex = ['Teff', 'logg', '[Fe/H]', 'Av', 'Rv']
    last_vec = None
    for i, filt in enumerate(filters):
        tabs = read_bolom(filt, iprefix)
        vec = np.array(
            [np.log10(tabs['Teff']), tabs['logg'], tabs['[Fe/H]'], tabs['Av']])
        if last_vec is not None and (last_vec != vec).sum() > 0:
            raise Exception("shouldn't happen")
        last_vec = vec.copy()
        if i == 0:
            np.save(os.path.join(oprefix, POINTS_NPY), vec)
        for k in tabs.columns:
            if k not in cols_ex:
                np.save(os.path.join(oprefix, FILT_NPY % k), tabs[k])
