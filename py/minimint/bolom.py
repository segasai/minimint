import glob
import itertools
import re
import os
import tempfile
import astropy.table as atpy
import numpy as np
from .utils import (get_data_path, get_data_path_for_grid, tail_head,
                    _get_cubic_coeffs, _interpolator_4d, _interpolator_5d)

POINTS_NPY = 'bolom_points.npy'
FILT_NPY = 'filt_%s.npy'


def _interpolator_4cubic(grid, ws, idxs):
    """
    Perform 4D cubic interpolation for bolometric corrections.
    """
    return _interpolator_4d(grid, ws[0], idxs[0], ws[1], idxs[1], ws[2],
                            idxs[2], ws[3], idxs[3])


def _interpolator_5cubic(grid, ws, idxs):
    """
    Perform 5D cubic interpolation for bolometric corrections.
    """
    return _interpolator_5d(grid, ws[0], idxs[0], ws[1], idxs[1], ws[2],
                            idxs[2], ws[3], idxs[3], ws[4], idxs[4])


def _header_preview_file(fin, nout=10):
    """
    Create a temporary file starting from the detected header line.
    Supports both v1.2 and v2.5 BC table formats.
    """
    fp = open(fin, 'r')
    lines = []
    for i, ll in enumerate(fp):
        lines.append(ll)
        if i > 200:
            break
    fp.close()

    header_idx = None
    for i, ll in enumerate(lines):
        lls = ll.strip()
        if not lls.startswith('#'):
            continue
        if ('Teff' in lls) or ('lgTef' in lls) or ('logT' in lls):
            header_idx = i
            break
    if header_idx is None:
        return tail_head(fin, 5, nout)

    fpout = tempfile.NamedTemporaryFile(delete=False, mode='w')
    for ll in lines[header_idx:header_idx + nout + 1]:
        print(ll, file=fpout, end='')
    fpout.close()
    return fpout.name


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
    tmpfile = _header_preview_file(fs[0], 10)
    tab0 = atpy.Table().read(tmpfile, format='ascii.fast_commented_header')
    os.unlink(tmpfile)
    tabs = []
    colnames = list(tab0.columns)
    for f in fs:
        curt = atpy.Table().read(f,
                                 format='ascii.basic',
                                 names=colnames,
                                 comment='#')
        tabs.append(curt)

    tabs = atpy.vstack(tabs)
    return tabs


class BCInterpolator:

    def __init__(self, prefix, filts):
        """
        Initialize bolometric-correction interpolation grids.

        Parameters
        ----------
        prefix: str
            Directory containing `bolom_points.npy` and `filt_*.npy` files.
        filts: iterable of str
            Filter names to load from `filt_<name>.npy`.
        """
        filts = set(filts)
        vec = np.load(prefix + '/' + POINTS_NPY)
        ndim = vec.shape[0]
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
            p_dim = p[:, i]
            if i == 2:
                p_dim = np.maximum(p_dim, self.uvecs[i][0])
            pos = np.searchsorted(self.uvecs[i], p_dim, 'right') - 1
            bad = bad | (pos < 0) | (pos >= (len(self.uvecs[i]) - 1))
            pos_clipped = np.clip(pos, 0, len(self.uvecs[i]) - 2)
            w, idx = _get_cubic_coeffs(p_dim, self.uvecs[i], pos_clipped)
            ws.append(w)
            idxs.append(idx)
        for f in self.filts:
            if self.ndim == 4:
                curres = _interpolator_4cubic(self.dats[f], ws, idxs)
            elif self.ndim == 5:
                curres = _interpolator_5cubic(self.dats[f], ws, idxs)
            else:
                raise RuntimeError(f'Unsupported BC dimensionality: {self.ndim}')
            res[f] = curres
            res[f][bad] = np.nan
        return res


def list_filters(path=None, mist_version='1.2', vvcrit=0.4):
    """
    Return filter names available in prepared bolometric-correction data.

    Parameters
    ----------
    path: str or None
        Directory to scan. If None, resolve from `mist_version` and `vvcrit`.
    mist_version: str
        MIST version used when resolving the default path.
    vvcrit: float
        Rotation value used when resolving the default path.
    """
    if path is None:
        path = get_data_path_for_grid(mist_version=mist_version,
                                      vvcrit=vvcrit,
                                      create=False)
        if not os.path.isdir(path):
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
    """
    Read bolometric-correction tables and save compact `.npy` grids.

    Parameters
    ----------
    iprefix: str
        Input directory containing raw BC table files.
    oprefix: str
        Output directory where `bolom_points.npy` and `filt_*.npy` are saved.
    filters: iterable of str
        Filter-system groups to read from the input directory.
    """
    cols_ex = ['Teff', 'logg', '[Fe/H]', 'Av', 'Rv', 'lgTef', 'Fe_H', 'a_Fe']
    last_vec = None
    for i, filt in enumerate(filters):
        tabs = read_bolom(filt, iprefix)
        if 'Teff' in tabs.colnames:
            vec = np.array(
                [np.log10(tabs['Teff']), tabs['logg'], tabs['[Fe/H]'],
                 tabs['Av']])
        elif 'lgTef' in tabs.colnames:
            vec = np.array(
                [tabs['lgTef'], tabs['logg'], tabs['Fe_H'], tabs['a_Fe'],
                 tabs['Av']])
        else:
            raise RuntimeError('Unrecognized BC table columns')
        if last_vec is not None and (last_vec != vec).sum() > 0:
            raise Exception("shouldn't happen")
        last_vec = vec.copy()
        if i == 0:
            np.save(os.path.join(oprefix, POINTS_NPY), vec)
        for k in tabs.columns:
            if k not in cols_ex:
                np.save(os.path.join(oprefix, FILT_NPY % k), tabs[k])
