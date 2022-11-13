import glob
import itertools
import re
import os
import astropy.table as atpy
import numpy as np
from .utils import get_data_path, tail_head

POINTS_NPY = 'bolom_points.npy'
FILT_NPY = 'filt_%s.npy'


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

        # These are the box coordinates for interpolation
        # 0,0 1,0, 0,1 1,1
        self.box_list = []
        for a in itertools.product(*[[0, 1]] * self.ndim):
            self.box_list.append((a))
        self.box_list = np.array(self.box_list)
        for f in filts:
            curd = np.zeros(size) - np.nan
            curd[tuple(self.uids)] = np.load(prefix + '/' + FILT_NPY % (f, ))
            self.dats[f] = curd
            # self.interps[f] = scipy.interpolate.RegularGridInterpolator(
            #    self.uvecs, dats[f], method='linear', bounds_error=False)

    def __call__(self, p):
        """
        Return bolometric corrections given the stellar parameters
        The input is an array shaped Nx4
        where the 4 dimensions corresponds to
        logteff, log ,feh, A_V
        and N for the number of stars
        """
        # assert arguments are shaped N,4
        res = {}
        pos1 = np.zeros(p.shape, dtype=int)
        xs = np.zeros(p.shape)
        bad = np.zeros(p.shape[0], dtype=bool)
        for i in range(self.ndim):
            pos1[:, i] = np.searchsorted(self.uvecs[i], p[:, i], 'right') - 1
            bad = bad | (pos1[:, i] < 0) | (pos1[:, i] >=
                                            (len(self.uvecs[i]) - 1))
            pos1[:, i][bad] = 0
            xs[:, i] = (p[:, i] - self.uvecs[i][pos1[:, i]]) / (
                self.uvecs[i][pos1[:, i] + 1] - self.uvecs[i][pos1[:, i]]
            )  # from 0 to 1

        curinds = []
        curcoeffs = []
        curinds = pos1[None, :, :] + self.box_list[:, None, :]
        # this is fancy math for the poly linear interpolation
        # where the value = sum F_j * x**a_j * (1-x)**(1-a_j)
        # where F_j is the value in the cube vertex
        # x is the n-dim vector where ieach axis goes from 0 to 1
        # and a_j are the 0,1 vectors corresponding to the vertices
        curcoeffs = (
            xs[None, :, :]**self.box_list[:, None, :] *
            (1 - xs[None, :, :])**(1 - self.box_list)[:, None, :]).prod(axis=2)
        curinds1 = np.ravel_multi_index(curinds.T,
                                        self.dats[list(self.filts)[0]].shape)
        for f in self.filts:
            curres = (self.dats[f].flat[curinds1] * curcoeffs.T).sum(axis=1)
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

    fs = glob.glob(path + '/' + FILT_NPY % '*')
    filts = []
    for f in fs:
        filts.append(re.match(FILT_NPY % '(.*)', f.split('/')[-1]).group(1))
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
            np.save(oprefix + '/' + POINTS_NPY, vec)
        for k in tabs.columns:
            if k not in cols_ex:
                np.save(oprefix + '/' + FILT_NPY % (k), tabs[k])
