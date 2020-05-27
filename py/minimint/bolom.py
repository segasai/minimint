import glob
import itertools
import re
import os
import scipy.interpolate
import scipy.spatial
import astropy.table as atpy
import numpy as np
from .utils import get_data_path, tail_head

POINTS_NPY = 'bolom_points.npy'
FILT_NPY = 'filt_%s.npy'

def read_bolom(filt, iprefix):
    fs = sorted(glob.glob('%s/*%s' % (iprefix, filt)))
    if len(fs) == 0:
        print ('Filter system %s bolometric correction not found in %s'%(filt, iprefix))
        raise RuntimeError('err')
    tmpfile = tail_head(fs[0], 5, 10)
    tab0 = atpy.Table().read(tmpfile,
                             format='ascii.fast_commented_header')
    os.unlink(tmpfile)
    tabs = []
    for f in fs:
        curt = atpy.Table().read(f, format='ascii')
        for i, k in enumerate(list(curt.columns)):
            curt.rename_column(k, list(tab0.columns)[i])
        tabs.append(curt)

    tabs = atpy.vstack(tabs)
    return tabs


## Triangulation based interpolator
class BCInterpolator0:
    def __init__(self, prefix, filts):
        vec = np.load(prefix + '/' + POINTS_NPY)
        if False:
            st = np.random.get_state()
            np.random.seed(1)
            permut = np.random.normal(size=vec.shape) * 1e-5
            vec = vec + permut  # to avoid exact gridding
            np.random.set_state(st)

        tri = scipy.spatial.Delaunay(vec.T, qhull_options="QJ Pp")
        dats = []
        for f in filts:
            dats.append(np.load(prefix + '/' + FILT_NPY % (f, )))
        self.interp = Interpolator0(tri, filts, dats)

    def __call__(self, p):
        return self.interp(p)


class Interpolator0:
    def __init__(self, triang, filts, dats):
        self.triang = triang
        self.dats = {}
        self.filts = filts
        for i, f in enumerate(filts):
            self.dats[f] = dats[i]

    def __call__(self, p):
        p = np.atleast_2d(p)
        ndim = self.triang.ndim
        xid = self.triang.find_simplex(p)
        goods = (xid != -1)
        xid = xid[goods]
        res = {}
        for f in self.filts:
            res[f] = np.zeros(len(p)) - np.nan

        #b = self.triang.transform[xid, :ndim, :].dot(p -
        #                                        self.triang.transform[xid, ndim, :])
        b = np.einsum('ijk,ik->ij', self.triang.transform[xid, :ndim, :],
                      p[goods, :] - self.triang.transform[xid, ndim, :])
        b1 = np.concatenate((b, 1 - b.sum(axis=1)[:, None]), axis=1)
        for f in self.filts:
            res[f][goods] = (self.dats[f][self.triang.simplices[xid]] *
                             b1).sum(axis=1)

        return res


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
        dats = {}
        self.filts = filts
        self.dats = {}
        for f in filts:
            curd = np.zeros(size) - np.nan
            curd[tuple(self.uids)] = np.load(prefix + '/' + FILT_NPY % (f, ))
            self.dats[f] = curd
            #self.interps[f] = scipy.interpolate.RegularGridInterpolator(
            #    self.uvecs, dats[f], method='linear', bounds_error=False)

    def __call__(self, p):
        ## assert arguments is np.log10(tabs['Teff']), tabs['logg'], tabs['[Fe/H]'], tabs['Av']])
        ## shaped N,4
        res = {}
        pos1 = np.zeros(p.shape, dtype=int)
        xs = np.zeros(p.shape)
        bad = np.zeros(p.shape[0], dtype=bool)
        for i in range(self.ndim):
            pos1[:, i] = np.searchsorted(self.uvecs[i], p[:, i],'right') - 1
            bad = bad | (pos1[:, i] < 0) | (pos1[:, i] >=
                                            (len(self.uvecs[i]) - 1))
            pos1[:, i][bad] = 0
            xs[:, i] = (p[:, i] - self.uvecs[i][pos1[:, i]]) / (
                self.uvecs[i][pos1[:, i] + 1] - self.uvecs[i][pos1[:, i]]
            )  # from 0 to 1

        curinds = []
        curcoeffs = []
        for a in itertools.product(*[[0, 1]] * self.ndim):
            a = np.array(a)
            curinds.append(
                tuple([(pos1[:, i] + a[i]) for i in range(self.ndim)]))
            curcoeffs.append(
                (xs**a[None, :] * (1 - xs)**(1 - a[None, :])).prod(axis=1))

        for f in self.filts:
            curres = np.zeros(p.shape[0])
            for curi, curc in zip(curinds, curcoeffs):
                curres[:] = curres + self.dats[f][curi] * curc
            res[f] = curres
            res[f][bad] = np.nan
        return res

def list_filters(path=None):
    if path is None:
        path = get_data_path()
    
    fs = glob.glob(path+'/'+FILT_NPY%'*')
    filts = []
    for  f in fs:
        filts.append(re.match(FILT_NPY%'(.*)', f.split('/')[-1]).group(1))
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
