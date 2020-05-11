import glob
import scipy.interpolate
import scipy.spatial
import os
import astropy.table as atpy
import numpy as np
POINTS_NPY = 'bolom_points.npy'
FILT_NPY = 'filt_%s.npy'


def read_bolom(filt, iprefix):
    fs = sorted(glob.glob('%s/*%s' % (iprefix, filt)))
    assert (len(fs) > 0)
    cmd = 'tail -n +6 %s |head   > /tmp/xx.tmp ' % (fs[0], )
    print(cmd)
    os.system(cmd)

    tab0 = atpy.Table().read('/tmp/xx.tmp',
                             format='ascii.fast_commented_header')

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
        vec = np.load(prefix + '/' + POINTS_NPY)
        ndim = 4
        self.ndim = ndim
        uids = [np.unique(vec[i, :], return_inverse=True) for i in range(ndim)]
        self.uvecs = [uids[_][0] for _ in range(ndim)]
        self.uids = [uids[_][1] for _ in range(ndim)]
        size = [len(self.uvecs[_]) for _ in range(ndim)]
        dats = {}
        self.filts = filts
        self.interps = {}
        for f in filts:
            curd = np.zeros(size) - np.nan
            curd[tuple(self.uids)] = np.load(prefix + '/' + FILT_NPY % (f, ))
            dats[f] = curd
            self.interps[f] = scipy.interpolate.RegularGridInterpolator(
                self.uvecs, dats[f], method='linear', bounds_error=False)

    def __call__(self, p):
        ## assert arguments is np.log10(tabs['Teff']), tabs['logg'], tabs['[Fe/H]'], tabs['Av']])
        ## shaped N,4
        res = {}
        for f in self.filts:
            res[f] = self.interps[f](p)
        return res



def prepare(iprefix,
            oprefix,
            filters=('SDSSugriz', 'SkyMapper', 'UBVRIplus', 'DECam','WISE')):
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
