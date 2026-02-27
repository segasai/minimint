import glob
import itertools
import re
import os
import tempfile
import astropy.table as atpy
import numpy as np
import warnings
from .utils import get_data_path, tail_head, _get_cubic_coeffs

POINTS_NPY = 'bolom_points.npy'
FILT_NPY = 'filt_%s.npy'


def _interpolator_4cubic(grid, ws, idxs):
    """
    Perform 4D cubic interpolation for bolometric corrections.
    The dimensions are usually logTeff, logg, [Fe/H], and Av.
    
    grid: 4D array of grid values
    ws: list of 4 weight arrays, each of shape (N, 4)
    idxs: list of 4 index arrays, each of shape (N, 4)
    """
    # 4D tensor-product cubic:
    # f=Σ_i Σ_j Σ_k Σ_l w0_i w1_j w2_k w3_l f(i,j,k,l)
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


def _interpolator_5cubic(grid, ws, idxs):
    """
    Perform 5D cubic interpolation for bolometric corrections.
    The dimensions are usually logTeff, logg, [Fe/H], [alpha/Fe], and Av.
    """
    # 5D tensor-product cubic:
    # f=Σ_i Σ_j Σ_k Σ_l Σ_m Π_d w_d * f(vertex)
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
                    for m in range(4):
                        res += (w_ijkl * ws[4][:, m] *
                                grid[idx_i, idx_j, idx_k, idx_l, idxs[4][:,
                                                                         m]])
    return res


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
        # fallback to legacy behavior
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

    def __init__(self, prefix, filts, linear=False):
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

        # These are the box coordinates for interpolation
        # 0,0 1,0, 0,1 1,1
        self.box_list = []
        for a in itertools.product(*[[0, 1]] * self.ndim):
            self.box_list.append((a))
        self.box_list = np.array(self.box_list)
        self.linear = bool(linear)
        self._warned_feh_floor = False
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
        logteff, logg, feh, A_V
        and N for the number of stars
        """
        # assert arguments are shaped N,4
        res = {}
        bad = np.zeros(p.shape[0], dtype=bool)
        ws = []
        idxs = []
        pos1 = np.zeros(p.shape, dtype=int)
        xs = np.zeros(p.shape)
        for i in range(self.ndim):
            p_dim = p[:, i]
            # For BC interpolation, clamp very metal-poor values to the BC
            # grid floor (v2.5 BC tables currently start at [Fe/H]=-3.0).
            if i == 2:
                feh_floor = self.uvecs[i][0]
                clipped = np.maximum(p_dim, feh_floor)
                if np.any(clipped != p_dim) and not self._warned_feh_floor:
                    warnings.warn(
                        f'Clipping [Fe/H] below BC grid floor ({feh_floor}) '
                        'to enable BC interpolation.')
                    self._warned_feh_floor = True
                p_dim = clipped
            pos = np.searchsorted(self.uvecs[i], p_dim, 'right') - 1
            bad = bad | (pos < 0) | (pos >= (len(self.uvecs[i]) - 1))
            pos1[:, i] = np.clip(pos, 0, len(self.uvecs[i]) - 2)
            xs[:, i] = (p_dim - self.uvecs[i][pos1[:, i]]) / (
                self.uvecs[i][pos1[:, i] + 1] - self.uvecs[i][pos1[:, i]])
            if not self.linear:
                w, idx = _get_cubic_coeffs(p_dim, self.uvecs[i], pos1[:, i])
                ws.append(w)
                idxs.append(idx)

        for f in self.filts:
            if self.linear:
                curinds = pos1[None, :, :] + self.box_list[:, None, :]
                # N-D poly-linear interpolation over BC grid:
                # BC(p) = Σ_v [Π_d x_d^a_{v,d}(1-x_d)^(1-a_{v,d})] * BC(vertex_v)
                curcoeffs = (
                    xs[None, :, :]**self.box_list[:, None, :] *
                    (1 - xs[None, :, :])**(1 - self.box_list)[:, None, :]
                ).prod(axis=2)
                curinds1 = np.ravel_multi_index(curinds.T, self.dats[f].shape)
                curres = (self.dats[f].flat[curinds1] * curcoeffs.T).sum(axis=1)
            elif self.ndim == 4:
                curres = _interpolator_4cubic(self.dats[f], ws, idxs)
            elif self.ndim == 5:
                curres = _interpolator_5cubic(self.dats[f], ws, idxs)
            else:
                raise RuntimeError(f'Unsupported BC dimensionality: {self.ndim}')
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
