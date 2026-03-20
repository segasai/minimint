import tempfile
import warnings
import glob
import os
import gc
import subprocess
import urllib.request
import astropy.table as atpy
import scipy.interpolate

import numpy as np
from minimint import bolom, utils
"""
Here we are often relying on bilinear interpolation
If the values at grid points  are
X_k, Y_k -> V_11
X_k+1, Y_k -> V_21
X_k, Y_k+1 -> V_12
X_k+1, Y_k+1 -> V_22

Then the value within the cube can be written as

V_11 * ( 1-x) *(1-y) + V_22 * xy +
V_21 * x * (1-y) + V_12 * (1-x)* y

where x,y are normalized coordinates
x = (X-X_k)/(X_{k+1}-X_k)
y = (Y-Y_k)/(Y_{k+1}-Y_k)

Typically throughout the code the metallicity is the first axis
and mass is the second axis
"""

def get_file(gridt):
    return '%s_grid.npy' % (gridt)


INTERP_NPZ = 'interp.npz'
VALID_EEP_MAX_NPY = 'valid_eep_max.npy'


def get_interp_ready_file(gridt):
    return f'{gridt}_interp_grid.npy'


def _require_supported_mist_version(mist_version):
    mist_version = utils.normalize_mist_version(mist_version)
    if mist_version != '1.2':
        raise ValueError(
            f'Only MIST v1.2 is supported in this release, got: {mist_version}'
        )
    return mist_version


def getheader(f):
    fp = open(f, 'r')
    for i in range(5):
        ll = fp.readline()
    vals = ll[1:].split()
    # Yinit        Zinit   [Fe/H]   [a/Fe]  v/vcrit

    D = {'feh': float(vals[2]), 'afe': float(vals[3])}
    for i in range(3):
        ll = fp.readline()
    #     initial_mass   N_pts   N_EEP   N_col   phase        type\n'
    # 3.8000000000E+00    1409       9      77     YES    low-mass\n'
    vals = ll[1:].split()
    D['initial_mass'] = float(vals[0])
    D['N_pts'] = int(vals[1])
    D['N_EEP'] = int(vals[2])
    D['type'] = (vals[5])
    return D


def read_grid(eep_prefix):
    mask = os.path.join(eep_prefix, '*EEPS', '*eep')
    fs = glob.glob(mask)
    if len(fs) == 0:
        raise RuntimeError(f'Failed to find eep files {mask}')
    tmpfile = utils.tail_head(fs[0], 11, 10)
    tab0 = atpy.Table().read(tmpfile, format='ascii.fast_commented_header')
    os.unlink(tmpfile)
    tabs0 = []
    N = len(fs)
    for i, f in enumerate(fs):
        if i % (N // 100) == 0:
            print('%d/%d' % (i, N))
        curt = atpy.Table().read(f, format='ascii.fast_no_header')
        for i, k in enumerate(list(curt.columns)):
            curt.rename_column(k, list(tab0.columns)[i])
        D = getheader(f)
        curt['initial_mass'] = D['initial_mass']
        curt['feh'] = D['feh']
        curt['EEP'] = np.arange(len(curt))
        tabs0.append(curt)

    tabs = atpy.vstack(tabs0)
    if 'comments' in tabs.meta:
        del tabs.meta['comments']
    del tabs0
    gc.collect()
    for k in list(tabs.columns):
        if k not in [
                'star_age', 'star_mass', 'log_L', 'log_g', 'log_Teff',
                'initial_mass', 'phase', 'feh', 'EEP'
        ]:
            tabs.remove_column(k)

    return tabs


def grid3d_filler(ima):
    """
    This fills nan gaps along one dimension in a 3d cube.
    I fill the gaps along mass dimension
    The array is modified
    """
    nx, ny, nz = ima.shape
    for i in range(nx):
        for k in range(nz):
            grid1d_filler(ima[i, :, k])


def grid1d_filler(arr):
    """
    This takes a vector with gaps filled with nans.
    It then fills the internal gaps with linear interpolation
    Input is modified
    """
    xids = np.nonzero(np.isfinite(arr))[0]
    if len(xids) == 0:
        return
    left, right = xids[0], xids[-1]
    xids1 = np.arange(left, right + 1)
    mask = ~np.isfinite(arr[xids1])
    if mask.any():
        arr[xids1[mask]] = scipy.interpolate.UnivariateSpline(xids,
                                                        arr[xids],
                                                        s=0,
                                                        k=1)(xids1[mask])


def build_interp_ready_grid(grid):
    """
    Prepare a finite grid for cubic interpolation.
    """
    grid_filled = np.array(grid, copy=True)
    for i in range(grid_filled.shape[0]):
        for j in range(grid_filled.shape[2]):
            arr = grid_filled[i, :, j]
            xids = np.nonzero(np.isfinite(arr))[0]
            if len(xids) > 0:
                arr[:xids[0]] = arr[xids[0]]
                arr[xids[-1] + 1:] = arr[xids[-1]]
            else:
                arr[:] = 0
    for i in range(grid_filled.shape[1]):
        for j in range(grid_filled.shape[2]):
            arr = grid_filled[:, i, j]
            xids = np.nonzero(np.isfinite(arr))[0]
            if len(xids) > 0:
                grid1d_filler(arr)
                arr[:xids[0]] = arr[xids[0]]
                arr[xids[-1] + 1:] = arr[xids[-1]]
            else:
                arr[:] = 0
    return grid_filled


def __bc_url(x):
    return 'https://waps.cfa.harvard.edu/MIST/BC_tables/%s.txz' % x


def __eep_url(x, vvcrit=0.4):
    return ('https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/' +
            'MIST_v1.2_feh_%s_afe_p0.0_vvcrit%.1f_EEPS.txz') % (x, vvcrit)


def download_and_prepare(filters=[
    'DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper', 'UBVRIplus',
    'WISE'
],
                         outp_prefix=None,
                         tmp_prefix=None,
                         vvcrit=0.4,
                         mist_version='1.2'):
    """ Download the MIST isochrones and prepare the prerocessed isochrones
    Parameters

    ----------
    filters: tuple
        List of filter systems ['DECam','GALEX',...']
    outp_prefix: string (optional)
        Output directory for processed files
    tmp_prefix: string (optional)
        Temporary directory for storing downloaded files
    vvcrit: float
        The value of V/Vcrit for the isochrones. The default value is 0.4, but
        you can also use the value of 0
    mist_version: str
        MIST version. This release supports only v1.2.
    """
    _require_supported_mist_version(mist_version)

    mets = ('m4.00,m3.50,m3.00,m2.50,m2.00,m1.75,m1.50,m1.25,' +
            'm1.00,m0.75,m0.50,m0.25,p0.00,p0.25,p0.50').split(',')
    if not np.isclose([0., 0.4], vvcrit).any():
        raise ValueError('Only 0 and 0.4 values are allowed')

    def writer(url, pref):
        print('Downloading', url)
        fd = urllib.request.urlopen(url)
        fname = url.split('/')[-1]
        fname_out = os.path.join(pref, fname)
        fdout = open(fname_out, 'wb')
        fdout.write(fd.read())
        fdout.close()
        fd.close()
        if os.name == 'nt':
            fname_out1 = fname_out.replace('.txz', '.tar')
            cmd = (f'cd /d {pref} && '
                   f'7z x {fname_out} && '
                   f'7z x {fname_out1}')
        else:
            cmd = f'cd {pref}; tar xfJ {fname_out}'
        ret = subprocess.run(cmd, shell=True, timeout=60, capture_output=True)
        if ret.returncode != 0:
            raise RuntimeError('Failed to untar the files' +
                               ret.stdout.decode() + ret.stderr.decode())

    with tempfile.TemporaryDirectory(dir=tmp_prefix) as T:
        for curfilt in filters:
            writer(__bc_url(curfilt), T)
        for curmet in mets:
            writer(__eep_url(curmet, vvcrit=vvcrit), T)
        prepare(T,
                T,
                outp_prefix=outp_prefix,
                filters=filters,
                vvcrit=vvcrit,
                mist_version=mist_version)


def prepare(eep_prefix,
            bolom_prefix,
            outp_prefix=None,
            filters=('DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper',
                     'UBVRIplus', 'WISE'),
            vvcrit=0.4,
            mist_version='1.2'):
    """
    Prepare the isochrone files

    Parameters
    ----------
    eep_prefix: string
        The path that has *EEP folders where *eep files will be searched
    bolom_prefix: string
        The path that has bolometric correction files *DECam *UBRI etc
    """
    mist_version = _require_supported_mist_version(mist_version)
    if outp_prefix is None:
        outp_prefix = utils.get_data_path_for_grid(mist_version=mist_version,
                                                   vvcrit=vvcrit)
    else:
        os.makedirs(outp_prefix, exist_ok=True)
    print('Reading EEP grid')
    if not os.path.isdir(eep_prefix) or not os.path.isdir(outp_prefix):
        raise RuntimeError(
            'The arguments must be paths to the directories with EEP \
            and bolometric corrections')
    tab = read_grid(eep_prefix)
    print('Processing EEPs')

    umass, mass_id = np.unique(np.array(tab['initial_mass']),
                               return_inverse=True)
    ufeh, feh_id = np.unique(np.array(tab['feh']), return_inverse=True)

    neep = int(np.max(np.asarray(tab['EEP'], dtype=np.int64))) + 1
    nfeh = len(ufeh)
    nmass = len(umass)
    grids = ['logage', 'logteff', 'logg', 'logl', 'phase']
    grid_store = {}
    for k in grids:
        grid = np.zeros((nfeh, nmass, neep)) - np.nan
        if k == 'logage':
            grid[feh_id, mass_id, tab['EEP']] = np.log10(tab['star_age'])
            grid[:, :, 1:] = np.diff(grid, axis=2)
        elif k == 'logteff':
            grid[feh_id, mass_id, tab['EEP']] = tab['log_Teff']
        elif k == 'logg':
            grid[feh_id, mass_id, tab['EEP']] = tab['log_g']
        elif k == 'logl':
            grid[feh_id, mass_id, tab['EEP']] = tab['log_L']
        elif k == 'phase':
            grid[feh_id, mass_id, tab['EEP']] = tab['phase']
        else:
            raise Exception('wrong ' + k)

        grid3d_filler(grid)

        if k == 'phase':
            grid[~np.isfinite(grid)] = -99
            grid = np.round(grid).astype(np.int8)
        if k == 'logage':
            grid[:, :, :] = np.cumsum(grid, axis=2)

        grid_store[k] = grid
        np.save(os.path.join(outp_prefix, get_file(k)), grid)

    valid_eep_max = np.max(np.where(np.isfinite(grid_store['logage']),
                                    np.arange(neep)[None, None, :], -1),
                           axis=2).astype(np.int16)
    np.save(os.path.join(outp_prefix, VALID_EEP_MAX_NPY), valid_eep_max)
    for k in ('logg', 'logl', 'logteff'):
        np.save(os.path.join(outp_prefix, get_interp_ready_file(k)),
                build_interp_ready_grid(grid_store[k]))
    np.savez(os.path.join(outp_prefix, INTERP_NPZ),
             umass=umass,
             ufeh=ufeh,
             neep=neep,
             mist_version=mist_version,
             vvcrit=np.float64(vvcrit))
    print('Reading/processing bolometric corrections')
    bolom.prepare(bolom_prefix, outp_prefix, filters)


def _binary_search(bads, logage, neep, getAge):
    """
    Peform a binary search on a grid to find pts
    such as getAge(pt)<logage<getAge(pt+1)

    Returns:
    lefts:
    rigths:
    bads:
    """
    # This will be our working subset
    curgood = np.nonzero(~bads)[0]
    # these will be left/right of the binary search
    lefts = np.zeros(len(logage), dtype=int)
    rights = np.zeros(len(logage), dtype=int) + neep - 1  # the last index
    leftX = lefts[curgood]
    rightX = rights[curgood]

    # binary search
    # we are dealing with values that should increase till they become nan
    # we start with LV, RV sitting at the edges
    # at each iteration we propose a midpt to be prop_V (LV+RV)//2
    # Then the options are
    # 1) it's smaller than the target value
    # 2) it's larger than the target value
    # 3) it's nan
    # if 1 we set the new left edge to the proposed point
    # if 2 we set the right edge to the proposed point
    # if 3 we do the same as 2

    # we stop when we find
    # the boundaries to be separated by one and then the options is
    # A we either in the situation of boundaries having
    # 2 finite values or
    # B one finite on the left and the other one nan
    leftY, rightY = [getAge(_, curgood) for _ in [leftX, rightX]]

    while True:
        targY = logage[curgood]

        propX = (leftX + rightX) // 2
        propY = getAge(propX, curgood)

        # It is written in this way to also include nans
        x1 = propY <= targY  # option 1
        x2 = propY > targY  # option 2
        x3 = (~x1) & (~x2)  # option 3
        leftX[x1] = propX[x1]
        rightX[x2] = propX[x2]
        rightX[x3] = propX[x3]
        leftY, rightY = [getAge(_, curgood) for _ in [leftX, rightX]]
        # we stop for either right-left==1 or for bads
        curbad = (targY < leftY) | (targY >= rightY)  # we'll exclude them
        curbad2 = (rightX == leftX + 1) & np.isnan(rightY)  # this is option B
        exclude = (rightX == leftX + 1) | curbad | curbad2
        lefts[curgood[exclude]] = leftX[exclude]
        rights[curgood[exclude]] = rightX[exclude]
        bads[curgood[curbad | curbad2]] = True
        if exclude.all():
            break
        curgood = curgood[~exclude]
        leftX = leftX[~exclude]
        rightX = rightX[~exclude]

    bads = bads | (rights >= neep)
    lefts[bads] = 0
    rights[bads] = 1
    return lefts, rights, bads


def _interpolator(grid, wfeh, ifehs, wmass, imasses, ieep):
    ieep = np.asarray(ieep, dtype=int)
    return utils._interpolator_2d(grid, wfeh, ifehs, wmass, imasses, ieep)


class TheoryInterpolator:

    def __init__(self,
                 prefix=None,
                 spatial_order=1,
                 mist_version='1.2',
                 vvcrit=0.4):
        """
        Construct the interpolator that computes theoretical
        quantities (logg, logl, logteff) given (mass, logage, feh)

        Parameters
        ----------
        prefix: str
            Path to the data folder
        spatial_order: int
            Order of spatial interpolation (1 for linear, 3 for cubic)
        mist_version: str
            MIST version. This release supports only v1.2.
        vvcrit: float
            The value of V/Vcrit used for prepared data selection.
        """
        mist_version = _require_supported_mist_version(mist_version)
        if prefix is None:
            prefix = utils.get_data_path_for_grid(mist_version=mist_version,
                                                  vvcrit=vvcrit,
                                                  create=False)
        (self.logg_grid, self.logl_grid, self.logteff_grid, self.logage_grid,
         self.phase_grid) = [
             np.load(os.path.join(prefix, get_file(curt)))
             for curt in ['logg', 'logl', 'logteff', 'logage', 'phase']
         ]
        meta_path = os.path.join(prefix, INTERP_NPZ)
        if not os.path.exists(meta_path):
            raise RuntimeError(
                f'Metadata file {INTERP_NPZ} not found in {prefix}. '
                'Please re-run minimint.download_and_prepare(...)')
        with np.load(meta_path) as meta:
            self.umass = np.array(meta['umass'])
            self.ufeh = np.array(meta['ufeh'])
            self.neep = int(meta['neep'])
            self.mist_version = str(meta.get('mist_version', mist_version))
            self.vvcrit = float(meta.get('vvcrit', vvcrit))
        valid_eep_path = os.path.join(prefix, VALID_EEP_MAX_NPY)
        if not os.path.exists(valid_eep_path):
            raise RuntimeError(
                f'Validity file {VALID_EEP_MAX_NPY} not found in {prefix}. '
                'Please re-run minimint.download_and_prepare(...)')
        self.valid_eep_max = np.load(valid_eep_path)
            
        if spatial_order not in (1, 3):
            raise ValueError('spatial_order must be 1 (linear) or 3 (cubic)')
        self.spatial_order = spatial_order

    def _eval_linear_interp(self, grid, DD, ieep, subset=None):
        if subset is None:
            subset = slice(None)
        return _interpolator(grid, DD['wfeh_lin'][subset], DD['ifehs_lin'][subset],
                             DD['wmass_lin'][subset], DD['imasses_lin'][subset], ieep)

    def _eval_spatial_interp(self, grid, DD, ieep, subset=None, use_cubic=False):
        if subset is None:
            subset = slice(None)
        if not (use_cubic and self.spatial_order == 3):
            return self._eval_linear_interp(grid, DD, ieep, subset=subset)

        wf = DD['wf'][subset]
        ifehs = DD['ifehs'][subset]
        wm = DD['wm'][subset]
        imasses = DD['imasses'][subset]
        ieep = np.asarray(ieep, dtype=int)

        res = utils._interpolator_bicubic(grid, wf, ifehs, wm, imasses, ieep)
        bad = np.zeros(len(res), dtype=bool)
        for i in range(4):
            for j in range(4):
                bad |= ~np.isfinite(grid[ifehs[:, i], imasses[:, j], ieep])
        if bad.any():
            bad_idx = np.nonzero(bad)[0]
            local = np.arange(len(res))
            res[bad] = self._eval_linear_interp(grid, DD, ieep[bad_idx],
                                                subset=local[bad])
        return res

    def __call__(self, mass, logage, feh):
        """
        Return the theoretical isochrone values such as logL, logg, logteff
        correspoding to the mass, logage and feh
        """
        feh, mass, logage = [
            np.atleast_1d(np.asarray(_)) for _ in [feh, mass, logage]
        ]
        N = len(logage)
        DD = self._get_eep_coeffs(mass, logage, feh)
        eep1, eep2, eep_frac, bad = (DD['eep1'], DD['eep2'], DD['eep_frac'],
                                     DD['bad'])
        good = ~bad
        xret = {}
        if good.any():
            good_idx = np.nonzero(good)[0]
            eep1_good, eep2_good, eep_frac_good = [
                _[good] for _ in [eep1, eep2, eep_frac]
            ]
            eep_m1_good = np.clip(eep1_good - 1, 0, self.neep - 1).astype(float)
            eep_0_good = eep1_good.astype(float)
            eep_1_good = eep2_good.astype(float)
            eep_2_good = np.clip(eep2_good + 1, 0, self.neep - 1).astype(float)

            for curkey, curarr in [('logg', self.logg_grid),
                                   ('logteff', self.logteff_grid),
                                   ('logl', self.logl_grid)]:
                curr = [
                    self._eval_spatial_interp(curarr, DD, eep_m1_good,
                                              subset=good_idx, use_cubic=True),
                    self._eval_spatial_interp(curarr, DD, eep_0_good,
                                              subset=good_idx, use_cubic=True),
                    self._eval_spatial_interp(curarr, DD, eep_1_good,
                                              subset=good_idx, use_cubic=True),
                    self._eval_spatial_interp(curarr, DD, eep_2_good,
                                              subset=good_idx, use_cubic=True)
                ]
                xret[curkey] = utils.steffen_interp(curr[0], curr[1], curr[2],
                                                    curr[3], eep_frac_good)

            # Keep phase interpolation linear to avoid cubic overshoot
            phase0 = self._eval_linear_interp(self.phase_grid, DD,
                                              eep1_good.astype(int),
                                              subset=good_idx)
            phase1 = self._eval_linear_interp(self.phase_grid, DD,
                                              eep2_good.astype(int),
                                              subset=good_idx)
            xret['phase'] = phase0 + eep_frac_good * (phase1 - phase0)

        ret = {}
        for k in ['logg', 'logteff', 'logl', 'phase']:
            ret[k] = np.zeros(N) + np.nan
            if good.any():
                ret[k][good] = xret[k]
        return ret

    def getLogAgeFromEEP(self, mass, eep, feh, returnJac=False):
        """
        This method returns the log(age) for given mass eep and feh

        if returnJac is true the derivative is of d(log(age))/deep is returned
        """
        feh, mass, eep = [
            np.atleast_1d(np.asarray(_, dtype=np.float64))
            for _ in [feh, mass, eep]
        ]
        neep = self.neep
        N = len(feh)
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        bad = np.zeros(N, dtype=bool)
        eep1 = eep.astype(int)
        eep2 = eep1 + 1
        bad = bad | (l2mass >= len(self.umass)) | (l2feh >= len(self.ufeh)) | (
            l1mass < 0) | (l1feh < 0) | (eep2 >= neep) | (eep1 < 0)
        l1mass[bad] = 0
        l2mass[bad] = 1
        l1feh[bad] = 0
        l2feh[bad] = 1
        eep1[bad] = 0
        eep2[bad] = 1
        eep_frac = (eep - eep1)

        goodsel = ~bad

        wfeh_lin, ifehs_lin = utils._get_linear_coeffs(feh, self.ufeh, l1feh)
        wmass_lin, imasses_lin = utils._get_linear_coeffs(mass, self.umass,
                                                          l1mass)
        wf = ifehs = wm = imasses = None
        if self.spatial_order == 3:
            wf, ifehs = utils._get_cubic_coeffs(feh, self.ufeh, l1feh)
            wm, imasses = utils._get_cubic_coeffs(mass, self.umass, l1mass)

        ret_logage = np.zeros_like(mass) + np.nan
        jac = np.zeros_like(mass) + np.nan
        if goodsel.any():
            eep_m1 = np.clip(eep1[goodsel] - 1, 0, neep - 1).astype(float)
            eep_0 = eep1[goodsel].astype(float)
            eep_1 = eep2[goodsel].astype(float)
            eep_2 = np.clip(eep2[goodsel] + 1, 0, neep - 1).astype(float)

            def getAge(cureep_vec):
                if self.spatial_order == 3:
                    ieep = np.asarray(cureep_vec, dtype=int)
                    res = utils._interpolator_bicubic(
                        self.logage_grid, wf[goodsel], ifehs[goodsel],
                        wm[goodsel], imasses[goodsel], ieep)
                    bad = np.zeros(len(res), dtype=bool)
                    for i in range(4):
                        for j in range(4):
                            bad |= ~np.isfinite(
                                self.logage_grid[ifehs[goodsel, i],
                                                 imasses[goodsel, j], ieep])
                    if bad.any():
                        res[bad] = _interpolator(
                            self.logage_grid, wfeh_lin[goodsel][bad],
                            ifehs_lin[goodsel][bad],
                            wmass_lin[goodsel][bad],
                            imasses_lin[goodsel][bad],
                            np.asarray(cureep_vec)[bad])
                    return res
                return _interpolator(self.logage_grid, wfeh_lin[goodsel],
                                     ifehs_lin[goodsel], wmass_lin[goodsel],
                                     imasses_lin[goodsel], cureep_vec)

            logage_m1 = getAge(eep_m1)
            logage_0 = getAge(eep_0)
            logage_1 = getAge(eep_1)
            logage_2 = getAge(eep_2)

            ret_logage[goodsel] = utils.steffen_interp(logage_m1, logage_0, logage_1, logage_2, eep_frac[goodsel])
            jac[goodsel] = logage_1 - logage_0
        
        if returnJac:
            ret = (ret_logage, jac)
        else:
            ret = ret_logage
        return ret

    def getMaxMassMS(self, logage, feh):
        """Find the approximate value of maximum mass on the main sequence """
        N = len(self.umass) - 1
        i1 = 1
        i2 = N - 1
        stop = False
        while not stop:
            ix = (i1 + i2) // 2
            if (i2 - i1) == 1:
                stop = True
            res = self(self.umass[ix], logage, feh)
            phase = res['phase'][0]
            bad = np.isnan(phase)
            if phase > 0.5 or bad:
                i2 = ix
            else:
                i1 = ix
        return self.umass[i1]

    def getMaxMass(self, logage, feh):
        """
        Determine the maximum mass that exists on the current isochrone
        Parameters:
        -----------
        logage: float
            Log10 of age
        feh: float
            Metallicity

        Returns:
        --------
        maxMass: float
            Maximum mass on the isochrone
        """
        logage, feh = np.float64(logage), np.float64(feh)
        niter = 40
        im1 = 0
        im2 = len(self.umass) - 1
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        if self._isvalid(self.umass[im2], logage, feh, l1feh=l1feh):
            return self.umass[im2]
        for _ in range(niter):
            curm = (im1 + im2) // 2
            good = self._isvalid(self.umass[curm], logage, feh, l1feh=l1feh)
            if not good:
                im1, im2 = im1, curm
            else:
                im1, im2 = curm, im2
            if im2 - im1 == 1:
                break

        lo = self.umass[im1]
        hi = self.umass[im2]

        def _isfinite_mass(m):
            return np.isfinite(self(m, logage, feh)['logl'][0])

        if not _isfinite_mass(lo):
            idx = im1
            while idx > 0 and not _isfinite_mass(self.umass[idx]):
                idx -= 1
            lo = self.umass[idx]
            hi = self.umass[min(idx + 1, len(self.umass) - 1)]
        elif _isfinite_mass(hi):
            idx = im2
            while idx + 1 < len(self.umass) and _isfinite_mass(
                    self.umass[idx + 1]):
                idx += 1
            if idx + 1 >= len(self.umass):
                return self.umass[idx]
            lo = self.umass[idx]
            hi = self.umass[idx + 1]

        tol = 1e-7
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if _isfinite_mass(mid):
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol:
                break
        return lo * (1 - 1e-10)

    def _get_eep_coeffs(self, mass, logage, feh):
        """
        This function gets all coefficients for interpolation.
The interpolation is done in two stages:
1) Spatial interpolation over (mass, feh) using axis weights
2) Final interpolation over EEP axis
"""
        feh, mass, logage = [
            np.atleast_1d(np.asarray(_, dtype=np.float64))
            for _ in [feh, mass, logage]
        ]
        N = len(logage)
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        bads = np.zeros(N, dtype=bool)
        bads = bads | (l2mass >= len(self.umass)) | (l2feh >= len(
            self.ufeh)) | (l1mass < 0) | (l1feh < 0)
        l1mass[bads] = 0
        l2mass[bads] = 1
        l1feh[bads] = 0
        l2feh[bads] = 1

        wfeh_lin, ifehs_lin = utils._get_linear_coeffs(feh, self.ufeh, l1feh)
        wmass_lin, imasses_lin = utils._get_linear_coeffs(mass, self.umass,
                                                          l1mass)

        wf = ifehs = wm = imasses = None
        if self.spatial_order == 3:
            wf, ifehs = utils._get_cubic_coeffs(feh, self.ufeh, l1feh)
            wm, imasses = utils._get_cubic_coeffs(mass, self.umass, l1mass)

        def getAge(cureep_vec, subset):
            if np.isscalar(cureep_vec):
                cureep_vec = np.full(len(subset), float(cureep_vec))
            if self.spatial_order == 3:
                ieep = np.asarray(cureep_vec, dtype=int)
                res = utils._interpolator_bicubic(self.logage_grid, wf[subset],
                                                  ifehs[subset], wm[subset],
                                                  imasses[subset], ieep)
                bad = np.zeros(len(res), dtype=bool)
                for i in range(4):
                    for j in range(4):
                        bad |= ~np.isfinite(
                            self.logage_grid[ifehs[subset, i],
                                             imasses[subset, j], ieep])
                if bad.any():
                    res[bad] = _interpolator(
                        self.logage_grid, wfeh_lin[subset][bad],
                        ifehs_lin[subset][bad], wmass_lin[subset][bad],
                        imasses_lin[subset][bad], ieep[bad])
                return res
            return _interpolator(self.logage_grid, wfeh_lin[subset],
                                 ifehs_lin[subset], wmass_lin[subset],
                                 imasses_lin[subset], cureep_vec)

        lefts, rights, bads = _binary_search(bads, logage, self.neep, getAge)
        eep_frac = np.zeros(len(mass))
        
        good = ~bads
        if good.any():
            left_m1 = np.clip(lefts[good] - 1, 0, self.neep - 1).astype(float)
            left_0 = lefts[good].astype(float)
            right_1 = rights[good].astype(float)
            right_2 = np.clip(rights[good] + 1, 0, self.neep - 1).astype(float)

            subset_idx = np.nonzero(good)[0]
            y_m1 = getAge(left_m1, subset_idx)
            y_0 = getAge(left_0, subset_idx)
            y_1 = getAge(right_1, subset_idx)
            y_2 = getAge(right_2, subset_idx)

            eep_frac[good] = utils.solve_steffen_t(y_m1, y_0, y_1, y_2, logage[good])
        
        ret = dict(wfeh_lin=wfeh_lin,
                    ifehs_lin=ifehs_lin,
                    wmass_lin=wmass_lin,
                    imasses_lin=imasses_lin,
                    eep_frac=eep_frac,
                    bad=bads,
                    l1feh=l1feh,
                    l2feh=l2feh,
                    l1mass=l1mass,
                    l2mass=l2mass,
                    eep1=lefts,
                    eep2=rights)
        if self.spatial_order == 3:
            ret.update(dict(wf=wf, ifehs=ifehs, wm=wm, imasses=imasses))
        return ret

    def _isvalid(self, mass, logage, feh, l1feh=None):
        """
        Checks if the point on the isochrone is valid
        """
        mass = np.float64(mass)
        logage = np.float64(logage)
        feh = np.float64(feh)
        if l1feh is None:
            l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1

        if ((l2mass >= len(self.umass)) or (l2feh >= len(self.ufeh))
                or (l1mass < 0) or (l1feh < 0)):
            return False
        wfeh_lin, ifehs_lin = utils._get_linear_coeffs(np.array([feh]),
                                                       self.ufeh,
                                                       np.array([l1feh]))
        wmass_lin, imasses_lin = utils._get_linear_coeffs(np.array([mass]),
                                                          self.umass,
                                                          np.array([l1mass]))
        if self.spatial_order == 3:
            wf, ifehs = utils._get_cubic_coeffs(np.array([feh]), self.ufeh,
                                                np.array([l1feh]))
            wm, imasses = utils._get_cubic_coeffs(np.array([mass]), self.umass,
                                                  np.array([l1mass]))
        i1 = 0
        i2 = int(
            np.min(self.valid_eep_max[[l1feh, l1feh, l2feh, l2feh],
                                      [l1mass, l2mass, l1mass, l2mass]]))
        if i2 < 1:
            return False

        def getAge(cureep):
            ieep = np.array([cureep], dtype=int)
            if self.spatial_order == 3:
                val = utils._interpolator_bicubic(self.logage_grid, wf, ifehs,
                                                  wm, imasses, ieep)[0]
                for i in range(4):
                    for j in range(4):
                        if not np.isfinite(
                                self.logage_grid[ifehs[0, i], imasses[0, j],
                                                 ieep[0]]):
                            return _interpolator(
                                self.logage_grid, wfeh_lin, ifehs_lin,
                                wmass_lin, imasses_lin, ieep)[0]
                return val
            return _interpolator(self.logage_grid, wfeh_lin, ifehs_lin,
                                 wmass_lin, imasses_lin, ieep)[0]

        # check invariants on edges
        if not getAge(i1) <= logage:
            return False
        if (getAge(i2) <= logage):
            return False
        stop = False
        while not stop:
            ix = (i1 + i2) // 2
            if i2 - i1 == 1:
                stop = True
            val = getAge(ix)
            if val <= logage:
                i1 = ix
            elif val > logage:
                return True
            else:
                # nan
                i2 = ix
        if np.isnan(getAge(i2)):
            return False
        return True

    def _getMaxMassBox(self, logage, feh, l1feh, l2feh, l1mass, l2mass):
        # here we are trying to find linear solutions
        # inside each EEP,mass,feh box to match our age

        x = (feh - self.ufeh[l1feh]) / (self.ufeh[l2feh] - self.ufeh[l1feh])
        # from 0 to 1

        V11 = self.logage_grid[l1feh, l1mass, :]
        V12 = self.logage_grid[l1feh, l2mass, :]
        V21 = self.logage_grid[l2feh, l1mass, :]
        V22 = self.logage_grid[l2feh, l2mass, :]
        with warnings.catch_warnings():
            # protect against warnings here because we
            # are actively searching for valid range
            warnings.simplefilter("ignore")
            if x == 0:
                yy = (logage - V11) / (V12 - V11)
            elif x == 1:
                yy = (logage - V21) / (V22 - V21)
            else:
                yy = (logage - V11 * (1 - x) - V21 * x) / ((V12 - V11) * (1 - x) + (V22 - V21) * x)
                
        yy = yy[np.isfinite(yy) & (yy <= 1) & (yy >= 0)]
        if len(yy) > 0:
            return self.umass[l1mass] + np.nanmax(
                (self.umass[l2mass] - self.umass[l1mass]) * yy)
        else:
            # this likely will happen if only *exactly* the edge
            # works
            return np.nan


class Interpolator:

    def __init__(self,
                 filts,
                 data_prefix=None,
                 spatial_order=1,
                 mist_version='1.2',
                 vvcrit=0.4):
        """
        Initialize the interpolator class, specifying filter names
        and optionally the folder where the preprocessed isochrones lie

        Parameters
        ----------
        filts: list
            List of strings, such as ['DECam_g','WISE_W1']
        data_prefix: str
            String for the data
        spatial_order: int
            Order of spatial interpolation (1 for linear, 3 for cubic)
        mist_version: str
            MIST version. This release supports only v1.2.
        vvcrit: float
            The value of V/Vcrit used for prepared data selection.
        """
        mist_version = _require_supported_mist_version(mist_version)
        if data_prefix is None:
            data_prefix = utils.get_data_path_for_grid(
                mist_version=mist_version, vvcrit=vvcrit, create=False)
        self.isoInt = TheoryInterpolator(data_prefix,
                                         spatial_order=spatial_order,
                                         mist_version=mist_version,
                                         vvcrit=vvcrit)
        self.bolomInt = bolom.BCInterpolator(data_prefix, filts)

    def __call__(self, mass, logage, feh):
        """
        Compute interpolated isochrone for a given mass log10(age) and feh

        Parameters
        ----------
        mass: float/numpy
            Either scalar or vector of masses
        logage: float/numpy
            Either scalar or vector of log10(age)
        feh: float/numpy
            Either scalar or vector of [Fe/H]

        """
        mass, logage, feh = [
            np.asarray(_, dtype=np.float64) for _ in [mass, logage, feh]
        ]
        mass, logage, feh = np.broadcast_arrays(mass, logage, feh)
        shape = mass.shape
        mass, logage, feh = [np.atleast_1d(_) for _ in [mass, logage, feh]]

        ret = self.isoInt(mass, logage, feh)
        good_sub = np.isfinite(ret['logl'])

        av = ret['logl'][good_sub] * 0
        # computing when no extinction
        arr = np.array([
            ret['logteff'][good_sub], ret['logg'][good_sub], feh[good_sub], av
        ]).T
        res0 = self.bolomInt(arr)
        ret['logage'] = logage
        ret['feh'] = feh
        ret['mass'] = mass
        MBolSun = 4.74
        for k in res0:
            ret[k] = np.zeros(len(mass)) - np.nan
            ret[k][good_sub] = MBolSun - 2.5 * ret['logl'][good_sub] - res0[k]
        for k in ret.keys():
            ret[k] = ret[k].reshape(shape)
        return ret

    def getMaxMass(self, logage, feh):
        """ Return the maximum mass on a given isochrone """
        return self.isoInt.getMaxMass(logage, feh)

    def getMaxMassMS(self, logage, feh):
        return self.isoInt.getMaxMassMS(logage, feh)
