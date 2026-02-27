import tempfile
import warnings
import glob
import os
import gc
import itertools
import subprocess
import pickle
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

TRACKS_FILE = 'tracks.fits'


def get_file(gridt):
    return '%s_grid.npy' % (gridt)


INTERP_PKL = 'interp.pkl'
INTERP_NPZ = 'interp.npz'


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


def _normalize_grid_version(grid_version):
    if grid_version is None:
        return "1.2"
    return str(grid_version).lstrip('v')


def read_grid(eep_prefix, outp_prefix):
    mask = os.path.join(eep_prefix, '*EEPS', '*eep')
    fs = glob.glob(mask)
    if len(fs) == 0:
        # v2.5 tarballs may unpack into a different directory layout
        mask = os.path.join(eep_prefix, '**', '*eep')
        fs = glob.glob(mask, recursive=True)
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
        curt['afe'] = D['afe']
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
                'initial_mass', 'phase', 'feh', 'afe', 'EEP'
        ]:
            tabs.remove_column(k)

    os.makedirs(outp_prefix, exist_ok=True)
    tabs.write(os.path.join(outp_prefix, TRACKS_FILE), overwrite=True)


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


def grid4d_filler(ima):
    """
    This fills nan gaps along the mass dimension in a 4D cube.
    The array is modified in-place.
    """
    nfeh, nafe, nmass, neep = ima.shape
    for i in range(nfeh):
        for j in range(nafe):
            for k in range(neep):
                grid1d_filler(ima[i, j, :, k])


def grid1d_filler(arr):
    """
    This takes a vector with gaps filled with nans.
    It then fills the internal gaps with linear interpolation
    Input is modified
    """
    xids = np.nonzero(np.isfinite(arr))[0]
    left, right = xids[0], xids[-1]
    xids1 = np.arange(left, right + 1)
    xids1 = xids1[~np.isfinite(arr[xids1])]
    if len(xids1) > 0:
        arr[xids1] = scipy.interpolate.UnivariateSpline(xids,
                                                        arr[xids],
                                                        s=0,
                                                        k=1)(xids1)


def __bc_url_v12(x):
    return 'https://waps.cfa.harvard.edu/MIST/BC_tables/%s.txz' % x


def __bc_url_v25(x):
    return 'https://mist.science/BC_tables/v2/%s.txz' % x


def _format_feh_v12(feh):
    sign = 'm' if feh < 0 else 'p'
    return f"{sign}{abs(feh):.2f}"


def _format_feh_v25(feh):
    sign = 'm' if feh < 0 else 'p'
    val = int(round(abs(feh) * 100))
    return f"{sign}{val:03d}"


def _format_afe_v25(afe):
    sign = 'm' if afe < 0 else 'p'
    val = int(round(abs(afe) * 10))
    return f"{sign}{val:d}"


def __eep_url_v12(feh, vvcrit=0.4):
    feh_tag = _format_feh_v12(feh)
    return ('https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/' +
            'MIST_v1.2_feh_%s_afe_p0.0_vvcrit%.1f_EEPS.txz') % (feh_tag,
                                                              vvcrit)


def __eep_url_v25(feh, afe, vvcrit=0.4):
    feh_tag = _format_feh_v25(feh)
    afe_tag = _format_afe_v25(afe)
    return ('https://mist.science/data/tarballs_v2.5/eeps/' +
            'MIST_v2.5_feh_%s_afe_%s_vvcrit%.1f_EEPS.txz') % (feh_tag,
                                                            afe_tag, vvcrit)


def download_and_prepare(filters=[
    'DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper', 'UBVRIplus',
    'WISE'
],
                         outp_prefix=None,
                         tmp_prefix=None,
                         vvcrit=0.4,
                         grid_version="1.2",
                         feh_values=None,
                         afe_values=None):
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
    grid_version: str
        MIST grid version, e.g. "1.2" or "2.5"
    feh_values: list (optional)
        List of [Fe/H] values to download. If None, uses the grid defaults.
    afe_values: list (optional)
        List of [alpha/Fe] values to download (v2.5 only). If None, uses
        the grid defaults. For v1.2, must be None or [0.0].
    """

    grid_version = _normalize_grid_version(grid_version)
    if outp_prefix is None:
        outp_prefix = utils.get_data_path_for_grid(grid_version, vvcrit)
    if grid_version == "1.2":
        if feh_values is None:
            feh_values = [
                -4.00, -3.50, -3.00, -2.50, -2.00, -1.75, -1.50, -1.25,
                -1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50
            ]
        if afe_values is None:
            afe_values = [0.0]
        if not np.all(np.isclose(afe_values, 0.0)):
            raise ValueError('MIST v1.2 supports only [alpha/Fe]=0.0')
    elif grid_version == "2.5":
        if feh_values is None:
            feh_values = ([-4.0, -3.5, -3.0] +
                          list(np.round(np.arange(-2.75, 0.50 + 0.25, 0.25),
                                        2)))
        if afe_values is None:
            afe_values = np.round(np.arange(-0.2, 0.4 + 0.2, 0.2), 2)
    else:
        raise ValueError(f'Unsupported grid version: {grid_version}')
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
            if grid_version == "1.2":
                writer(__bc_url_v12(curfilt), T)
            else:
                writer(__bc_url_v25(curfilt), T)
        if grid_version == "1.2":
            for curfeh in feh_values:
                writer(__eep_url_v12(curfeh, vvcrit=vvcrit), T)
        else:
            for curfeh in feh_values:
                for curafe in afe_values:
                    writer(__eep_url_v25(curfeh, curafe, vvcrit=vvcrit), T)
        prepare(T,
                T,
                outp_prefix=outp_prefix,
                filters=filters,
                grid_version=grid_version,
                vvcrit=vvcrit)


def prepare(eep_prefix,
            bolom_prefix,
            outp_prefix=None,
            filters=('DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper',
                     'UBVRIplus', 'WISE'),
            grid_version="1.2",
            vvcrit=0.4):
    """
    Prepare the isochrone files

    Parameters
    ----------
    eep_prefix: string
        The path that has *EEP folders where *eep files will be searched
    bolom_prefix: string
        The path that has bolometric correction files *DECam *UBRI etc
    """
    grid_version = _normalize_grid_version(grid_version)
    if outp_prefix is None:
        outp_prefix = utils.get_data_path_for_grid(grid_version, vvcrit)
    print('Reading EEP grid')
    if not os.path.isdir(eep_prefix) or not os.path.isdir(outp_prefix):
        raise RuntimeError(
            'The arguments must be paths to the directories with EEP \
            and bolometric corrections')
    read_grid(eep_prefix, outp_prefix)
    print('Processing EEPs')
    tab = atpy.Table().read(os.path.join(outp_prefix, TRACKS_FILE))
    os.unlink(os.path.join(outp_prefix, TRACKS_FILE))  # remove after reading

    umass, mass_id = np.unique(np.array(tab['initial_mass']),
                               return_inverse=True)
    ufeh, feh_id = np.unique(np.array(tab['feh']), return_inverse=True)
    uafe, afe_id = np.unique(np.array(tab['afe']), return_inverse=True)

    neep = int(np.max(tab['EEP'])) + 1
    nfeh = len(ufeh)
    nafe = len(uafe)
    nmass = len(umass)
    grid_ndim = 4 if nafe > 1 else 3
    grids = ['logage', 'logteff', 'logg', 'logl', 'phase']
    for k in grids:
        if grid_ndim == 3:
            grid = np.zeros((nfeh, nmass, neep)) - np.nan
            if k == 'logage':
                grid[feh_id, mass_id,
                     tab['EEP']] = np.log10(tab['star_age'])
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
        else:
            grid = np.zeros((nfeh, nafe, nmass, neep)) - np.nan
            if k == 'logage':
                grid[feh_id, afe_id, mass_id,
                     tab['EEP']] = np.log10(tab['star_age'])
                grid[:, :, :, 1:] = np.diff(grid, axis=3)
            elif k == 'logteff':
                grid[feh_id, afe_id, mass_id, tab['EEP']] = tab['log_Teff']
            elif k == 'logg':
                grid[feh_id, afe_id, mass_id, tab['EEP']] = tab['log_g']
            elif k == 'logl':
                grid[feh_id, afe_id, mass_id, tab['EEP']] = tab['log_L']
            elif k == 'phase':
                grid[feh_id, afe_id, mass_id, tab['EEP']] = tab['phase']
            else:
                raise Exception('wrong ' + k)

            grid4d_filler(grid)
            if k == 'phase':
                grid[~np.isfinite(grid)] = -99
                grid = np.round(grid).astype(np.int8)
            if k == 'logage':
                grid[:, :, :, :] = np.cumsum(grid, axis=3)

        np.save(os.path.join(outp_prefix, get_file(k)), grid)

    np.savez(os.path.join(outp_prefix, INTERP_NPZ),
             umass=umass,
             ufeh=ufeh,
             uafe=uafe,
             neep=neep,
             grid_ndim=grid_ndim,
             grid_version=grid_version,
             vvcrit=vvcrit)
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


def _get_polylin_coeff(feh, ufeh, mass, umass, feh_ind1, feh_ind2, mass_ind1,
                       mass_ind2):

    x = (feh - ufeh[feh_ind1]) / (ufeh[feh_ind2] - ufeh[feh_ind1])
    # from 0 to 1
    y = (mass - umass[mass_ind1]) / (umass[mass_ind2] - umass[mass_ind1]
                                     )  # from 0 to 1
    # this is now bilinear interpolation in the space of mass/metallicity
    C11 = (1 - x) * (1 - y)
    C12 = (1 - x) * y
    C21 = x * (1 - y)
    C22 = x * y
    return C11, C12, C21, C22


def _interpolator(grid, C11, C12, C21, C22, ifeh1, ifeh2, imass1, imass2,
                  ieep):
    """
    Perform a bilinear interpolation given coefficients C11,...C22
    indices along feh dimension ifeh1, ifeh2 (referring to sequential indices)
    of the grid box in question
    indices along mass
    and index along eep axis
    """
    return (C11 * grid[ifeh1, imass1, ieep] + C12 * grid[ifeh1, imass2, ieep] +
            C21 * grid[ifeh2, imass1, ieep] + C22 * grid[ifeh2, imass2, ieep])


def _get_polylin_nd_coeffs(coords, axes, left_inds):
    """
    Compute generic N-D poly-linear coefficients and vertex indices.

    Parameters
    ----------
    coords: list of arrays
        Coordinate values per axis, each length N.
    axes: list of arrays
        Grid vectors per axis.
    left_inds: list of arrays
        Left indices per axis, each length N.
    """
    ndim = len(coords)
    left = np.stack(left_inds, axis=1)
    # Per-axis normalized coordinate x in [0,1] inside the enclosing cell:
    # x = (p - grid[left]) / (grid[left+1] - grid[left]).
    frac = np.stack([
        (coords[i] - axes[i][left_inds[i]]) /
        (axes[i][left_inds[i] + 1] - axes[i][left_inds[i]])
        for i in range(ndim)
    ],
                    axis=1)
    box = np.array(list(itertools.product(*[[0, 1]] * ndim)), dtype=int)
    # Poly-linear basis over hypercube vertices:
    # w_v = Π_d x_d^a_{v,d} (1-x_d)^(1-a_{v,d}),  a_{v,d}∈{0,1}.
    coeff = (frac[None, :, :]**box[:, None, :] *
             (1 - frac[None, :, :])**(1 - box)[:, None, :]).prod(axis=2)
    inds = left[None, :, :] + box[:, None, :]
    return coeff, inds


def _interpolator_polylin_nd(grid, coeff, inds, ieep):
    """
    Evaluate generic N-D poly-linear interpolation over grid[..., eep].
    """
    nvert, npts, ndim = inds.shape
    res = np.zeros(npts)
    # Evaluate f(p) = Σ_v w_v f(vertex_v) for each target point.
    for v in range(nvert):
        idx = tuple(inds[v, :, d] for d in range(ndim)) + (ieep, )
        res += coeff[v] * grid[idx]
    return res


class TheoryInterpolator:

    def __init__(self,
                 prefix=None,
                 grid_version="1.2",
                 vvcrit=0.4,
                 linear=False):
        """
        Construct the interpolator that computes theoretical
        quantities (logg, logl, logteff) given (mass, logage, feh)

        Parameters
        ----------
        prefix: str
            Path to the data folder
        """
        if prefix is None:
            grid_version = _normalize_grid_version(grid_version)
            prefix = utils.get_data_path_for_grid(grid_version, vvcrit)
        self.linear = bool(linear)
        meta_path = os.path.join(prefix, INTERP_NPZ)
        if not os.path.exists(meta_path):
            legacy = os.path.join(prefix, INTERP_PKL)
            if os.path.exists(legacy):
                with open(legacy, 'rb') as fp:
                    D = pickle.load(fp)
                self.umass = np.array(D['umass'])
                self.ufeh = np.array(D['ufeh'])
                self.uafe = np.array([0.0])
                self.neep = int(D['neep'])
                self.grid_ndim = 3
                np.savez(meta_path,
                         umass=self.umass,
                         ufeh=self.ufeh,
                         uafe=self.uafe,
                         neep=self.neep,
                         grid_ndim=self.grid_ndim,
                         grid_version="1.2",
                         vvcrit=0.4)
                warnings.warn(
                    'Converted legacy interp.pkl to interp.npz for future use.'
                )
            else:
                raise RuntimeError(
                    f'Metadata file {INTERP_NPZ} not found in {prefix}')
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            self.umass = np.array(meta['umass'])
            self.ufeh = np.array(meta['ufeh'])
            self.uafe = np.array(meta['uafe']) if 'uafe' in meta.files else np.array([0.0])
            self.neep = int(meta['neep'])
            self.grid_ndim = int(
                meta['grid_ndim']) if 'grid_ndim' in meta.files else 3
            meta.close()

        (self.logg_grid, self.logl_grid, self.logteff_grid, self.logage_grid,
         self.phase_grid) = [
             np.load(os.path.join(prefix, get_file(curt)))
             for curt in ['logg', 'logl', 'logteff', 'logage', 'phase']
         ]
        self._warned_afe = False

        # We fill nans along EEP axis by repeating the last finite value.
        # This provides the necessary 4-point footprint for cubic interpolation
        # at the track ends without requiring external data.
        # The original unfilled grid is kept for validity checks.
        self.logage_grid_unfilled = self.logage_grid.copy()
        for g in [
                self.logg_grid, self.logl_grid, self.logteff_grid,
                self.logage_grid
        ]:
            if g.dtype == np.float32 or g.dtype == np.float64:
                mask = np.isnan(g)
                if np.any(mask):
                    # For each track, find the last finite index and repeat it to the end
                    eep_idx = np.arange(g.shape[-1])
                    idx = np.where(~mask, eep_idx, 0)
                    idx = np.maximum.accumulate(idx, axis=-1)
                    grid_filled = np.take_along_axis(g, idx, axis=-1)
                    g[:] = grid_filled

    def __call__(self, mass, logage, feh, afe=0.0):
        """
        Return the theoretical isochrone values such as logL, logg, logteff
        correspoding to the mass, logage, feh and optionally afe
        """
        if self.grid_ndim == 3 and not self._warned_afe:
            if np.any(~np.isclose(afe, 0.0)):
                warnings.warn('[alpha/Fe] is ignored for MIST v1.2 grids.')
                self._warned_afe = True
        feh, mass, logage, afe = np.broadcast_arrays(
            np.atleast_1d(np.asarray(feh, dtype=np.float64)),
            np.atleast_1d(np.asarray(mass, dtype=np.float64)),
            np.atleast_1d(np.asarray(logage, dtype=np.float64)),
            np.atleast_1d(np.asarray(afe, dtype=np.float64)))

        N = len(logage)
        D_grids = {
            'logg': self.logg_grid,
            'logteff': self.logteff_grid,
            'logl': self.logl_grid,
            'phase': self.phase_grid
        }
        xret = {}
        if self.linear:
            if self.grid_ndim == 3:
                DD = self._get_eep_coeffs_3d_linear(mass, logage, feh)
            else:
                DD = self._get_eep_coeffs_4d_linear(mass, logage, feh, afe)
            bad = DD['bad']
            good = ~bad
            for curkey, curarr in D_grids.items():
                c1 = DD['coeff'][:, good]
                i1 = DD['inds'][:, good, :]
                e1 = DD['eep1'][good]
                e2 = DD['eep2'][good]
                ef = DD['eep_frac'][good]
                # Two-stage legacy scheme:
                # 1) poly-linear in (feh, [afe], mass) at fixed EEP endpoints,
                # 2) linear blend along EEP with fraction ef.
                v1 = _interpolator_polylin_nd(curarr, c1, i1, e1)
                v2 = _interpolator_polylin_nd(curarr, c1, i1, e2)
                xret[curkey] = v1 * (1 - ef) + ef * v2
        else:
            if self.grid_ndim == 3:
                DD = self._get_eep_coeffs_3d(mass, logage, feh)
                wf, ifehs, wm, imasses = (DD['wf'], DD['ifehs'], DD['wm'],
                                          DD['imasses'])
                we, ieeps, bad = (DD['we'], DD['ieeps'], DD['bad'])
            else:
                DD = self._get_eep_coeffs_4d(mass, logage, feh, afe)
                wf, ifehs, wa, iafes, wm, imasses = (DD['wf'], DD['ifehs'],
                                                     DD['wa'], DD['iafes'],
                                                     DD['wm'], DD['imasses'])
                we, ieeps, bad = (DD['we'], DD['ieeps'], DD['bad'])
            good = ~bad
            if self.grid_ndim == 3:
                (wf_good, ifehs_good, wm_good, imasses_good, we_good,
                 ieeps_good) = [
                     _[good] for _ in [wf, ifehs, wm, imasses, we, ieeps]
                 ]
                for curkey, curarr in D_grids.items():
                    xret[curkey] = utils._interpolator_tricubic(
                        curarr, wf_good, ifehs_good, wm_good, imasses_good,
                        we_good, ieeps_good)
            else:
                (wf_good, ifehs_good, wa_good, iafes_good, wm_good,
                 imasses_good, we_good, ieeps_good) = [
                     _[good]
                     for _ in [wf, ifehs, wa, iafes, wm, imasses, we, ieeps]
                 ]
                for curkey, curarr in D_grids.items():
                    xret[curkey] = utils._interpolator_quadcubic(
                        curarr, wf_good, ifehs_good, wa_good, iafes_good,
                        wm_good, imasses_good, we_good, ieeps_good)

        ret = {}
        for k in ['logg', 'logteff', 'logl', 'phase']:
            ret[k] = np.zeros(N) + np.nan
            ret[k][good] = xret[k]
        return ret

    def getLogAgeFromEEP(self, mass, eep, feh, afe=0.0, returnJac=False):
        """
        This method returns the log(age) for given mass eep and feh

        if returnJac is true the derivative is of d(log(age))/deep is returned
        """
        feh, mass, eep, afe = [
            np.atleast_1d(np.asarray(_, dtype=np.float64))
            for _ in [feh, mass, eep, afe]
        ]
        neep = self.neep
        N = len(feh)
        eep1 = eep.astype(int)
        if self.linear:
            eep2 = eep1 + 1
            if self.grid_ndim == 3:
                if not self._warned_afe:
                    if np.any(~np.isclose(afe, 0.0)):
                        warnings.warn(
                            '[alpha/Fe] is ignored for MIST v1.2 grids.')
                        self._warned_afe = True
                l1feh = np.searchsorted(self.ufeh, feh) - 1
                l1mass = np.searchsorted(self.umass, mass) - 1
                bad = np.zeros(N, dtype=bool)
                bad = bad | (l1mass + 1 >= len(self.umass)) | (
                    l1feh + 1 >= len(self.ufeh)) | (l1mass < 0) | (
                        l1feh < 0) | (eep2 >= neep) | (eep1 < 0)
                l1feh[bad] = 0
                l1mass[bad] = 0
                eep1[bad] = 0
                eep2[bad] = 1
                coeff, inds = _get_polylin_nd_coeffs([feh, mass],
                                                     [self.ufeh, self.umass],
                                                     [l1feh, l1mass])
            else:
                l1feh = np.searchsorted(self.ufeh, feh) - 1
                l1afe = np.searchsorted(self.uafe, afe) - 1
                l1mass = np.searchsorted(self.umass, mass) - 1
                bad = np.zeros(N, dtype=bool)
                bad = bad | (l1mass + 1 >= len(self.umass)) | (
                    l1feh + 1 >= len(self.ufeh)) | (
                        l1afe + 1 >= len(self.uafe)) | (l1mass < 0) | (
                            l1feh < 0) | (l1afe < 0) | (eep2 >= neep) | (
                                eep1 < 0)
                l1feh[bad] = 0
                l1afe[bad] = 0
                l1mass[bad] = 0
                eep1[bad] = 0
                eep2[bad] = 1
                coeff, inds = _get_polylin_nd_coeffs(
                    [feh, afe, mass], [self.ufeh, self.uafe, self.umass],
                    [l1feh, l1afe, l1mass])

            eep_frac = eep - eep1
            good = ~bad
            ret_logage = np.zeros_like(mass) + np.nan
            if np.any(good):
                # logage(EEP) on legacy mode is linear in the local [eep1,eep2] segment.
                logage1 = _interpolator_polylin_nd(self.logage_grid,
                                                   coeff[:, good],
                                                   inds[:, good, :],
                                                   eep1[good])
                logage2 = _interpolator_polylin_nd(self.logage_grid,
                                                   coeff[:, good],
                                                   inds[:, good, :],
                                                   eep2[good])
                ret_logage[good] = (logage1 * (1 - eep_frac[good]) +
                                    eep_frac[good] * logage2)
            if returnJac:
                jac = np.zeros_like(mass) + np.nan
                if np.any(good):
                    jac[good] = logage2 - logage1
                return ret_logage, jac
            return ret_logage

        if self.grid_ndim == 3:
            if not self._warned_afe:
                if np.any(~np.isclose(afe, 0.0)):
                    warnings.warn(
                        '[alpha/Fe] is ignored for MIST v1.2 grids.')
                    self._warned_afe = True
            l1feh = np.searchsorted(self.ufeh, feh) - 1
            l1mass = np.searchsorted(self.umass, mass) - 1

            bad = np.zeros(N, dtype=bool)
            bad = bad | (l1mass + 1 >= len(self.umass)) | (
                l1feh + 1 >= len(self.ufeh)) | (l1mass < 0) | (l1feh < 0) | (
                    eep1 + 1 >= neep) | (eep1 < 0)
            l1mass[bad] = 0
            l1feh[bad] = 0
            eep1[bad] = 0

            wf, ifehs = utils._get_cubic_coeffs(feh, self.ufeh, l1feh)
            wm, imasses = utils._get_cubic_coeffs(mass, self.umass, l1mass)
            ueep = np.arange(neep)
            we, ieeps = utils._get_cubic_coeffs(eep, ueep, eep1)

            goodsel = ~bad
            ret_logage = np.zeros_like(mass) + np.nan
            ret_logage[goodsel] = utils._interpolator_tricubic(
                self.logage_grid, wf[goodsel], ifehs[goodsel], wm[goodsel],
                imasses[goodsel], we[goodsel], ieeps[goodsel])
            if returnJac:
                jac = np.zeros_like(mass) + np.nan
                dwe = utils._get_cubic_coeffs_deriv(eep, ueep, eep1)
                jac[goodsel] = utils._interpolator_tricubic(
                    self.logage_grid, wf[goodsel], ifehs[goodsel],
                    wm[goodsel], imasses[goodsel], dwe[goodsel],
                    ieeps[goodsel])
                return ret_logage, jac
            return ret_logage
        else:
            l1feh = np.searchsorted(self.ufeh, feh) - 1
            l1afe = np.searchsorted(self.uafe, afe) - 1
            l1mass = np.searchsorted(self.umass, mass) - 1

            bad = np.zeros(N, dtype=bool)
            bad = bad | (l1mass + 1 >= len(self.umass)) | (
                l1feh + 1 >= len(self.ufeh)) | (
                    l1afe + 1 >= len(self.uafe)) | (l1mass < 0) | (
                        l1feh < 0) | (l1afe < 0) | (eep1 + 1 >= neep) | (
                            eep1 < 0)
            l1mass[bad] = 0
            l1feh[bad] = 0
            l1afe[bad] = 0
            eep1[bad] = 0

            wf, ifehs = utils._get_cubic_coeffs(feh, self.ufeh, l1feh)
            wa, iafes = utils._get_cubic_coeffs(afe, self.uafe, l1afe)
            wm, imasses = utils._get_cubic_coeffs(mass, self.umass, l1mass)
            ueep = np.arange(neep)
            we, ieeps = utils._get_cubic_coeffs(eep, ueep, eep1)

            goodsel = ~bad
            ret_logage = np.zeros_like(mass) + np.nan
            ret_logage[goodsel] = utils._interpolator_quadcubic(
                self.logage_grid, wf[goodsel], ifehs[goodsel], wa[goodsel],
                iafes[goodsel], wm[goodsel], imasses[goodsel], we[goodsel],
                ieeps[goodsel])
            if returnJac:
                jac = np.zeros_like(mass) + np.nan
                dwe = utils._get_cubic_coeffs_deriv(eep, ueep, eep1)
                jac[goodsel] = utils._interpolator_quadcubic(
                    self.logage_grid, wf[goodsel], ifehs[goodsel],
                    wa[goodsel], iafes[goodsel], wm[goodsel],
                    imasses[goodsel], dwe[goodsel], ieeps[goodsel])
                return ret_logage, jac
            return ret_logage

    def getMaxMassMS(self, logage, feh, afe=0.0):
        """Find the approximate value of maximum mass on the main sequence """
        if self.grid_ndim == 3 and not self._warned_afe:
            if np.any(~np.isclose(afe, 0.0)):
                warnings.warn('[alpha/Fe] is ignored for MIST v1.2 grids.')
                self._warned_afe = True
        N = len(self.umass) - 1
        i1 = 1
        i2 = N - 1
        stop = False
        while not stop:
            ix = (i1 + i2) // 2
            if (i2 - i1) == 1:
                stop = True
            res = self(self.umass[ix], logage, feh, afe)
            phase = res['phase'][0]
            bad = np.isnan(phase)
            # this is a max phase among interpolation box vertices
            if phase > 0.5 or bad:
                i2 = ix
            else:
                i1 = ix
        return self.umass[i1]

    def getMaxMass(self, logage, feh, afe=0.0):
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
        # The algorithm is the following
        # we first go over the mass grid find the right one
        # by binary search
        # Then we zoom in on that interval and use the fact that inside
        # that interval we'll have linear dependence of age on mass
        logage, feh, afe = np.float64(logage), np.float64(feh), np.float64(
            afe)
        if self.grid_ndim == 3 and not self._warned_afe:
            if np.any(~np.isclose(afe, 0.0)):
                warnings.warn('[alpha/Fe] is ignored for MIST v1.2 grids.')
                self._warned_afe = True
        # ensure 64bit float otherwise incosistencies in float computations
        # will kill us
        niter = 40
        im1 = 0
        # self.umass[1]
        im2 = len(self.umass) - 1  # self.umass[-1]
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l1afe = np.searchsorted(self.uafe, afe) - 1 if self.grid_ndim == 4 else None
        if self._isvalid(self.umass[im2],
                         logage,
                         feh,
                         afe,
                         l1feh=l1feh,
                         l1afe=l1afe):
            return self.umass[im2]
        for i in range(niter):
            curm = (im1 + im2) // 2
            good = self._isvalid(self.umass[curm],
                                 logage,
                                 feh,
                                 afe,
                                 l1feh=l1feh,
                                 l1afe=l1afe)
            if not good:
                im1, im2 = im1, curm
            else:
                im1, im2 = curm, im2
            if im2 - im1 == 1:
                break
        lo = self.umass[im1]
        hi = self.umass[im2]

        def _isfinite_mass(m):
            return np.isfinite(self(m, logage, feh, afe)['logl'][0])

        # Ensure lo is valid and hi is invalid for the refinement.
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

        # Refine the boundary so that lo is valid and hi is invalid.
        # Use a strict tolerance so that lo is finite but lo+tol is not.
        tol = 1e-7
        for _ in range(40):
            if hi - lo <= tol:
                break
            mid = 0.5 * (lo + hi)
            if _isfinite_mass(mid):
                lo = mid
            else:
                hi = mid

        return lo

    def _get_eep_coeffs_3d(self, mass, logage, feh):
        """
        This function gets all the necessary coefficients for the interpolation
The interpolation is done in two stages:
1) Bilinear integration over mass, feh with coefficients C11,C12,C21,C22
2) Then there is a final interpolation over EEP axis
"""
        feh, mass, logage = np.broadcast_arrays(
            np.atleast_1d(np.asarray(feh, dtype=np.float64)),
            np.atleast_1d(np.asarray(mass, dtype=np.float64)),
            np.atleast_1d(np.asarray(logage, dtype=np.float64)))

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

        feh_arr = np.atleast_1d(feh)
        mass_arr = np.atleast_1d(mass)
        l1feh_arr = np.atleast_1d(l1feh)
        l1mass_arr = np.atleast_1d(l1mass)

        wf, ifehs = utils._get_cubic_coeffs(feh_arr, self.ufeh, l1feh_arr)
        wm, imasses = utils._get_cubic_coeffs(mass_arr, self.umass,
                                              l1mass_arr)

        def getAge(cureep, subset):
            return utils._interpolator_bicubic(
                self.logage_grid_unfilled, wf[subset], ifehs[subset],
                wm[subset], imasses[subset], cureep)

        lefts, rights, bads = _binary_search(bads, logage, self.neep, getAge)

        good = ~bads
        LV = np.zeros(len(mass))
        RV = LV + 1
        LV[good] = getAge(lefts[good], good)
        RV[good] = getAge(rights[good], good)
        eep_frac = (logage - LV) / (RV - LV)
        eep_float = lefts + eep_frac

        # eep_float refinement with Newton steps for C1 continuity.
        # Since we search for the EEP phase that matches a target age,
        # and the age-to-EEP mapping is itself cubic, a simple linear
        # fractional step (eep_frac) leaves small C1 discontinuities.
        # Two Newton steps ensure the phase matches the age perfectly in C1 space.
        ueep = np.arange(self.neep)
        for _ in range(2):
            we, ieeps = utils._get_cubic_coeffs(eep_float, ueep, lefts)
            curr_age = utils._interpolator_tricubic(self.logage_grid, wf, ifehs, wm,
                                              imasses, we, ieeps)
            dwe = utils._get_cubic_coeffs_deriv(eep_float, ueep, lefts)
            curr_dage = utils._interpolator_tricubic(self.logage_grid, wf, ifehs, wm,
                                               imasses, dwe, ieeps)
            # only update good points and keep them within the bracket
            # found by binary search
            # Newton step on F(eep)=age_cubic(eep)-target_age:
            # eep <- eep - F/F'  with F'=d(age)/d(eep).
            step = (curr_age[good] - logage[good]) / np.where(
                curr_dage[good] > 0, curr_dage[good], 1e-10)
            eep_float[good] = np.clip(eep_float[good] - step, lefts[good],
                                      rights[good])

        we, ieeps = utils._get_cubic_coeffs(eep_float, ueep, lefts)

        return dict(wf=wf,
                    ifehs=ifehs,
                    wm=wm,
                    imasses=imasses,
                    we=we,
                    ieeps=ieeps,
                    bad=bads)

    def _get_eep_coeffs_4d(self, mass, logage, feh, afe):
        """
        This function gets all the necessary coefficients for 4D interpolation
        (feh, afe, mass, eep).
        """
        feh, mass, logage, afe = np.broadcast_arrays(
            np.atleast_1d(np.asarray(feh, dtype=np.float64)),
            np.atleast_1d(np.asarray(mass, dtype=np.float64)),
            np.atleast_1d(np.asarray(logage, dtype=np.float64)),
            np.atleast_1d(np.asarray(afe, dtype=np.float64)))

        N = len(logage)
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1afe = np.searchsorted(self.uafe, afe) - 1
        l2afe = l1afe + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        bads = np.zeros(N, dtype=bool)
        bads = bads | (l2mass >= len(self.umass)) | (l2feh >= len(
            self.ufeh)) | (l2afe >= len(self.uafe)) | (l1mass < 0) | (
                l1feh < 0) | (l1afe < 0)
        l1mass[bads] = 0
        l2mass[bads] = 1
        l1feh[bads] = 0
        l2feh[bads] = 1
        l1afe[bads] = 0
        l2afe[bads] = 1

        wf, ifehs = utils._get_cubic_coeffs(feh, self.ufeh, l1feh)
        wa, iafes = utils._get_cubic_coeffs(afe, self.uafe, l1afe)
        wm, imasses = utils._get_cubic_coeffs(mass, self.umass, l1mass)

        def getAge(cureep, subset):
            return utils._interpolator_tricubic_3d_eep(
                self.logage_grid_unfilled, wf[subset], ifehs[subset],
                wa[subset], iafes[subset], wm[subset], imasses[subset],
                cureep)

        lefts, rights, bads = _binary_search(bads, logage, self.neep, getAge)

        good = ~bads
        LV = np.zeros(len(mass))
        RV = LV + 1
        LV[good] = getAge(lefts[good], good)
        RV[good] = getAge(rights[good], good)
        eep_frac = (logage - LV) / (RV - LV)
        eep_float = lefts + eep_frac

        ueep = np.arange(self.neep)
        for _ in range(2):
            we, ieeps = utils._get_cubic_coeffs(eep_float, ueep, lefts)
            curr_age = utils._interpolator_quadcubic(self.logage_grid, wf,
                                                     ifehs, wa, iafes, wm,
                                                     imasses, we, ieeps)
            dwe = utils._get_cubic_coeffs_deriv(eep_float, ueep, lefts)
            curr_dage = utils._interpolator_quadcubic(self.logage_grid, wf,
                                                      ifehs, wa, iafes, wm,
                                                      imasses, dwe, ieeps)
            # Same Newton refinement in 4D coefficient space.
            step = (curr_age[good] - logage[good]) / np.where(
                curr_dage[good] > 0, curr_dage[good], 1e-10)
            eep_float[good] = np.clip(eep_float[good] - step, lefts[good],
                                      rights[good])

        we, ieeps = utils._get_cubic_coeffs(eep_float, ueep, lefts)

        return dict(wf=wf,
                    ifehs=ifehs,
                    wa=wa,
                    iafes=iafes,
                    wm=wm,
                    imasses=imasses,
                    we=we,
                    ieeps=ieeps,
                    bad=bads)

    def _get_eep_coeffs_3d_linear(self, mass, logage, feh):
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

        coeff, inds = _get_polylin_nd_coeffs([feh, mass],
                                             [self.ufeh, self.umass],
                                             [l1feh, l1mass])

        def getAge(cureep, subset):
            # Age at integer EEP from bilinear interpolation in (feh,mass).
            return _interpolator_polylin_nd(self.logage_grid_unfilled,
                                            coeff[:, subset], inds[:, subset, :],
                                            cureep)

        lefts, rights, bads = _binary_search(bads, logage, self.neep, getAge)
        LV = np.zeros(len(mass))
        RV = LV + 1
        good = ~bads
        LV[good] = getAge(lefts[good], good)
        RV[good] = getAge(rights[good], good)
        # eep_frac solves: logage = LV*(1-f) + RV*f  =>  f = (logage-LV)/(RV-LV)
        eep_frac = (logage - LV) / (RV - LV)
        return dict(coeff=coeff,
                    inds=inds,
                    eep_frac=eep_frac,
                    bad=bads,
                    eep1=lefts,
                    eep2=rights)

    def _get_eep_coeffs_4d_linear(self, mass, logage, feh, afe):
        feh, mass, logage, afe = [
            np.atleast_1d(np.asarray(_, dtype=np.float64))
            for _ in [feh, mass, logage, afe]
        ]
        N = len(logage)
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1afe = np.searchsorted(self.uafe, afe) - 1
        l2afe = l1afe + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        bads = np.zeros(N, dtype=bool)
        bads = bads | (l2mass >= len(self.umass)) | (l2feh >= len(
            self.ufeh)) | (l2afe >= len(self.uafe)) | (l1mass < 0) | (
                l1feh < 0) | (l1afe < 0)
        l1mass[bads] = 0
        l2mass[bads] = 1
        l1feh[bads] = 0
        l2feh[bads] = 1
        l1afe[bads] = 0
        l2afe[bads] = 1

        coeff, inds = _get_polylin_nd_coeffs([feh, afe, mass],
                                             [self.ufeh, self.uafe, self.umass],
                                             [l1feh, l1afe, l1mass])

        def getAge(cureep, subset):
            # Age at integer EEP from trilinear interpolation in (feh,afe,mass).
            return _interpolator_polylin_nd(self.logage_grid_unfilled,
                                            coeff[:, subset], inds[:, subset, :],
                                            cureep)

        lefts, rights, bads = _binary_search(bads, logage, self.neep, getAge)
        LV = np.zeros(len(mass))
        RV = LV + 1
        good = ~bads
        LV[good] = getAge(lefts[good], good)
        RV[good] = getAge(rights[good], good)
        eep_frac = (logage - LV) / (RV - LV)
        return dict(coeff=coeff,
                    inds=inds,
                    eep_frac=eep_frac,
                    bad=bads,
                    eep1=lefts,
                    eep2=rights)

    def _isvalid(self, mass, logage, feh, afe=0.0, l1feh=None, l1afe=None):
        """
        Checks if the point on the isochrone is valid
        """
        mass = np.float64(mass)
        logage = np.float64(logage)
        feh = np.float64(feh)
        if self.linear:
            if self.grid_ndim == 3:
                if l1feh is None:
                    l1feh = np.searchsorted(self.ufeh, feh) - 1
                l2feh = l1feh + 1
                l1mass = np.searchsorted(self.umass, mass) - 1
                l2mass = l1mass + 1

                if ((l2mass >= len(self.umass)) or (l2feh >= len(self.ufeh))
                        or (l1mass < 0) or (l1feh < 0)):
                    return False

                coeff, inds = _get_polylin_nd_coeffs(
                    [np.atleast_1d(feh), np.atleast_1d(mass)],
                    [self.ufeh, self.umass],
                    [np.atleast_1d(l1feh), np.atleast_1d(l1mass)])

                i1, i2 = 0, self.neep - 1

                def getAge(cureep):
                    return _interpolator_polylin_nd(self.logage_grid_unfilled,
                                                    coeff, inds, cureep)
            else:
                afe = np.float64(afe)
                if l1feh is None:
                    l1feh = np.searchsorted(self.ufeh, feh) - 1
                if l1afe is None:
                    l1afe = np.searchsorted(self.uafe, afe) - 1
                l2feh = l1feh + 1
                l2afe = l1afe + 1
                l1mass = np.searchsorted(self.umass, mass) - 1
                l2mass = l1mass + 1

                if ((l2mass >= len(self.umass)) or (l2feh >= len(self.ufeh))
                        or (l2afe >= len(self.uafe)) or (l1mass < 0)
                        or (l1feh < 0) or (l1afe < 0)):
                    return False

                coeff, inds = _get_polylin_nd_coeffs(
                    [np.atleast_1d(feh), np.atleast_1d(afe),
                     np.atleast_1d(mass)],
                    [self.ufeh, self.uafe, self.umass],
                    [np.atleast_1d(l1feh), np.atleast_1d(l1afe),
                     np.atleast_1d(l1mass)])

                i1, i2 = 0, self.neep - 1

                def getAge(cureep):
                    return _interpolator_polylin_nd(self.logage_grid_unfilled,
                                                    coeff, inds, cureep)

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
                    i2 = ix
            if np.isnan(getAge(i2)):
                return False
            return True

        if self.grid_ndim == 3:
            if l1feh is None:
                l1feh = np.searchsorted(self.ufeh, feh) - 1
            l2feh = l1feh + 1
            l1mass = np.searchsorted(self.umass, mass) - 1
            l2mass = l1mass + 1

            if ((l2mass >= len(self.umass)) or (l2feh >= len(self.ufeh))
                    or (l1mass < 0) or (l1feh < 0)):
                return False

            wf, ifehs = utils._get_cubic_coeffs(np.atleast_1d(feh), self.ufeh,
                                                np.atleast_1d(l1feh))
            wm, imasses = utils._get_cubic_coeffs(
                np.atleast_1d(mass), self.umass, np.atleast_1d(l1mass))

            i1, i2 = 0, self.neep - 1

            def getAge(cureep):
                return utils._interpolator_bicubic(self.logage_grid_unfilled,
                                                   wf, ifehs, wm, imasses,
                                                   cureep)

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
                    i2 = ix
            if np.isnan(getAge(i2)):
                return False
            return True
        else:
            afe = np.float64(afe)
            if l1feh is None:
                l1feh = np.searchsorted(self.ufeh, feh) - 1
            if l1afe is None:
                l1afe = np.searchsorted(self.uafe, afe) - 1
            l2feh = l1feh + 1
            l2afe = l1afe + 1
            l1mass = np.searchsorted(self.umass, mass) - 1
            l2mass = l1mass + 1

            if ((l2mass >= len(self.umass)) or (l2feh >= len(self.ufeh))
                    or (l2afe >= len(self.uafe)) or (l1mass < 0)
                    or (l1feh < 0) or (l1afe < 0)):
                return False

            wf, ifehs = utils._get_cubic_coeffs(np.atleast_1d(feh), self.ufeh,
                                                np.atleast_1d(l1feh))
            wa, iafes = utils._get_cubic_coeffs(np.atleast_1d(afe), self.uafe,
                                                np.atleast_1d(l1afe))
            wm, imasses = utils._get_cubic_coeffs(
                np.atleast_1d(mass), self.umass, np.atleast_1d(l1mass))

            i1, i2 = 0, self.neep - 1

            def getAge(cureep):
                return utils._interpolator_tricubic_3d_eep(
                    self.logage_grid_unfilled, wf, ifehs, wa, iafes, wm,
                    imasses, cureep)

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
                    i2 = ix
            if np.isnan(getAge(i2)):
                return False
            return True

    def _getMaxMassBox(self, logage, feh, l1feh, l2feh, l1mass, l2mass, afe=0.0):
        # here we are trying to find linear solutions
        # inside each EEP,mass,feh box to match our age
        if self.grid_ndim == 4:
            raise NotImplementedError(
                '_getMaxMassBox is not implemented for 4D grids')

        x = (feh - self.ufeh[l1feh]) / (self.ufeh[l2feh] - self.ufeh[l1feh])
        # from 0 to 1

        V11 = self.logage_grid_unfilled[l1feh, l1mass, :]
        V12 = self.logage_grid_unfilled[l1feh, l2mass, :]
        V21 = self.logage_grid_unfilled[l2feh, l1mass, :]
        V22 = self.logage_grid_unfilled[l2feh, l2mass, :]
        with warnings.catch_warnings():
            # protect against warnings here because we
            # are actively searching for valid range
            warnings.simplefilter("ignore")
            yy = (logage - V11 *
                  (1 - x) - V21 * x) / ((V12 - V11) *
                                        (1 - x) + V22 * x - V21 * x)
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
                 grid_version="1.2",
                 vvcrit=0.4,
                 linear=False):
        """
        Initialize the interpolator class, specifying filter names
        and optionally the folder where the preprocessed isochrones lie

        Parameters
        ----------
        filts: list
            List of strings, such as ['DECam_g','WISE_W1']
        data_prefix: str
            String for the data

        """
        if data_prefix is None:
            data_prefix = utils.get_data_path_for_grid(grid_version, vvcrit)
        self.isoInt = TheoryInterpolator(data_prefix,
                                         grid_version=grid_version,
                                         vvcrit=vvcrit,
                                         linear=linear)
        self.bolomInt = bolom.BCInterpolator(data_prefix, filts, linear=linear)

    def __call__(self, mass, logage, feh, afe=0.0):
        """
        Compute interpolated isochrone for a given mass log10(age), feh,
        and optionally afe

        Parameters
        ----------
        mass: float/numpy
            Either scalar or vector of masses
        logage: float/numpy
            Either scalar or vector of log10(age)
        feh: float/numpy
            Either scalar or vector of [Fe/H]
        afe: float/numpy
            Either scalar or vector of [alpha/Fe] (MIST v2.5 only)

        """
        mass, logage, feh, afe = [
            np.asarray(_, dtype=np.float64)
            for _ in [mass, logage, feh, afe]
        ]
        mass, logage, feh, afe = np.broadcast_arrays(mass, logage, feh, afe)
        shape = mass.shape
        mass, logage, feh, afe = [
            np.atleast_1d(_) for _ in [mass, logage, feh, afe]
        ]

        ret = self.isoInt(mass, logage, feh, afe)
        good_sub = np.isfinite(ret['logl'])

        av = ret['logl'][good_sub] * 0
        # computing when no extinction
        if self.bolomInt.ndim == 4:
            arr = np.array([
                ret['logteff'][good_sub], ret['logg'][good_sub],
                feh[good_sub], av
            ]).T
        elif self.bolomInt.ndim == 5:
            arr = np.array([
                ret['logteff'][good_sub], ret['logg'][good_sub],
                feh[good_sub], afe[good_sub], av
            ]).T
        else:
            raise RuntimeError(
                f'Unsupported BC dimensionality: {self.bolomInt.ndim}')
        res0 = self.bolomInt(arr)
        ret['logage'] = logage
        ret['feh'] = feh
        ret['afe'] = afe
        ret['mass'] = mass
        MBolSun = 4.74
        for k in res0:
            ret[k] = np.zeros(len(mass)) - np.nan
            ret[k][good_sub] = MBolSun - 2.5 * ret['logl'][good_sub] - res0[k]
        for k in ret.keys():
            ret[k] = ret[k].reshape(shape)
        return ret

    def getMaxMass(self, logage, feh, afe=0.0):
        """ Return the maximum mass on a given isochrone """
        return self.isoInt.getMaxMass(logage, feh, afe)

    def getMaxMassMS(self, logage, feh, afe=0.0):
        return self.isoInt.getMaxMassMS(logage, feh, afe)
