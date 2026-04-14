import tempfile
import warnings
import glob
import os
import gc
import subprocess
import urllib.request
import astropy.table as atpy
import scipy.interpolate
import itertools

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
    """Return filename for a saved grid array by `gridt` key."""
    return '%s_grid.npy' % (gridt)


INTERP_NPZ = 'interp.npz'
VALID_EEP_MAX_NPY = 'valid_eep_max.npy'
KNOWN_BAD_TRACK = dict(feh=-2.0, afe=0.2, initial_mass=0.1)


def get_interp_ready_file(gridt):
    """Return filename for finite interpolation-ready grid by `gridt` key."""
    return f'{gridt}_interp_grid.npy'


def _normalize_mist_version(mist_version):
    """Validate and normalize a MIST version string."""
    mist_version = utils.normalize_mist_version(mist_version)
    if mist_version not in ('1.2', '2.5'):
        raise ValueError('Only MIST versions 1.2 and 2.5 are supported'
                         f'got: {mist_version}')
    return mist_version


def _is_known_bad_track(feh, afe, initial_mass):
    """Return True for the known problematic MIST v2.5 low-mass track."""
    return (np.isclose(feh, KNOWN_BAD_TRACK['feh'])
            and np.isclose(afe, KNOWN_BAD_TRACK['afe'])
            and np.isclose(initial_mass, KNOWN_BAD_TRACK['initial_mass']))


def _is_substellar_lowmass_track(track_type, initial_mass):
    """Return True for substellar 0.1 Msun tracks."""
    return (np.isclose(initial_mass, KNOWN_BAD_TRACK['initial_mass'])
            and str(track_type).strip().lower().startswith('substellar'))


def _patch_known_bad_track(grid, ufeh, uafe, umass, grid_name):
    """
    Fill the known-bad missing track (feh=-2, afe=0.2, mass=0.1) from
    neighboring alpha tracks.
    """
    if grid.ndim != 4:
        return False
    feh_idx = np.where(np.isclose(ufeh, KNOWN_BAD_TRACK['feh']))[0]
    afe_idx = np.where(np.isclose(uafe, KNOWN_BAD_TRACK['afe']))[0]
    mass_idx = np.where(np.isclose(umass, KNOWN_BAD_TRACK['initial_mass']))[0]
    if len(feh_idx) == 0 or len(afe_idx) == 0 or len(mass_idx) == 0:
        return False
    fi, ai, mi = int(feh_idx[0]), int(afe_idx[0]), int(mass_idx[0])
    if np.isfinite(grid[fi, ai, mi, :]).all():
        return False

    lo_idx = np.where(np.isclose(uafe, 0.0))[0]
    hi_idx = np.where(np.isclose(uafe, 0.4))[0]
    if len(lo_idx) == 0 or len(hi_idx) == 0:
        return False
    a0, a1 = int(lo_idx[0]), int(hi_idx[0])
    t = (KNOWN_BAD_TRACK['afe'] - uafe[a0]) / (uafe[a1] - uafe[a0])
    v0 = grid[fi, a0, mi, :]
    v1 = grid[fi, a1, mi, :]
    both = np.isfinite(v0) & np.isfinite(v1)
    only0 = np.isfinite(v0) & ~np.isfinite(v1)
    only1 = ~np.isfinite(v0) & np.isfinite(v1)
    out = np.zeros_like(v0, dtype=np.float64) + np.nan
    out[both] = v0[both] * (1 - t) + v1[both] * t
    out[only0] = v0[only0]
    out[only1] = v1[only1]

    # Last resort: copy from adjacent mass track at same (feh, afe).
    if mi + 1 < grid.shape[2]:
        miss = ~np.isfinite(out)
        out[miss] = grid[fi, ai, mi + 1, :][miss]

    if grid_name == 'logage':
        good = np.isfinite(out)
        if good.any():
            out[good] = np.maximum.accumulate(out[good])
    if grid_name == 'phase':
        out[np.isfinite(out)] = np.round(out[np.isfinite(out)])

    grid[fi, ai, mi, :] = out
    return True


def getheader(f):
    """Parse MIST EEP file header metadata."""
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
    """Read and merge EEP tables found under `eep_prefix`."""
    mask = os.path.join(eep_prefix, '*EEPS', '*eep')
    fs = glob.glob(mask)
    if len(fs) == 0:
        mask = os.path.join(eep_prefix, '**', '*eep')
        fs = glob.glob(mask, recursive=True)
    if len(fs) == 0:
        raise RuntimeError(f'Failed to find eep files under {eep_prefix}')
    tmpfile = utils.tail_head(fs[0], 11, 10)
    tab0 = atpy.Table().read(tmpfile, format='ascii.fast_commented_header')
    os.unlink(tmpfile)
    tabs0 = []
    N = len(fs)
    nskip_bad = 0
    nskip_substellar = 0
    nstep = max(N // 100, 1)
    for i, f in enumerate(fs):
        if i % nstep == 0:
            print('%d/%d' % (i, N))
        D = getheader(f)
        if _is_substellar_lowmass_track(D['type'], D['initial_mass']):
            if _is_known_bad_track(D['feh'], D['afe'], D['initial_mass']):
                nskip_bad += 1
            else:
                nskip_substellar += 1
            continue
        curt = atpy.Table().read(f, format='ascii.fast_no_header')
        for i, k in enumerate(list(curt.columns)):
            curt.rename_column(k, list(tab0.columns)[i])
        curt['initial_mass'] = D['initial_mass']
        curt['feh'] = D['feh']
        curt['afe'] = D['afe']
        curt['EEP'] = np.arange(len(curt))
        tabs0.append(curt)
    if nskip_bad > 0:
        warnings.warn(f'Skipped {nskip_bad} known-bad MIST v2.5 track(s) '
                      '(feh=-2.0, afe=0.2, mass=0.1).')
    if nskip_substellar > 0:
        warnings.warn(
            f'Skipped {nskip_substellar} substellar 0.1 Msun track(s). '
            'These edge-of-grid tracks are excluded and will not be '
            'interpolated.')

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

    return tabs


def grid3d_filler(ima):
    """
    This fills nan gaps along one dimension in a 3d cube.
    I fill the gaps along mass dimension
    The array is modified

    Parameters
    ----------
    ima: np.ndarray
        Input 3D array modified in-place.
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

    Parameters
    ----------
    arr: np.ndarray
        Input 1D array modified in-place.
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


def build_interp_ready_grid_4d(grid):
    """
    Prepare a finite 4D grid (feh, afe, mass, eep) for cubic interpolation.
    """
    grid_filled = np.array(grid, copy=True)
    for i in range(grid_filled.shape[0]):
        for j in range(grid_filled.shape[1]):
            for k in range(grid_filled.shape[3]):
                arr = grid_filled[i, j, :, k]
                xids = np.nonzero(np.isfinite(arr))[0]
                if len(xids) > 0:
                    grid1d_filler(arr)
                    arr[:xids[0]] = arr[xids[0]]
                    arr[xids[-1] + 1:] = arr[xids[-1]]
                else:
                    arr[:] = 0
    for j in range(grid_filled.shape[1]):
        for m in range(grid_filled.shape[2]):
            for k in range(grid_filled.shape[3]):
                arr = grid_filled[:, j, m, k]
                xids = np.nonzero(np.isfinite(arr))[0]
                if len(xids) > 0:
                    grid1d_filler(arr)
                    arr[:xids[0]] = arr[xids[0]]
                    arr[xids[-1] + 1:] = arr[xids[-1]]
                else:
                    arr[:] = 0
    for i in range(grid_filled.shape[0]):
        for m in range(grid_filled.shape[2]):
            for k in range(grid_filled.shape[3]):
                arr = grid_filled[i, :, m, k]
                xids = np.nonzero(np.isfinite(arr))[0]
                if len(xids) > 0:
                    grid1d_filler(arr)
                    arr[:xids[0]] = arr[xids[0]]
                    arr[xids[-1] + 1:] = arr[xids[-1]]
                else:
                    arr[:] = 0
    return grid_filled


def _get_bc_url_v12(x):
    """Return BC tarball URL for MIST v1.2."""
    return 'https://waps.cfa.harvard.edu/MIST/BC_tables/v1/%s.txz' % x


def _get_bc_url_v25(x):
    """Return BC tarball URL for MIST v2.5."""
    return 'https://mist.science/BC_tables/v2/%s.txz' % x


def _get_eep_url_v12(feh, vvcrit=0.4):
    """Return EEP tarball URL for MIST v1.2 at one [Fe/H]."""
    feh_tag = _format_feh_v12(feh)
    return ('https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/' +
            'MIST_v1.2_feh_%s_afe_p0.0_vvcrit%.1f_EEPS.txz') % (feh_tag,
                                                                vvcrit)


def _get_eep_url_v25(feh, afe, vvcrit=0.4):
    """Return EEP tarball URL for MIST v2.5 at one [Fe/H] and [alpha/Fe]."""
    feh_tag = _format_feh_v25(feh)
    afe_tag = _format_afe(afe)
    return ('https://mist.science/data/tarballs_v2.5/eeps/' +
            'MIST_v2.5_feh_%s_afe_%s_vvcrit%.1f_EEPS.txz') % (feh_tag, afe_tag,
                                                              vvcrit)


def _get_default_grid(mist_version):
    """Return default [Fe/H] and [alpha/Fe] grids for a MIST version."""

    if mist_version == '1.2':
        feh_values = np.concatenate(
            [np.arange(-4, -2 + .1, 0.5),
             np.arange(-1.75, 0.5 + .1, 0.25)])
        afe_values = [0.0]
    else:
        feh_values = np.concatenate(
            [np.arange(-4, -3 + .1, 0.5),
             np.arange(-2.75, 0.5 + .1, 0.25)])
        afe_values = np.arange(-0.2, 0.6 + 0.1, 0.2)
    return {'feh': feh_values, 'afe': afe_values}


def _format_feh_v12(feh):
    """Format [Fe/H] token for MIST v1.2 filename conventions."""
    sign = 'm' if feh < 0 else 'p'
    val = abs(feh)
    return f"{sign}{val:.2f}"


def _format_feh_v25(feh):
    """Format [Fe/H] token for MIST v2.5 filename conventions."""
    sign = 'm' if feh < 0 else 'p'
    val = int(round(abs(feh) * 100))
    return f"{sign}{val:03d}"


def _format_afe(afe):
    """Format [alpha/Fe] token for MIST v2.5 filename conventions."""
    sign = 'm' if afe < 0 else 'p'
    val = int(round(abs(afe) * 10))
    return f"{sign}{val:d}"


def _download_and_unpack(url, pref):
    """
    Download a URL and unpack it in the folder
    """
    print('Downloading', url)
    with urllib.request.urlopen(url) as fd:
        fname = url.split('/')[-1]
        fname_out = os.path.join(pref, fname)
        with open(fname_out, 'wb') as fd_out:
            fd_out.write(fd.read())
    if os.name == 'nt':
        fname_out1 = fname_out.replace('.txz', '.tar')
        cmd = (f'cd /d {pref} && '
               f'7z x {fname_out} && '
               f'7z x {fname_out1}')
    else:
        cmd = f'cd {pref}; tar xfJ {fname_out}'
    ret = subprocess.run(cmd, shell=True, timeout=60, capture_output=True)
    if ret.returncode != 0:
        raise RuntimeError('Failed to untar the files' + ret.stdout.decode() +
                           ret.stderr.decode())


def get_bc_urls(filters, mist_version='1.2'):
    """
    Get bolometric-correction download URLs.

    Parameters
    ----------
    filters: iterable of str
        Filter-system groups to download.
    mist_version: str
        MIST version string ("1.2" or "2.5").
    """
    ret = []
    get_bc_url = _get_bc_url_v12
    if mist_version == '2.5':
        get_bc_url = _get_bc_url_v25
    for curfilt in filters:
        ret.append(get_bc_url(curfilt))
    return ret


def get_eep_urls(feh_values=None,
                 afe_values=None,
                 mist_version='1.2',
                 vvcrit=0.4):
    """
    Get EEP track download URLs.

    Parameters
    ----------
    feh_values: iterable or None
        [Fe/H] values to include. If None, version defaults are used.
    afe_values: iterable or None
        [alpha/Fe] values to include. If None, version defaults are used.
    mist_version: str
        MIST version string ("1.2" or "2.5").
    vvcrit: float
        Rotation value used in URL naming.
    """
    ret = []
    default_grid = _get_default_grid(mist_version)
    if feh_values is None:
        feh_values = default_grid['feh']
    if afe_values is None:
        afe_values = default_grid['afe']
    if mist_version == '1.2':
        for cur_feh in feh_values:
            ret.append(_get_eep_url_v12(cur_feh, vvcrit=vvcrit))
    if mist_version == '2.5':
        for cur_feh in feh_values:
            for cur_afe in afe_values:
                if np.isclose(cur_feh, 0.5) and np.isclose(cur_afe, 0.6):
                    # problematic tracks
                    continue
                ret.append(_get_eep_url_v25(cur_feh, cur_afe, vvcrit=vvcrit))
    return ret


def download_and_prepare(filters=[
    'DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper', 'UBVRIplus',
    'WISE'
],
                         outp_prefix=None,
                         tmp_prefix=None,
                         vvcrit=0.4,
                         mist_version='1.2',
                         bc_only=False,
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
    mist_version: str
        MIST version ("1.2" or "2.5").
    bc_only: bool
        If true, only download the bolometric corrections
    feh_values: list (optional)
        List of [Fe/H] values to download. If None, uses defaults for version.
    afe_values: list (optional)
        List of [alpha/Fe] values to download. Ignored for v1.2.
    """
    mist_version = _normalize_mist_version(mist_version)
    if outp_prefix is None:
        outp_prefix = utils.get_data_path_for_grid(mist_version=mist_version,
                                                   vvcrit=vvcrit)
    default_grid = _get_default_grid(mist_version)
    if feh_values is None:
        feh_values = default_grid['feh']
    if afe_values is None:
        afe_values = default_grid['afe']
    else:
        if mist_version == '1.2' and not np.all(np.isclose(afe_values, 0.0)):
            raise ValueError('MIST v1.2 supports only [alpha/Fe]=0.0')
    if not np.isclose([0., 0.4], vvcrit).any():
        raise ValueError('Only 0 and 0.4 values are allowed')

    with tempfile.TemporaryDirectory(dir=tmp_prefix) as cur_dir:
        urls = get_bc_urls(filters, mist_version=mist_version)
        if not bc_only:
            if mist_version == '2.5':
                print('WARNING the temporary size of the downloaded'
                      'tracks for the full grid is ~ 100 GB')
            urls = urls + get_eep_urls(feh_values=feh_values,
                                       afe_values=afe_values,
                                       mist_version=mist_version,
                                       vvcrit=vvcrit)
        for u in urls:
            _download_and_unpack(u, cur_dir)

        prepare(cur_dir,
                bolom_prefix=cur_dir,
                outp_prefix=outp_prefix,
                filters=filters,
                vvcrit=vvcrit,
                bc_only=bc_only,
                mist_version=mist_version)


def prepare(eep_prefix,
            bolom_prefix=None,
            outp_prefix=None,
            bc_only=False,
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
    outp_prefix: string or None
        Output directory for prepared arrays.
    bc_only: bool
        If True, prepare only bolometric-correction data.
    filters: iterable of str
        Filter-system groups used for BC preparation.
    vvcrit: float
        Rotation value used to select versioned output paths.
    mist_version: str
        MIST version string ("1.2" or "2.5").
    """
    mist_version = _normalize_mist_version(mist_version)
    if bolom_prefix is None:
        bolom_prefix = eep_prefix
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

    print('Reading/processing bolometric corrections')
    bolom.prepare(bolom_prefix, outp_prefix, filters)

    if bc_only:
        return

    tab = read_grid(eep_prefix)
    print('Processing EEPs')

    umass, mass_id = np.unique(np.array(tab['initial_mass']),
                               return_inverse=True)
    ufeh, feh_id = np.unique(np.array(tab['feh']), return_inverse=True)
    uafe, afe_id = np.unique(np.array(tab['afe']), return_inverse=True)

    neep = int(np.max(np.asarray(tab['EEP'], dtype=np.int64))) + 1
    nfeh = len(ufeh)
    nafe = len(uafe)
    nmass = len(umass)
    grid_ndim = 4 if nafe > 1 else 3
    grids = ['logage', 'logteff', 'logg', 'logl', 'phase']
    grid_store = {}
    for k in grids:
        if grid_ndim == 3:
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
            for i in range(nfeh):
                for j in range(nafe):
                    for e in range(neep):
                        grid1d_filler(grid[i, j, :, e])
            if k != 'logage':
                _patch_known_bad_track(grid, ufeh, uafe, umass, k)

        if k == 'phase':
            grid[~np.isfinite(grid)] = -99
            grid = np.round(grid).astype(np.int8)
        if k == 'logage':
            if grid_ndim == 3:
                grid[:, :, :] = np.cumsum(grid, axis=2)
            else:
                grid[:, :, :, :] = np.cumsum(grid, axis=3)
                _patch_known_bad_track(grid, ufeh, uafe, umass, k)

        grid_store[k] = grid
        np.save(os.path.join(outp_prefix, get_file(k)), grid)

    if grid_ndim == 3:
        valid_eep_max = np.max(np.where(np.isfinite(grid_store['logage']),
                                        np.arange(neep)[None, None, :], -1),
                               axis=2).astype(np.int16)
    else:
        valid_eep_max = np.max(np.where(np.isfinite(grid_store['logage']),
                                        np.arange(neep)[None, None, None, :],
                                        -1),
                               axis=3).astype(np.int16)
    np.save(os.path.join(outp_prefix, VALID_EEP_MAX_NPY), valid_eep_max)
    for k in ('logg', 'logl', 'logteff'):
        if grid_ndim == 3:
            interp_ready = build_interp_ready_grid(grid_store[k])
        else:
            interp_ready = build_interp_ready_grid_4d(grid_store[k])
        np.save(os.path.join(outp_prefix, get_interp_ready_file(k)),
                interp_ready)
    np.savez(os.path.join(outp_prefix, INTERP_NPZ),
             umass=umass,
             ufeh=ufeh,
             uafe=uafe,
             neep=neep,
             grid_ndim=grid_ndim,
             mist_version=mist_version,
             vvcrit=np.float64(vvcrit))


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


def _interpolator_2d_eep(grid, wfeh, ifehs, wmass, imasses, ieep):
    """Evaluate 2D (feh, mass) interpolation at fixed EEP indices."""
    ieep = np.asarray(ieep, dtype=int)
    return utils._interpolator_2d(grid, wfeh, ifehs, wmass, imasses, ieep)


class TheoryInterpolator:

    def __init__(self,
                 prefix=None,
                 interp_mode='linear',
                 mist_version='1.2',
                 vvcrit=0.4):
        """
        Initialize theory-grid interpolator for stellar quantities.

        Parameters
        ----------
        prefix: str or None
            Directory containing prepared theory grids.
        interp_mode: str
            Spatial interpolation mode: 'linear' or 'cubic'.
        mist_version: str
            MIST version string ("1.2" or "2.5").
        vvcrit: float
            Rotation value used when resolving default data path.
        """
        mist_version = _normalize_mist_version(mist_version)
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
            self.uafe = np.array(
                meta['uafe']) if 'uafe' in meta.files else np.array([0.0])
            self.neep = int(meta['neep'])
            self.grid_ndim = int(
                meta['grid_ndim']) if 'grid_ndim' in meta.files else 3
            self.mist_version = str(meta.get('mist_version', mist_version))
            self.vvcrit = float(meta.get('vvcrit', vvcrit))
        valid_eep_path = os.path.join(prefix, VALID_EEP_MAX_NPY)
        if not os.path.exists(valid_eep_path):
            raise RuntimeError(
                f'Validity file {VALID_EEP_MAX_NPY} not found in {prefix}. '
                'Please re-run minimint.download_and_prepare(...)')
        self.valid_eep_max = np.load(valid_eep_path)
        interp_mode = str(interp_mode).strip().lower()
        if interp_mode not in ('linear', 'cubic'):
            raise ValueError("interp_mode must be 'linear' or 'cubic'")
        self.interp_mode = interp_mode
        self._warned_afe = False

    def _warn_afe_ignored(self, afe):
        """Warn once when non-zero [alpha/Fe] is passed to a 3D (v1.2) grid."""
        if self.grid_ndim == 3 and (not self._warned_afe):
            if np.any(~np.isclose(np.asarray(afe, dtype=np.float64), 0.0)):
                warnings.warn('[alpha/Fe] is ignored for MIST v1.2 grids.')
                self._warned_afe = True

    def _eval_linear_interp(self, grid, DD, ieep, subset=None):
        """Evaluate linear spatial interpolation for a given EEP selection."""
        if subset is None:
            subset = slice(None)
        if self.grid_ndim == 3:
            return _interpolator_2d_eep(grid, DD['wfeh_lin'][subset],
                                        DD['ifehs_lin'][subset],
                                        DD['wmass_lin'][subset],
                                        DD['imasses_lin'][subset], ieep)
        return utils._interpolator_3d_eep(grid, DD['wfeh_lin'][subset],
                                          DD['ifehs_lin'][subset],
                                          DD['wafe_lin'][subset],
                                          DD['iafes_lin'][subset],
                                          DD['wmass_lin'][subset],
                                          DD['imasses_lin'][subset], ieep)

    def _eval_spatial_interp(self,
                             grid,
                             DD,
                             ieep,
                             subset=None,
                             use_cubic=False):
        """Evaluate spatial interpolation with optional cubic mode and fallback."""
        if subset is None:
            subset = slice(None)
        if not (use_cubic and self.interp_mode == 'cubic'):
            return self._eval_linear_interp(grid, DD, ieep, subset=subset)

        ieep = np.asarray(ieep, dtype=int)
        if self.grid_ndim == 3:
            wf = DD['wf'][subset]
            ifehs = DD['ifehs'][subset]
            wm = DD['wm'][subset]
            imasses = DD['imasses'][subset]
            res = utils._interpolator_bicubic(grid, wf, ifehs, wm, imasses,
                                              ieep)
            bad = np.zeros(len(res), dtype=bool)
            for i in range(4):
                for j in range(4):
                    bad |= ~np.isfinite(grid[ifehs[:, i], imasses[:, j], ieep])
        else:
            wf = DD['wf'][subset]
            ifehs = DD['ifehs'][subset]
            wa = DD['wa'][subset]
            iafes = DD['iafes'][subset]
            wm = DD['wm'][subset]
            imasses = DD['imasses'][subset]
            res = utils._interpolator_3d_eep(grid, wf, ifehs, wa, iafes, wm,
                                             imasses, ieep)
            bad = np.zeros(len(res), dtype=bool)
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        bad |= ~np.isfinite(grid[ifehs[:, i], iafes[:, j],
                                                 imasses[:, k], ieep])
        if bad.any():
            bad_idx = np.nonzero(bad)[0]
            if isinstance(subset, slice):
                subset_idx = np.arange(DD['wfeh_lin'].shape[0])[subset]
            else:
                subset_idx = np.asarray(subset)
            res[bad] = self._eval_linear_interp(grid,
                                                DD,
                                                ieep[bad_idx],
                                                subset=subset_idx[bad_idx])
        return res

    def _get_eep_coeffs(self, mass, logage, feh, afe=0.0):
        """Compute interpolation coefficients and EEP bracketing for queries."""
        feh, mass, logage, afe = [
            np.atleast_1d(np.asarray(_, dtype=np.float64))
            for _ in [feh, mass, logage, afe]
        ]
        N = len(logage)
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        bads = np.zeros(N, dtype=bool)
        if self.grid_ndim == 3:
            bads |= (l2mass >= len(self.umass)) | (l2feh >= len(self.ufeh))
            bads |= (l1mass < 0) | (l1feh < 0)
        else:
            l1afe = np.searchsorted(self.uafe, afe) - 1
            l2afe = l1afe + 1
            bads |= (l2mass >= len(self.umass)) | (l2feh >= len(self.ufeh))
            bads |= (l2afe >= len(self.uafe))
            bads |= (l1mass < 0) | (l1feh < 0) | (l1afe < 0)
            l1afe[bads] = 0
            l2afe[bads] = 1
        l1mass[bads] = 0
        l2mass[bads] = 1
        l1feh[bads] = 0
        l2feh[bads] = 1

        if self.grid_ndim == 3:
            wfeh_lin, ifehs_lin = utils._get_linear_coeffs(
                feh, self.ufeh, l1feh)
            wmass_lin, imasses_lin = utils._get_linear_coeffs(
                mass, self.umass, l1mass)
            wf = ifehs = wm = imasses = None
            if self.interp_mode == 'cubic':
                wf, ifehs = utils._get_cubic_coeffs(feh, self.ufeh, l1feh)
                wm, imasses = utils._get_cubic_coeffs(mass, self.umass, l1mass)

            def getAge(cureep_vec, subset):
                if np.isscalar(cureep_vec):
                    cureep_vec = np.full(len(subset), float(cureep_vec))
                ieep = np.asarray(cureep_vec, dtype=int)
                if self.interp_mode == 'cubic':
                    res = utils._interpolator_bicubic(self.logage_grid,
                                                      wf[subset],
                                                      ifehs[subset],
                                                      wm[subset],
                                                      imasses[subset], ieep)
                    bad = np.zeros(len(res), dtype=bool)
                    for i in range(4):
                        for j in range(4):
                            bad |= ~np.isfinite(
                                self.logage_grid[ifehs[subset, i],
                                                 imasses[subset, j], ieep])
                    if bad.any():
                        res[bad] = _interpolator_2d_eep(
                            self.logage_grid, wfeh_lin[subset][bad],
                            ifehs_lin[subset][bad], wmass_lin[subset][bad],
                            imasses_lin[subset][bad], ieep[bad])
                    return res
                return _interpolator_2d_eep(self.logage_grid, wfeh_lin[subset],
                                            ifehs_lin[subset],
                                            wmass_lin[subset],
                                            imasses_lin[subset], ieep)
        else:
            wfeh_lin, ifehs_lin = utils._get_linear_coeffs(
                feh, self.ufeh, l1feh)
            wafe_lin, iafes_lin = utils._get_linear_coeffs(
                afe, self.uafe, l1afe)
            wmass_lin, imasses_lin = utils._get_linear_coeffs(
                mass, self.umass, l1mass)
            wf = ifehs = wa = iafes = wm = imasses = None
            if self.interp_mode == 'cubic':
                wf, ifehs = utils._get_cubic_coeffs(feh, self.ufeh, l1feh)
                wa, iafes = utils._get_cubic_coeffs(afe, self.uafe, l1afe)
                wm, imasses = utils._get_cubic_coeffs(mass, self.umass, l1mass)

            def getAge(cureep_vec, subset):
                if np.isscalar(cureep_vec):
                    cureep_vec = np.full(len(subset), float(cureep_vec))
                ieep = np.asarray(cureep_vec, dtype=int)
                if self.interp_mode == 'cubic':
                    res = utils._interpolator_3d_eep(self.logage_grid,
                                                     wf[subset], ifehs[subset],
                                                     wa[subset], iafes[subset],
                                                     wm[subset],
                                                     imasses[subset], ieep)
                    bad = np.zeros(len(res), dtype=bool)
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                bad |= ~np.isfinite(
                                    self.logage_grid[ifehs[subset, i],
                                                     iafes[subset, j],
                                                     imasses[subset, k], ieep])
                    if bad.any():
                        res[bad] = utils._interpolator_3d_eep(
                            self.logage_grid, wfeh_lin[subset][bad],
                            ifehs_lin[subset][bad], wafe_lin[subset][bad],
                            iafes_lin[subset][bad], wmass_lin[subset][bad],
                            imasses_lin[subset][bad], ieep[bad])
                    return res
                return utils._interpolator_3d_eep(
                    self.logage_grid, wfeh_lin[subset], ifehs_lin[subset],
                    wafe_lin[subset], iafes_lin[subset], wmass_lin[subset],
                    imasses_lin[subset], ieep)

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
            eep_frac[good] = utils.solve_steffen_t(y_m1, y_0, y_1, y_2,
                                                   logage[good])

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
        if self.grid_ndim == 4:
            ret.update(
                dict(wafe_lin=wafe_lin,
                     iafes_lin=iafes_lin,
                     l1afe=l1afe,
                     l2afe=l2afe))
        if self.interp_mode == 'cubic':
            ret.update(dict(wf=wf, ifehs=ifehs, wm=wm, imasses=imasses))
            if self.grid_ndim == 4:
                ret.update(dict(wa=wa, iafes=iafes))
        return ret

    def __call__(self, mass, logage, feh, afe=0.0):
        """
        Interpolate theoretical quantities (`logg`, `logteff`, `logl`, `phase`).

        Parameters
        ----------
        mass: float or array-like
            Stellar mass values.
        logage: float or array-like
            Log10 age values.
        feh: float or array-like
            Metallicity [Fe/H] values.
        afe: float or array-like
            Alpha enhancement [alpha/Fe] values.
        """
        self._warn_afe_ignored(afe)
        feh, mass, logage, afe = [
            np.atleast_1d(np.asarray(_)) for _ in [feh, mass, logage, afe]
        ]
        N = len(logage)
        DD = self._get_eep_coeffs(mass, logage, feh, afe=afe)
        eep1, eep2, eep_frac, bad = (DD['eep1'], DD['eep2'], DD['eep_frac'],
                                     DD['bad'])
        good = ~bad
        xret = {}
        if good.any():
            good_idx = np.nonzero(good)[0]
            eep1_good, eep2_good, eep_frac_good = [
                _[good] for _ in [eep1, eep2, eep_frac]
            ]
            eep_m1_good = np.clip(eep1_good - 1, 0,
                                  self.neep - 1).astype(float)
            eep_0_good = eep1_good.astype(float)
            eep_1_good = eep2_good.astype(float)
            eep_2_good = np.clip(eep2_good + 1, 0, self.neep - 1).astype(float)

            for curkey, curarr in [('logg', self.logg_grid),
                                   ('logteff', self.logteff_grid),
                                   ('logl', self.logl_grid)]:
                curr = [
                    self._eval_spatial_interp(curarr,
                                              DD,
                                              eep_m1_good,
                                              subset=good_idx,
                                              use_cubic=True),
                    self._eval_spatial_interp(curarr,
                                              DD,
                                              eep_0_good,
                                              subset=good_idx,
                                              use_cubic=True),
                    self._eval_spatial_interp(curarr,
                                              DD,
                                              eep_1_good,
                                              subset=good_idx,
                                              use_cubic=True),
                    self._eval_spatial_interp(curarr,
                                              DD,
                                              eep_2_good,
                                              subset=good_idx,
                                              use_cubic=True)
                ]
                xret[curkey] = utils.steffen_interp(curr[0], curr[1], curr[2],
                                                    curr[3], eep_frac_good)

            phase0 = self._eval_linear_interp(self.phase_grid,
                                              DD,
                                              eep1_good.astype(int),
                                              subset=good_idx)
            phase1 = self._eval_linear_interp(self.phase_grid,
                                              DD,
                                              eep2_good.astype(int),
                                              subset=good_idx)
            xret['phase'] = phase0 + eep_frac_good * (phase1 - phase0)

        ret = {}
        for k in ['logg', 'logteff', 'logl', 'phase']:
            ret[k] = np.zeros(N) + np.nan
            if good.any():
                ret[k][good] = xret[k]
        return ret

    def getLogAgeFromEEP(self, mass, eep, feh, afe=0.0, returnJac=False):
        """
        Interpolate log-age as a function of EEP, mass, and composition.

        Parameters
        ----------
        mass: float or array-like
            Stellar mass values.
        eep: float or array-like
            EEP positions.
        feh: float or array-like
            Metallicity [Fe/H] values.
        afe: float or array-like
            Alpha enhancement [alpha/Fe] values.
        returnJac: bool
            If True, also return an approximate derivative d(logage)/dEEP.
        """
        self._warn_afe_ignored(afe)
        feh, mass, eep, afe = [
            np.atleast_1d(np.asarray(_, dtype=np.float64))
            for _ in [feh, mass, eep, afe]
        ]
        neep = self.neep
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        bad = np.zeros(len(feh), dtype=bool)
        if self.grid_ndim == 3:
            bad |= (l2mass >= len(self.umass)) | (l2feh >= len(self.ufeh))
            bad |= (l1mass < 0) | (l1feh < 0)
        else:
            l1afe = np.searchsorted(self.uafe, afe) - 1
            l2afe = l1afe + 1
            bad |= (l2mass >= len(self.umass)) | (l2feh >= len(self.ufeh))
            bad |= (l2afe >= len(self.uafe))
            bad |= (l1mass < 0) | (l1feh < 0) | (l1afe < 0)
            l1afe[bad] = 0
        l1feh[bad] = 0
        l1mass[bad] = 0

        DD = {}
        DD['wfeh_lin'], DD['ifehs_lin'] = utils._get_linear_coeffs(
            feh, self.ufeh, l1feh)
        DD['wmass_lin'], DD['imasses_lin'] = utils._get_linear_coeffs(
            mass, self.umass, l1mass)
        if self.grid_ndim == 4:
            DD['wafe_lin'], DD['iafes_lin'] = utils._get_linear_coeffs(
                afe, self.uafe, l1afe)
        if self.interp_mode == 'cubic':
            DD['wf'], DD['ifehs'] = utils._get_cubic_coeffs(
                feh, self.ufeh, l1feh)
            DD['wm'], DD['imasses'] = utils._get_cubic_coeffs(
                mass, self.umass, l1mass)
            if self.grid_ndim == 4:
                DD['wa'], DD['iafes'] = utils._get_cubic_coeffs(
                    afe, self.uafe, l1afe)

        eep1 = eep.astype(int)
        eep2 = eep1 + 1
        bad |= (eep1 < 0) | (eep2 >= neep)
        eep1[bad] = 0
        eep2[bad] = 1
        eep_frac = eep - eep1
        goodsel = ~bad

        ret_logage = np.zeros_like(mass) + np.nan
        jac = np.zeros_like(mass) + np.nan
        if goodsel.any():
            good_idx = np.nonzero(goodsel)[0]
            eep_m1 = np.clip(eep1[goodsel] - 1, 0, neep - 1).astype(float)
            eep_0 = eep1[goodsel].astype(float)
            eep_1 = eep2[goodsel].astype(float)
            eep_2 = np.clip(eep2[goodsel] + 1, 0, neep - 1).astype(float)
            logage_m1 = self._eval_spatial_interp(self.logage_grid,
                                                  DD,
                                                  eep_m1,
                                                  subset=good_idx,
                                                  use_cubic=True)
            logage_0 = self._eval_spatial_interp(self.logage_grid,
                                                 DD,
                                                 eep_0,
                                                 subset=good_idx,
                                                 use_cubic=True)
            logage_1 = self._eval_spatial_interp(self.logage_grid,
                                                 DD,
                                                 eep_1,
                                                 subset=good_idx,
                                                 use_cubic=True)
            logage_2 = self._eval_spatial_interp(self.logage_grid,
                                                 DD,
                                                 eep_2,
                                                 subset=good_idx,
                                                 use_cubic=True)
            ret_logage[goodsel] = utils.steffen_interp(logage_m1, logage_0,
                                                       logage_1, logage_2,
                                                       eep_frac[goodsel])
            jac[goodsel] = logage_1 - logage_0
        return (ret_logage, jac) if returnJac else ret_logage

    def getMaxMassMS(self, logage, feh, afe=0.0):
        """
        Return the approximate maximum main-sequence mass for `logage`, `feh`,
        and `afe`.
        """
        self._warn_afe_ignored(afe)
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
            if phase > 0.5 or bad:
                i2 = ix
            else:
                i1 = ix
        return self.umass[i1]

    def getMaxMass(self, logage, feh, afe=0.0):
        """
        Return the maximum mass with finite interpolation for `logage`, `feh`,
        and `afe`.
        """
        self._warn_afe_ignored(afe)
        logage, feh, afe = np.float64(logage), np.float64(feh), np.float64(afe)
        niter = 40
        im1 = 0
        im2 = len(self.umass) - 1
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l1afe = np.searchsorted(self.uafe,
                                afe) - 1 if self.grid_ndim == 4 else None
        if self._isvalid(self.umass[im2],
                         logage,
                         feh,
                         afe,
                         l1feh=l1feh,
                         l1afe=l1afe):
            return self.umass[im2]
        for _ in range(niter):
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

    def _isvalid(self, mass, logage, feh, afe=0.0, l1feh=None, l1afe=None):
        """Check whether a query point is valid for isochrone interpolation."""
        mass = np.float64(mass)
        logage = np.float64(logage)
        feh = np.float64(feh)
        afe = np.float64(afe)
        if l1feh is None:
            l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        if self.grid_ndim == 3:
            if ((l2mass >= len(self.umass)) or (l2feh >= len(self.ufeh))
                    or (l1mass < 0) or (l1feh < 0)):
                return False
            wfeh_lin, ifehs_lin = utils._get_linear_coeffs(
                np.array([feh]), self.ufeh, np.array([l1feh]))
            wmass_lin, imasses_lin = utils._get_linear_coeffs(
                np.array([mass]), self.umass, np.array([l1mass]))
            wf = ifehs = wm = imasses = None
            if self.interp_mode == 'cubic':
                wf, ifehs = utils._get_cubic_coeffs(np.array([feh]), self.ufeh,
                                                    np.array([l1feh]))
                wm, imasses = utils._get_cubic_coeffs(np.array([mass]),
                                                      self.umass,
                                                      np.array([l1mass]))
            i2 = int(
                np.min(self.valid_eep_max[[l1feh, l1feh, l2feh, l2feh],
                                          [l1mass, l2mass, l1mass, l2mass]]))

            def getAge(cureep):
                ieep = np.array([cureep], dtype=int)
                if self.interp_mode == 'cubic':
                    val = utils._interpolator_bicubic(self.logage_grid, wf,
                                                      ifehs, wm, imasses,
                                                      ieep)[0]
                    for i in range(4):
                        for j in range(4):
                            if not np.isfinite(
                                    self.logage_grid[ifehs[0, i],
                                                     imasses[0, j], ieep[0]]):
                                return _interpolator_2d_eep(
                                    self.logage_grid, wfeh_lin, ifehs_lin,
                                    wmass_lin, imasses_lin, ieep)[0]
                    return val
                return _interpolator_2d_eep(self.logage_grid, wfeh_lin,
                                            ifehs_lin, wmass_lin, imasses_lin,
                                            ieep)[0]
        else:
            if l1afe is None:
                l1afe = np.searchsorted(self.uafe, afe) - 1
            l2afe = l1afe + 1
            if ((l2mass >= len(self.umass)) or (l2feh >= len(self.ufeh))
                    or (l2afe >= len(self.uafe)) or (l1mass < 0) or (l1feh < 0)
                    or (l1afe < 0)):
                return False
            wfeh_lin, ifehs_lin = utils._get_linear_coeffs(
                np.array([feh]), self.ufeh, np.array([l1feh]))
            wafe_lin, iafes_lin = utils._get_linear_coeffs(
                np.array([afe]), self.uafe, np.array([l1afe]))
            wmass_lin, imasses_lin = utils._get_linear_coeffs(
                np.array([mass]), self.umass, np.array([l1mass]))
            wf = ifehs = wa = iafes = wm = imasses = None
            if self.interp_mode == 'cubic':
                wf, ifehs = utils._get_cubic_coeffs(np.array([feh]), self.ufeh,
                                                    np.array([l1feh]))
                wa, iafes = utils._get_cubic_coeffs(np.array([afe]), self.uafe,
                                                    np.array([l1afe]))
                wm, imasses = utils._get_cubic_coeffs(np.array([mass]),
                                                      self.umass,
                                                      np.array([l1mass]))
            inds = list(
                itertools.product([l1feh, l2feh], [l1afe, l2afe],
                                  [l1mass, l2mass]))
            i2 = int(np.min([self.valid_eep_max[i, j, k] for i, j, k in inds]))

            def getAge(cureep):
                ieep = np.array([cureep], dtype=int)
                if self.interp_mode == 'cubic':
                    val = utils._interpolator_3d_eep(self.logage_grid, wf,
                                                     ifehs, wa, iafes, wm,
                                                     imasses, ieep)[0]
                    for i in range(4):
                        for j in range(4):
                            for k in range(4):
                                if not np.isfinite(
                                        self.logage_grid[ifehs[0, i], iafes[0,
                                                                            j],
                                                         imasses[0,
                                                                 k], ieep[0]]):
                                    return utils._interpolator_3d_eep(
                                        self.logage_grid, wfeh_lin, ifehs_lin,
                                        wafe_lin, iafes_lin, wmass_lin,
                                        imasses_lin, ieep)[0]
                    return val
                return utils._interpolator_3d_eep(self.logage_grid, wfeh_lin,
                                                  ifehs_lin, wafe_lin,
                                                  iafes_lin, wmass_lin,
                                                  imasses_lin, ieep)[0]

        if i2 < 1:
            return False

        i1 = 0
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

    def _getMaxMassBox(self,
                       logage,
                       feh,
                       l1feh,
                       l2feh,
                       l1mass,
                       l2mass,
                       afe=0.0):
        """Estimate max mass inside one linear interpolation cell (3D grids)."""
        if self.grid_ndim == 4:
            raise NotImplementedError(
                '_getMaxMassBox is not implemented for 4D grids')
        x = (feh - self.ufeh[l1feh]) / (self.ufeh[l2feh] - self.ufeh[l1feh])
        V11 = self.logage_grid[l1feh, l1mass, :]
        V12 = self.logage_grid[l1feh, l2mass, :]
        V21 = self.logage_grid[l2feh, l1mass, :]
        V22 = self.logage_grid[l2feh, l2mass, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if x == 0:
                yy = (logage - V11) / (V12 - V11)
            elif x == 1:
                yy = (logage - V21) / (V22 - V21)
            else:
                yy = (logage - V11 *
                      (1 - x) - V21 * x) / ((V12 - V11) * (1 - x) +
                                            (V22 - V21) * x)
        yy = yy[np.isfinite(yy) & (yy <= 1) & (yy >= 0)]
        if len(yy) > 0:
            return self.umass[l1mass] + np.nanmax(
                (self.umass[l2mass] - self.umass[l1mass]) * yy)
        return np.nan


class Interpolator:

    def __init__(self,
                 filts,
                 data_prefix=None,
                 interp_mode='linear',
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
        interp_mode: str
            Spatial interpolation mode: 'linear' or 'cubic'
        mist_version: str
            MIST version ("1.2" or "2.5").
        vvcrit: float
            The value of V/Vcrit used for prepared data selection.
        """
        mist_version = _normalize_mist_version(mist_version)
        if data_prefix is None:
            data_prefix = utils.get_data_path_for_grid(
                mist_version=mist_version, vvcrit=vvcrit, create=False)
        self.isoInt = TheoryInterpolator(data_prefix,
                                         interp_mode=interp_mode,
                                         mist_version=mist_version,
                                         vvcrit=vvcrit)
        self.bolomInt = bolom.BCInterpolator(data_prefix, filts)

    def __call__(self, mass, logage, feh, afe=0.0):
        """
        Compute interpolated isochrone for a given mass log10(age), feh, and
        optionally [alpha/Fe].

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
            np.asarray(_, dtype=np.float64) for _ in [mass, logage, feh, afe]
        ]
        mass, logage, feh, afe = np.broadcast_arrays(mass, logage, feh, afe)
        shape = mass.shape
        mass, logage, feh, afe = [
            np.atleast_1d(_) for _ in [mass, logage, feh, afe]
        ]

        ret = self.isoInt(mass, logage, feh, afe)
        good_sub = np.isfinite(ret['logl'])

        av = ret['logl'][good_sub] * 0
        if self.bolomInt.ndim == 4:
            arr = np.array([
                ret['logteff'][good_sub], ret['logg'][good_sub], feh[good_sub],
                av
            ]).T
        elif self.bolomInt.ndim == 5:
            arr = np.array([
                ret['logteff'][good_sub], ret['logg'][good_sub], feh[good_sub],
                afe[good_sub], av
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
        """
        Return the maximum mass on a given isochrone.

        Parameters
        ----------
        logage: float
            Log10 age value.
        feh: float
            Metallicity [Fe/H].
        afe: float
            Alpha enhancement [alpha/Fe].
        """
        return self.isoInt.getMaxMass(logage, feh, afe)

    def getMaxMassMS(self, logage, feh, afe=0.0):
        """
        Return the maximum mass still on the main sequence.

        Parameters
        ----------
        logage: float
            Log10 age value.
        feh: float
            Metallicity [Fe/H].
        afe: float
            Alpha enhancement [alpha/Fe].
        """
        return self.isoInt.getMaxMassMS(logage, feh, afe)
