import urllib.request
import tempfile
import glob
import os
import gc
import pickle
import astropy.table as atpy
import scipy.interpolate
import numpy as np
from minimint import bolom, utils

TRACKS_FILE = 'tracks.fits'
LOGL_FILE = 'logl_grid.npy'
LOGG_FILE = 'logg_grid.npy'
LOGAGE_FILE = 'logage_grid.npy'
LOGTEFF_FILE = 'logteff_grid.npy'
INTERP_PKL = 'interp.pkl'


def getheader(f):
    fp = open(f, 'r')
    for i in range(5):
        l = fp.readline()
    vals = l[1:].split()
    #Yinit        Zinit   [Fe/H]   [a/Fe]  v/vcrit

    D = {'feh': float(vals[2]), 'afe': float(vals[3])}
    for i in range(3):
        l = fp.readline()
    #     initial_mass   N_pts   N_EEP   N_col   phase        type\n'
    # 3.8000000000E+00    1409       9      77     YES    low-mass\n'
    vals = l[1:].split()
    D['initial_mass'] = float(vals[0])
    D['N_pts'] = int(vals[1])
    D['N_EEP'] = int(vals[2])
    D['type'] = (vals[5])
    return D


def read_grid(eep_prefix, outp_prefix):
    fs = glob.glob('%s/*EEPS/*eep' % (eep_prefix, ))
    assert (len(fs) > 0)
    tmpfile = utils.tail_head(fs[0], 11, 10)
    tab0 = atpy.Table().read(tmpfile, format='ascii.fast_commented_header')
    os.unlink(tmpfile)
    tabs0 = []
    N = len(fs)
    for i, f in enumerate(fs):
        if i % (N // 100) == 0:
            print('%d/%d' % (i, N))
        curt = atpy.Table().read(f, format='ascii')
        for i, k in enumerate(list(curt.columns)):
            curt.rename_column(k, list(tab0.columns)[i])
        D = getheader(f)
        curt['initial_mass'] = D['initial_mass']
        curt['feh'] = D['feh']
        curt['EEP'] = np.arange(len(curt))
        tabs0.append(curt)

    tabs = atpy.vstack(tabs0)
    del tabs0
    gc.collect()
    for k in list(tabs.columns):
        if k not in [
                'star_age', 'star_mass', 'log_L', 'log_g', 'log_Teff',
                'initial_mass', 'phase', 'feh', 'EEP'
        ]:
            tabs.remove_column(k)

    os.makedirs(outp_prefix, exist_ok=True)
    tabs.write(outp_prefix + '/' + TRACKS_FILE, overwrite=True)


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
    left, right = xids[0], xids[-1]
    xids1 = np.arange(left, right + 1)
    xids1 = xids1[~np.isfinite(arr[xids1])]
    if len(xids1) > 0:
        arr[xids1] = scipy.interpolate.UnivariateSpline(xids,
                                                        arr[xids],
                                                        s=0,
                                                        k=1)(xids1)


def download_and_prepare(filters=[
    'DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper', 'UBVRIplus',
    'WISE'
],
                         outp_prefix=None,
                         tmp_prefix=None):
    """ Download the MIST isochrones and prepare the prerocessed isochrones
    Parameters

    ----------
    filters: tuple
        List of filter systems ['DECam','GALEX',...']
    outp_prefix: string (optional)
        Output directory for processed files
    tmp_prefix: string (optional)
        Temporary directory for storing downloaded files
    """

    bc_url = lambda x: 'https://waps.cfa.harvard.edu/MIST/BC_tables/%s.txz' % x
    eep_url = lambda x: 'https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_feh_%s_afe_p0.0_vvcrit0.4_EEPS.txz' % x
    mets = 'm4.00,m3.50,m3.00,m2.50,m2.00,m1.75,m1.50,m1.25,m1.00,m0.75,m0.50,m0.25,p0.00,p0.25,p0.50'.split(
        ',')

    def writer(url, pref):
        print('Downloading', url)
        fd = urllib.request.urlopen(url)
        fname = url.split('/')[-1]
        fdout = open(pref + '/' + fname, 'wb')
        fdout.write(fd.read())
        fdout.close()
        fd.close()
        cmd = 'cd %s; tar xfJ %s' % (pref, fname)
        os.system(cmd)

    with tempfile.TemporaryDirectory(dir=tmp_prefix) as T:
        for curfilt in filters:
            writer(bc_url(curfilt), T)
        for curmet in mets:
            writer(eep_url(curmet), T)
        prepare(T, T, outp_prefix=outp_prefix, filters=filters)


def prepare(eep_prefix,
            bolom_prefix,
            outp_prefix=None,
            filters=('DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper',
                     'UBVRIplus', 'WISE')):
    """
    Prepare the isochrone files 
    
    Parameters
    ----------
    eep_prefix: string
        The path that has *EEP folders where *eep files will be searched
    bolom_prefix: string
        The path that has bolometric correction files *DECam *UBRI etc
    """
    if outp_prefix is None:
        outp_prefix = utils.get_data_path()
    print('Reading EEP grid')
    if not os.path.isdir(eep_prefix) or not os.path.isdir(outp_prefix):
        raise RuntimeError(
            'The arguments must be paths to the directories with *EEP and bolometric corrections'
        )
    read_grid(eep_prefix, outp_prefix)
    print('Processing EEPs')
    tab = atpy.Table().read(outp_prefix + '/' + TRACKS_FILE)
    os.unlink(outp_prefix + '/' + TRACKS_FILE)  # remove after reading

    umass, mass_id = np.unique(np.array(tab['initial_mass']),
                               return_inverse=True)
    ufeh, feh_id = np.unique(np.array(tab['feh']), return_inverse=True)

    neep = 1710
    nfeh = len(ufeh)
    nmass = len(umass)

    logage_grid = np.zeros((nfeh, nmass, neep)) - np.nan
    logteff_grid = np.zeros((nfeh, nmass, neep)) - np.nan
    logg_grid = np.zeros((nfeh, nmass, neep)) - np.nan
    logl_grid = np.zeros((nfeh, nmass, neep)) - np.nan

    logage_grid[feh_id, mass_id, tab['EEP']] = np.log10(tab['star_age'])
    logage_grid[:, :, 1:] = np.diff(logage_grid, axis=2)
    logg_grid[feh_id, mass_id, tab['EEP']] = tab['log_g']
    logteff_grid[feh_id, mass_id, tab['EEP']] = tab['log_Teff']
    logl_grid[feh_id, mass_id, tab['EEP']] = tab['log_L']

    if True:
        grid3d_filler(logg_grid)
        grid3d_filler(logteff_grid)
        grid3d_filler(logl_grid)
        grid3d_filler(logage_grid)
    logage_grid[:, :, :] = np.cumsum(logage_grid, axis=2)

    np.save(outp_prefix + '/' + LOGG_FILE, logg_grid)
    np.save(outp_prefix + '/' + LOGL_FILE, logl_grid)
    np.save(outp_prefix + '/' + LOGTEFF_FILE, logteff_grid)
    np.save(outp_prefix + '/' + LOGAGE_FILE, logage_grid)
    with open(outp_prefix + '/' + INTERP_PKL, 'wb') as fp:
        pickle.dump(dict(umass=umass, ufeh=ufeh, neep=neep), fp)
    print('Reading/processing bolometric corrections')
    bolom.prepare(bolom_prefix, outp_prefix, filters)


class TheoryInterpolator:
    def __init__(self, prefix=None):
        """
        Construct the interpolator that computes theoretical 
        quantities (logg, logl, logteff) given (mass, logage, feh)
        
        Parameters
        ----------
        prefix: str
            Path to the data folder
        """
        if prefix is None:
            prefix = utils.get_data_path()
        self.logg_grid = np.load(prefix + '/' + LOGG_FILE)
        self.logl_grid = np.load(prefix + '/' + LOGL_FILE)
        self.logteff_grid = np.load(prefix + '/' + LOGTEFF_FILE)
        self.logage_grid = np.load(prefix + '/' + LOGAGE_FILE)
        with open(prefix + '/' + INTERP_PKL, 'rb') as fp:
            D = pickle.load(fp)
            self.umass = np.array(D['umass'])
            self.ufeh = np.array(D['ufeh'])
            self.neep = D['neep']

    def __call__(self, mass, logage, feh):
        feh, mass, logage = [
            np.atleast_1d(np.asarray(_)) for _ in [feh, mass, logage]
        ]
        #print (age)
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        l2feh = l1feh + 1
        l1mass = np.searchsorted(self.umass, mass) - 1
        l2mass = l1mass + 1
        bad = np.zeros(len(feh), dtype=bool)
        bad = bad | (l2mass >= len(self.umass) - 1) | (
            l2feh >= len(self.ufeh) - 1) | (l1mass < 0) | (l1feh < 0)
        l1mass[bad] = 0
        l2mass[bad] = 1
        l1feh[bad] = 0
        l2feh[bad] = 1

        poss = []
        x = (feh - self.ufeh[l1feh]) / (self.ufeh[l2feh] - self.ufeh[l1feh]
                                        )  # from 0 to 1
        y = (mass - self.umass[l1mass]) / (
            self.umass[l2mass] - self.umass[l1mass])  # from 0 to 1
        C1 = (1 - x) * (1 - y)
        C2 = (1 - x) * y
        C3 = x * (1 - y)
        C4 = x * y
        logage_new, logg_new, logteff_new, logl_new = [
            (C1[:, None] * _[l1feh, l1mass] + C2[:, None] * _[l1feh, l2mass] +
             C3[:, None] * _[l2feh, l1mass] + C4[:, None] * _[l2feh, l2mass])
            for _ in [
                self.logage_grid, self.logg_grid, self.logteff_grid,
                self.logl_grid
            ]
        ]
        large = 1e100
        logage_new[~np.isfinite(logage_new)] = large
        maxep = (np.isfinite(logage_new) *
                 np.arange(self.neep)[None, :]).max(axis=1)
        pos1 = np.zeros(len(logage), dtype=int)

        for i in range(len(logage)):
            pos1[i] = np.searchsorted(logage_new[i, :], logage[i]) - 1
        # this needs to be sped up

        pos2 = pos1 + 1
        bad = bad | (pos1 < 0) | ((pos1) >= (maxep - 1))
        pos1[bad] = 0
        pos2[bad] = 1
        ids = np.arange(len(logage))
        x1 = (logage - logage_new[ids, pos1]) / (logage_new[ids, pos2] -
                                                 logage_new[ids, pos1])

        ret = [
            _[ids, pos1] + x1 * (_[ids, pos2] - _[ids, pos1])
            for _ in [logg_new, logteff_new, logl_new]
        ]
        ret = {'logg': ret[0], 'logteff': ret[1], 'logl': ret[2]}
        for k in ['logg', 'logteff', 'logl']:
            ret[k][bad] = np.nan
        return ret


class Interpolator:
    def __init__(self, filts, data_prefix=None):
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
            data_prefix = utils.get_data_path()
        self.isoInt = TheoryInterpolator(data_prefix)
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
        mass, logage, feh = [np.asarray(_) for _ in [mass, logage, feh]]
        mass, logage, feh = np.broadcast_arrays(mass, logage, feh)
        shape = mass.shape
        mass, logage, feh = [np.atleast_1d(_) for _ in [mass, logage, feh]]
        maxn = int(1e4)
        curl = len(mass)
        keys = ['logg', 'logteff', 'logl']
        # split if many are asked
        if curl > maxn:
            nsplits = int(np.ceil(curl * 1. / maxn))
            rets = []
            ret1 = {}
            for i in range(nsplits):
                cursl = slice(i * maxn, (i + 1) * maxn)
                rets.append(self.isoInt(mass[cursl], logage[cursl],
                                        feh[cursl]))
            for k in keys:
                ret1[k] = np.concatenate([_[k] for _ in rets])
        else:
            ret1 = self.isoInt(mass, logage, feh)
        logg, logteff, logl = [ret1[_] for _ in keys]
        xind = np.isfinite(logl)
        av = feh * 0.
        arr = np.array([logteff[xind], logg[xind], feh[xind], av[xind]]).T
        res0 = self.bolomInt(arr)
        res = dict(logg=logg,
                   logteff=logteff,
                   logl=logl,
                   mass=mass,
                   logage=logage,
                   feh=feh)
        for k in res0:
            res[k] = np.zeros(len(logg)) - np.nan
            res[k][xind] = 4.74 - 2.5 * (logl[xind]) - res0[k]
        for k in res.keys():
            res[k] = res[k].reshape(shape)
        return res
