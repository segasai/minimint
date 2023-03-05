import tempfile
import warnings
import glob
import os
import gc
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


def __bc_url(x):
    return 'https://waps.cfa.harvard.edu/MIST/BC_tables/%s.txz' % x


def __eep_url(x):
    return ('https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/' +
            'MIST_v1.2_feh_%s_afe_p0.0_vvcrit0.4_EEPS.txz') % x


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

    mets = ('m4.00,m3.50,m3.00,m2.50,m2.00,m1.75,m1.50,m1.25,' +
            'm1.00,m0.75,m0.50,m0.25,p0.00,p0.25,p0.50').split(',')

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
            writer(__bc_url(curfilt), T)
        for curmet in mets:
            writer(__eep_url(curmet), T)
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
            'The arguments must be paths to the directories with EEP \
            and bolometric corrections')
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
    grids = ['logage', 'logteff', 'logg', 'logl', 'phase']
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

        np.save(outp_prefix + '/' + get_file(k), grid)

    with open(outp_prefix + '/' + INTERP_PKL, 'wb') as fp:
        pickle.dump(dict(umass=umass, ufeh=ufeh, neep=neep), fp)
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
        self.logg_grid = np.load(prefix + '/' + get_file('logg'))
        self.logl_grid = np.load(prefix + '/' + get_file('logl'))
        self.logteff_grid = np.load(prefix + '/' + get_file('logteff'))
        self.logage_grid = np.load(prefix + '/' + get_file('logage'))
        self.phase_grid = np.load(prefix + '/' + get_file('phase'))

        with open(prefix + '/' + INTERP_PKL, 'rb') as fp:
            D = pickle.load(fp)
            self.umass = np.array(D['umass'])
            self.ufeh = np.array(D['ufeh'])
            self.neep = D['neep']

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
        C11, C12, C21, C22 = (DD['C11'], DD['C12'], DD['C21'], DD['C22'])
        l1feh, l2feh, l1mass, l2mass = (DD['l1feh'], DD['l2feh'], DD['l1mass'],
                                        DD['l2mass'])
        eep1, eep2, eep_frac, bad = (DD['eep1'], DD['eep2'], DD['eep_frac'],
                                     DD['bad'])
        good = ~bad
        (C11_good, C12_good, C21_good, C22_good, l1feh_good, l2feh_good,
         l1mass_good, l2mass_good, eep1_good, eep2_good, eep_frac_good) = [
             _[good] for _ in [
                 C11, C12, C21, C22, l1feh, l2feh, l1mass, l2mass, eep1, eep2,
                 eep_frac
             ]
         ]
        DD = {
            'logg': self.logg_grid,
            'logteff': self.logteff_grid,
            'logl': self.logl_grid,
            'phase': self.phase_grid
        }
        xret = {}
        for curkey, curarr in DD.items():
            curr = []
            for j, cureep in enumerate([eep1_good, eep2_good]):
                curr.append(
                    _interpolator(curarr, C11_good, C12_good, C21_good,
                                  C22_good, l1feh_good, l2feh_good,
                                  l1mass_good, l2mass_good, cureep))
            xret[curkey] = curr[0] + eep_frac_good * (curr[1] - curr[0])
            # perfoming the linear interpolation with age
            # the formula is (1-eep_frac) * V_left +  eep_frac V_right
            # so V_left  + eep_frac * (V_right - V_left)

        ret = {}
        for k in ['logg', 'logteff', 'logl', 'phase']:
            ret[k] = np.zeros(N) + np.nan
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
        neep = 1710
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
        C11, C12, C21, C22 = _get_polylin_coeff(feh, self.ufeh, mass,
                                                self.umass, l1feh, l2feh,
                                                l1mass, l2mass)

        goodsel = ~bad

        def getAge(cureep):
            return _interpolator(self.logage_grid, C11[goodsel], C12[goodsel],
                                 C21[goodsel], C22[goodsel], l1feh[goodsel],
                                 l2feh[goodsel], l1mass[goodsel],
                                 l2mass[goodsel], cureep)

        ret_logage = np.zeros_like(mass)
        logage1 = getAge(eep1)
        logage2 = getAge(eep2)
        # these are boundaries in the age grid
        ret_logage[goodsel] = logage1 * (1 - eep_frac) + (eep_frac) * logage2
        if returnJac:
            jac = mass * 0
            jac[goodsel] = logage2 - logage1
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
            R = self._get_eep_coeffs(self.umass[ix], logage, feh)
            eep = R['eep1']
            bad = R['bad'][0]
            phase = max(self.phase_grid[R['l1feh'], R['l1mass'], eep],
                        self.phase_grid[R['l2feh'], R['l1mass'], eep],
                        self.phase_grid[R['l1feh'], R['l2mass'], eep],
                        self.phase_grid[R['l2feh'], R['l2mass'], eep])
            # this is a max phase among interpolation box vertices
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
        # The algorithm is the following
        # we first go over the mass grid find the right one
        # by binary search
        # Then we zoom in on that interval and use the fact that inside
        # that interval we'll have linear dependence of age on mass
        logage, feh = np.float64(logage), np.float64(feh)
        # ensure 64bit float otherwise incosistencies in float computations
        # will kill us
        niter = 40
        im1 = 0
        # self.umass[1]
        im2 = len(self.umass) - 1  # self.umass[-1]
        l1feh = np.searchsorted(self.ufeh, feh) - 1
        if self._isvalid(self.umass[im2], logage, feh, l1feh=l1feh):
            return self.umass[im2]
        for i in range(niter):
            curm = (im1 + im2) // 2
            good = self._isvalid(self.umass[curm], logage, feh, l1feh=l1feh)
            if not good:
                im1, im2 = im1, curm
            else:
                im1, im2 = curm, im2
            if im2 - im1 == 1:
                break
        ret = self._getMaxMassBox(logage, feh, l1feh, l1feh + 1, im1, im2)
        if not (np.isfinite(ret)):
            return self.umass[im1]  # the edge
        else:
            return ret * (1 - 1e-10)

    def _get_eep_coeffs(self, mass, logage, feh):
        """
        This function gets all the necessary coefficients for the interpolation
The interpolation is done in two stages:
1) Bilinear integration over mass, feh with coefficients C11,C12,C21,C22
2) Then there is a final interpolation over EEP axis
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

        C11, C12, C21, C22 = _get_polylin_coeff(feh, self.ufeh, mass,
                                                self.umass, l1feh, l2feh,
                                                l1mass, l2mass)

        def getAge(cureep, subset):
            return _interpolator(self.logage_grid, C11[subset], C12[subset],
                                 C21[subset], C22[subset], l1feh[subset],
                                 l2feh[subset], l1mass[subset], l2mass[subset],
                                 cureep)

        lefts, rights, bads = _binary_search(bads, logage, self.neep, getAge)
        LV = np.zeros(len(mass))
        RV = LV + 1
        LV[~bads] = getAge(lefts[~bads], ~bads)
        RV[~bads] = getAge(rights[~bads], ~bads)
        eep_frac = (logage - LV) / (RV - LV)
        # eep_frac is the coefficient for interpolation in EEP axis
        # 0<=eep_frac<1
        # eep1 is the position in the EEP axis (essentially floor(EEP))
        # 0<=eep1<neep
        # eep_frac is zero if the pt is close to left edge, and one if close to
        # right edge, so interpolation then needs to be done as
        # (1-eep_frac) * V_left + eep_frac * V_right
        return dict(C11=C11,
                    C12=C12,
                    C21=C21,
                    C22=C22,
                    eep_frac=eep_frac,
                    bad=bads,
                    l1feh=l1feh,
                    l2feh=l2feh,
                    l1mass=l1mass,
                    l2mass=l2mass,
                    eep1=lefts,
                    eep2=rights)

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
        C11, C12, C21, C22 = _get_polylin_coeff(feh, self.ufeh, mass,
                                                self.umass, l1feh, l2feh,
                                                l1mass, l2mass)

        # we want to find there is a point i in the age grid
        # where grid[i]<=logage<grid[i+1]
        # and grid[i+1] is not nan
        # proceed by invariant grid[l]<=X and (NOT grid[r]<=X)
        # the rhs condition can be satistfied by either grid[r]>X
        # or grid[r] is not finite
        i1, i2 = 0, self.neep - 1

        def getAge(cureep):
            return _interpolator(self.logage_grid, C11, C12, C21, C22, l1feh,
                                 l2feh, l1mass, l2mass, cureep)

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
        # if I'm here that means
        # grid[i1]<=logage and (grid[i2]> logage or grid[i2] is nan)
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
