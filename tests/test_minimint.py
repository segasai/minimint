import minimint
import numpy as np
import os
import pytest
from minimint import utils as mm_utils


def test_install():
    if os.environ.get('LOCAL_TESTING') is not None:
        pass
    else:
        minimint.download_and_prepare()


def test_run():
    ii = minimint.Interpolator(
        ['DECam_g', 'DECam_r', 'DECam_i', 'DECam_z', 'WISE_W1', 'SkyMapper_g'])
    ii(.9, 9, -3.3)
    ii(.9 + np.zeros(100), 9 + np.zeros(100), -3.3 + np.zeros(100))
    Nlarge = int(1.2e4)
    ii(.9 + np.zeros(Nlarge), 9 + np.zeros(Nlarge), -3.3 + np.zeros(Nlarge))


def test_filters():
    minimint.list_filters()


def test_example():

    filters = [
        'DECam_g', 'DECam_r', "Gaia_G_EDR3", "Gaia_BP_EDR3", 'Gaia_RP_EDR3',
        'WISE_W1', 'WISE_W2'
    ]

    # Define interpolation object
    ii = minimint.Interpolator(filters)

    massgrid = 10**np.linspace(np.log10(0.1), np.log10(10), 10000)
    logagegrid = [7, 8, 9, 10]
    fehgrid = [-2, -1, 0]
    for feh in fehgrid:
        for lage in logagegrid:
            iso = ii(massgrid, lage, feh)
            x, y = (iso['DECam_g'] - iso['DECam_r'], iso['DECam_r'])
            mass = ii.getMaxMass(lage, feh)
            massMS = ii.getMaxMassMS(lage, feh)
            assert (massMS < mass)
            assert (np.isfinite(ii(mass, lage, feh)['DECam_g']))
            assert (not np.isfinite(ii(mass + 1e-6, lage, feh)['DECam_g']))

    for feh in fehgrid:
        for lage in logagegrid:
            iso = ii(massgrid, lage, feh)
            x, y = (iso['Gaia_BP_EDR3'] - iso['Gaia_RP_EDR3'],
                    iso['Gaia_G_EDR3'])
            del x, y

    # Compute the evolutionary track

    lagegrid = np.linspace(5, 10.1, 100000)

    mass = 1.04
    iso1 = ii(mass, lagegrid, -1)
    mass = 5.04
    iso2 = ii(mass, lagegrid, -1)
    (iso1['DECam_g'] - iso1['DECam_r'], iso1['DECam_r'])
    (iso2['DECam_g'] - iso2['DECam_r'], iso2['DECam_r'])

    lagegrid = np.linspace(5, 10., 50000)
    mass = 1.04
    iso1 = ii(mass, lagegrid, -1)
    (iso1['logteff'], iso1['logl'])

    # check high metallicity edge
    mass = 0.5
    logagegrid = [7, 8, 9, 10]
    for lage in logagegrid:
        for feh in [-3.85, .45]:
            iso = ii(mass, lage, feh)
            assert (np.isfinite(iso['DECam_g']))

    ii.isoInt.getLogAgeFromEEP(1, 140, -1, True)


@pytest.fixture(scope='module')
def cubic_interp():
    return minimint.Interpolator(['DECam_g'], spatial_order=3)


def test_cubic_age_eep_monotonic(cubic_interp):
    ii = cubic_interp
    eep = np.arange(0, ii.isoInt.neep - 1, 4, dtype=float)
    cases = [(0.9, -2.0), (1.1, -1.0), (1.5, -0.5)]
    for mass, feh in cases:
        ages = ii.isoInt.getLogAgeFromEEP(np.full_like(eep, mass), eep,
                                          np.full_like(eep, feh))
        good = np.isfinite(ages)
        # Age as a function of EEP must stay monotonic for binary search.
        diff = np.diff(ages[good])
        assert np.min(diff) >= -1e-10


def test_cubic_getmaxmass_boundary_consistency(cubic_interp):
    ii = cubic_interp
    cases = [(-3.5, 8.033333333333333), (-2.0, 9.0), (-1.0, 10.0),
             (0.0, 9.5)]
    for feh, lage in cases:
        mass = ii.getMaxMass(lage, feh)
        assert np.isfinite(ii(mass, lage, feh)['DECam_g'])
        if mass < ii.isoInt.umass[-1] - 1e-8:
            assert (not ii.isoInt._isvalid(mass + 1e-4, lage, feh))


def test_numba_dispatch_parity():
    rng = np.random.default_rng(12345)
    has_numba_orig = mm_utils.HAS_NUMBA
    try:
        # 2D kernel parity (2x2 path)
        n = 256
        grid2 = rng.normal(size=(7, 9, 11))
        ie = rng.integers(0, 11, size=n)
        i0 = rng.integers(0, 7, size=(n, 2))
        i1 = rng.integers(0, 9, size=(n, 2))
        w0 = rng.random((n, 2))
        w1 = rng.random((n, 2))
        w0 /= w0.sum(axis=1, keepdims=True)
        w1 /= w1.sum(axis=1, keepdims=True)

        mm_utils.HAS_NUMBA = False
        y_np = mm_utils._interpolator_2d(grid2, w0, i0, w1, i1, ie)
        mm_utils.HAS_NUMBA = True
        y_nb = mm_utils._interpolator_2d(grid2, w0, i0, w1, i1, ie)
        np.testing.assert_allclose(y_np, y_nb, rtol=1e-12, atol=1e-12)

        # 2D kernel parity (4x4 path)
        i0 = rng.integers(0, 7, size=(n, 4))
        i1 = rng.integers(0, 9, size=(n, 4))
        w0 = rng.random((n, 4))
        w1 = rng.random((n, 4))
        w0 /= w0.sum(axis=1, keepdims=True)
        w1 /= w1.sum(axis=1, keepdims=True)

        mm_utils.HAS_NUMBA = False
        y_np = mm_utils._interpolator_2d(grid2, w0, i0, w1, i1, ie)
        mm_utils.HAS_NUMBA = True
        y_nb = mm_utils._interpolator_2d(grid2, w0, i0, w1, i1, ie)
        np.testing.assert_allclose(y_np, y_nb, rtol=1e-12, atol=1e-12)

        # 4D kernel parity
        grid4 = rng.normal(size=(6, 7, 8, 5))
        i0 = rng.integers(0, 6, size=(n, 4))
        i1 = rng.integers(0, 7, size=(n, 4))
        i2 = rng.integers(0, 8, size=(n, 4))
        i3 = rng.integers(0, 5, size=(n, 4))
        w0 = rng.random((n, 4))
        w1 = rng.random((n, 4))
        w2 = rng.random((n, 4))
        w3 = rng.random((n, 4))
        w0 /= w0.sum(axis=1, keepdims=True)
        w1 /= w1.sum(axis=1, keepdims=True)
        w2 /= w2.sum(axis=1, keepdims=True)
        w3 /= w3.sum(axis=1, keepdims=True)

        mm_utils.HAS_NUMBA = False
        y_np = mm_utils._interpolator_4d(grid4, w0, i0, w1, i1, w2, i2, w3, i3)
        mm_utils.HAS_NUMBA = True
        y_nb = mm_utils._interpolator_4d(grid4, w0, i0, w1, i1, w2, i2, w3, i3)
        np.testing.assert_allclose(y_np, y_nb, rtol=1e-12, atol=1e-12)

        # End-to-end parity (small sample)
        ii = minimint.Interpolator(['DECam_g', 'DECam_r'], spatial_order=3)
        m = np.linspace(0.1, 1.2, 400)
        a = np.full_like(m, 7.0)
        f = np.zeros_like(m)

        mm_utils.HAS_NUMBA = False
        out_np = ii(m, a, f)
        mm_utils.HAS_NUMBA = True
        out_nb = ii(m, a, f)

        for k in ['logg', 'logteff', 'logl', 'phase', 'DECam_g', 'DECam_r']:
            np.testing.assert_allclose(out_np[k], out_nb[k], rtol=1e-10,
                                       atol=1e-10, equal_nan=True)
    finally:
        mm_utils.HAS_NUMBA = has_numba_orig
