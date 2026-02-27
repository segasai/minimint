import minimint
import numpy as np
import os


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

    ii.isoInt.getLogAgeFromEEP(1, 140, -1, returnJac=True)


def test_v25_optional():
    if os.environ.get('MIST_V25_TEST') is None:
        return
    minimint.download_and_prepare(grid_version="2.5",
                                  feh_values=[0.0],
                                  afe_values=[0.0],
                                  filters=['DECam'])
    ii = minimint.Interpolator(['DECam_g'], grid_version="2.5")
    ii(1.0, 9.0, 0.0, afe=0.0)


def test_linear_mode_compatibility(tmp_path):
    from minimint.mist_interpolator import INTERP_NPZ, get_file
    from minimint.bolom import POINTS_NPY, FILT_NPY

    umass = np.array([0.8, 1.2, 1.6])
    ufeh = np.array([-1.0, 0.0, 1.0])
    uafe = np.array([0.0])
    neep = 6

    shape = (len(ufeh), len(umass), neep)
    ii, jj, kk = np.indices(shape)
    logage = 7.0 + 0.2 * ii + 0.1 * jj + 0.3 * kk
    logteff = 3.55 + 0.03 * ii + 0.04 * jj + 0.01 * kk
    logg = 4.7 - 0.08 * jj + 0.015 * kk + 0.01 * ii
    logl = (ii - 1.0)**2 + 0.5 * (jj - 1.0)**2 + 0.05 * kk
    phase = 0.1 * kk

    np.save(tmp_path / get_file('logage'), logage)
    np.save(tmp_path / get_file('logteff'), logteff)
    np.save(tmp_path / get_file('logg'), logg)
    np.save(tmp_path / get_file('logl'), logl)
    np.save(tmp_path / get_file('phase'), phase)
    np.savez(tmp_path / INTERP_NPZ,
             umass=umass,
             ufeh=ufeh,
             uafe=uafe,
             neep=neep,
             grid_ndim=3,
             grid_version='1.2',
             vvcrit=0.4)

    u0 = np.array([3.50, 3.70, 3.90])
    u1 = np.array([3.50, 4.00, 5.00])
    u2 = np.array([-1.0, 0.0, 1.0])
    u3 = np.array([0.0, 0.5, 1.0])
    g0, g1, g2, g3 = np.meshgrid(u0, u1, u2, u3, indexing='ij')
    vec = np.array([g0.ravel(), g1.ravel(), g2.ravel(), g3.ravel()])
    bc = (g0**2 + 0.3 * g1 + 0.5 * g2 + 0.2 * g3).ravel()
    np.save(tmp_path / POINTS_NPY, vec)
    np.save(tmp_path / (FILT_NPY % 'TEST_F'), bc)

    ii_linear = minimint.Interpolator(['TEST_F'],
                                      data_prefix=str(tmp_path),
                                      linear=True)
    ii_cubic = minimint.Interpolator(['TEST_F'],
                                     data_prefix=str(tmp_path),
                                     linear=False)

    par = dict(mass=1.05, logage=7.95, feh=-0.35)
    out_linear = ii_linear(**par)
    out_cubic = ii_cubic(**par)

    assert np.isfinite(out_linear['TEST_F'])
    assert np.isfinite(out_cubic['TEST_F'])
    assert np.abs(out_linear['logl'] - out_cubic['logl']) > 1e-6

    la, jac = ii_linear.isoInt.getLogAgeFromEEP(np.array([1.05]),
                                                np.array([2.4]),
                                                np.array([-0.35]),
                                                returnJac=True)
    assert np.isfinite(la[0])
    assert np.isfinite(jac[0])
