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

    ii.isoInt.getLogAgeFromEEP(1, 140, -1, True)
