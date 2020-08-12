import minimint
import numpy as np
import sys


def test_install():
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
        'DECam_g', 'DECam_r', "Gaia_G_MAW", "Gaia_BP_MAWf", 'Gaia_RP_MAW',
        'Gaia_BP_DR2Rev', 'Gaia_RP_DR2Rev', 'WISE_W1', 'WISE_W2'
    ]

    # Define interpolation object
    ii = minimint.Interpolator(filters)

    massgrid = 10**np.linspace(np.log10(0.1), np.log10(10), 10000)
    logagegrid = [7, 8, 9, 10]
    fehgrid = [-1, 0]
    for feh in fehgrid:
        for lage in logagegrid:
            iso = ii(massgrid, lage, feh)
            x, y = (iso['DECam_g'] - iso['DECam_r'], iso['DECam_r'])

    for feh in fehgrid:
        for lage in logagegrid:
            iso = ii(massgrid, lage, feh)
            x, y = (iso['Gaia_BP_MAWf'] - iso['Gaia_RP_MAW'],
                    iso['Gaia_G_MAW'])

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
    x, y = (iso1['logteff'], iso1['logl'])
    x, y = (iso1['logteff'], iso1['logg'])
