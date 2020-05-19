import minimint
import numpy as np
import sys

def test_install():
    minimint.prepare('EEPS', 'BCS')


def test_run():
    ii = minimint.Interpolator(
        ['DECam_g', 'DECam_r', 'DECam_i', 'DECam_z', 'WISE_W1', 'SkyMapper_g'])
    ii(.9, 9, -3.3)
    ii(.9 + np.zeros(100), 9 + np.zeros(100), -3.3 + np.zeros(100))
    Nlarge = int(1.2e4)
    ii(.9 + np.zeros(Nlarge), 9 + np.zeros(Nlarge), -3.3 + np.zeros(Nlarge))

def test_filters():
    minimint.list_filters()
