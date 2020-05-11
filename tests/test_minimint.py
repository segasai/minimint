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

def test_filters():
    minimint.list_filters()
