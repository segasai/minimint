Examples
========

This page mirrors the main workflows from ``examples/Example.ipynb`` as plain
Python snippets.

Setup
-----

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import minimint

Basic isochrone operations
------------------------

.. code-block:: python

   filters = ['DECam_g', 'DECam_r', 'Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3',
              'WISE_W1', 'WISE_W2']
   ii = minimint.Interpolator(filters, interp_mode='cubic')

   massgrid = 10 ** np.linspace(np.log10(0.1), np.log10(10), 100000)
   lage = 10 # log10(age)
   feh = -1

   iso = ii(massgrid, lage, feh)
   plt.plot(iso['DECam_g'] - iso['DECam_r'], iso['DECam_r'])

   plt.xlabel('g-r')
   plt.ylabel('r')
   plt.ylim(20, -5)


Use maximum valid mass per isochrone
------------------------------------

.. code-block:: python

   lage = 9.0
   feh = -1.0
   maxmass = ii.getMaxMass(lage, feh)
   minmass = 0.1
   massgrid = maxmass - (maxmass - minmass) * 10 ** np.linspace(-5, 0, 1000)
   iso = ii(massgrid, lage, feh)

MIST 2.5 with alpha enhancement
-------------------------------

.. code-block:: python

   ii25 = minimint.Interpolator(filters, interp_mode='cubic', mist_version='2.5')

   feh = -3
   afe = 0.1

   iso = ii25(massgrid, 10.0, feh, afe=afe)
   sel = iso['phase'] < 5
   plt.plot((iso['DECam_g'] - iso['DECam_r'])[sel], iso['DECam_r'][sel])
