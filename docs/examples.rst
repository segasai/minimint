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

   print(minimint.list_filters())

Basic isochrone plotting
------------------------

.. code-block:: python

   filters = ['DECam_g', 'DECam_r', 'Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3',
              'WISE_W1', 'WISE_W2']
   ii = minimint.Interpolator(filters, interp_mode='cubic')

   massgrid = 10 ** np.linspace(np.log10(0.1), np.log10(10), 100000)
   logagegrid = [7, 8, 9, 10]
   fehgrid = [-1, 0]

   for feh in fehgrid:
       for lage in logagegrid:
           iso = ii(massgrid, lage, feh)
           plt.plot(iso['DECam_g'] - iso['DECam_r'], iso['DECam_r'])

   plt.xlabel('g-r')
   plt.ylabel('r')
   plt.ylim(20, -5)

Evolutionary tracks
-------------------

.. code-block:: python

   lagegrid = np.linspace(5, 10.1, 100000)
   iso1 = ii(1.04, lagegrid, -1)
   iso2 = ii(5.04, lagegrid, -1)

   plt.plot(iso1['DECam_g'] - iso1['DECam_r'], iso1['DECam_r'], '.')
   plt.plot(iso2['DECam_g'] - iso2['DECam_r'], iso2['DECam_r'], '.')
   plt.ylim(5, -5)

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

   massgrid = 10 ** np.linspace(np.log10(0.1), np.log10(10), 100000)
   fehgrid = [-4.0, -3.0, -2.0, -1.0, 0.0]
   afegrid = [-0.2 + 0.01, 0.0, 0.2, 0.4, 0.6 - 0.01]

   for feh in fehgrid:
       for afe in afegrid:
           iso = ii25(massgrid, 10.0, feh, afe=afe)
           sel = iso['phase'] < 5
           plt.plot((iso['DECam_g'] - iso['DECam_r'])[sel], iso['DECam_r'][sel])
