Quickstart
==========

Download and prepare MIST isochrones

.. code-block:: python

   import minimint
   minimint.download_and_prepare()

Create an interpolator and evaluate an isochrone:

.. code-block:: python

   import numpy as np
   import minimint

   filters = ['DECam_g', 'DECam_r', 'Gaia_G_EDR3']
   ii = minimint.Interpolator(filters, interp_mode='cubic', mist_version='1.2')

   mass = 10 ** np.linspace(np.log10(0.1), np.log10(10.0), 2000)
   out = ii(mass, logage=9.0, feh=-1.0)

``out`` is a dictionary with theoretical quantities (``logg``, ``logteff``,
``logl``, ``phase``) and requested filters.

You can also query isochrone bounds:

.. code-block:: python

   mmax = ii.getMaxMass(logage=9.0, feh=-1.0)

Version note:

- ``mist_version='1.2'`` uses the v1.2 grid (no alpha-enhanced dimension).
- ``mist_version='2.5'`` uses the v2.5 grid and supports alpha-enhanced
  isochrones via the ``afe`` argument.
