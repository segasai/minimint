Interpolation Modes
===================

Use ``interp_mode`` to select spatial interpolation:

- ``'linear'``: more conservative, often more robust on sparse/coarse grids.
- ``'cubic'``: smoother interpolation, but can overshoot locally on sparse grids.

.. code-block:: python

   import minimint

   ii_linear = minimint.Interpolator(['DECam_g', 'DECam_r'], interp_mode='linear')
   ii_cubic = minimint.Interpolator(['DECam_g', 'DECam_r'], interp_mode='cubic')

The same ``interp_mode`` option is used by ``TheoryInterpolator``.
