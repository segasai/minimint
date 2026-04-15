Manual Download and Prepare
===========================

If you prefer to download MIST data manually, minimint can provide URL lists. You can then use these downloaded files to
prepare the isochrones for the interpolators.

Get archive URLs
----------------

.. code-block:: python

   import minimint

   bc_urls = minimint.get_bc_urls(['DECam', 'WISE'], mist_version='2.5')
   eep_urls = minimint.get_eep_urls(
       feh_values=[-1.0, -0.5, 0.0],
       afe_values=[0.0, 0.2, 0.4],
       mist_version='2.5',
       vvcrit=0.4,
   )

Prepare unpacked files
----------------------

Here we assume that you have downloaded the EEPs and bolometric correction files and unpacked them.

.. code-block:: python

   import minimint

   minimint.prepare(
       eep_prefix='/path/to/unpacked/eep_data',
       bolom_prefix='/path/to/unpacked/bc_data',
       outp_prefix='/path/to/output',
       mist_version='2.5',
       vvcrit=0.4,
   )

Bolometric-corrections only:

.. code-block:: python

   minimint.prepare(
       eep_prefix='/path/to/unpacked/data',
       bolom_prefix='/path/to/unpacked/data',
       outp_prefix='/path/to/output',
       bc_only=True,
       mist_version='2.5',
       vvcrit=0.4,
   )
