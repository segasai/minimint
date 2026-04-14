Installation
============

Install from PyPI:

.. code-block:: bash

   pip install minimint

Or install from a local checkout:

.. code-block:: bash

   pip install /path/to/minimint

MIST versions
-------------

minimint supports two MIST grids via ``mist_version``:

- ``'1.2'``: classic MIST v1.2 grid (single ``[alpha/Fe]=0`` sequence).
- ``'2.5'``: MIST v2.5 grid, including alpha-enhanced isochrones over
  multiple ``[alpha/Fe]`` values.

Use the same ``mist_version`` consistently when downloading/preparing data and
when constructing interpolators.

Prepare isochrone data (required)
---------------------------------

After installation, download and prepare MIST data before creating an
interpolator:

.. code-block:: python

   import minimint
   minimint.download_and_prepare()

This step is required at least once for each dataset location.

Manual download flow
--------------------

If you downloaded files yourself, place them into the target data directory
and run:

.. code-block:: python

   import minimint
   minimint.prepare()

You can get source URLs with ``minimint.get_eep_urls()`` and
``minimint.get_bc_urls()``.

Data location
-------------

Prepared datasets are stored under ``MINIMINT_DATA_PATH``. If the variable is
not set, minimint uses its package-local ``data`` directory.
