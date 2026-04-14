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


Manual download flow
--------------------

If you downloaded files yourself, place them into the target data directory
and run: 

.. code-block:: python

   import minimint
   minimint.prepare()

This will create everything needed for minimint.

You can get the URLs to download with ``minimint.get_eep_urls()`` and
``minimint.get_bc_urls()``.

Data location
-------------

Prepared datasets are stored by default in the data/ folder where minimint is installed, unless ``MINIMINT_DATA_PATH`` is defined.
