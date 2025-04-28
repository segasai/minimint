[![Build Status](https://github.com/segasai/minimint/workflows/Minimint/badge.svg)](https://github.com/segasai/minimint/actions?query=workflow%3AMinimint)
[![Coverage Status](https://coveralls.io/repos/github/segasai/minimint/badge.svg?branch=master)](https://coveralls.io/github/segasai/minimint?branch=master)[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5610692.svg)](https://doi.org/10.5281/zenodo.5610692)


# Minimint (MIni Mist INTerpolation)

This is a python package to perform interpolation of MIST isochrones. It is simple, in the sense that it only does one thing - predict the stellar parameters ($\log g$, $T_{eff}$, $\log L$) and photometry in a given filter for a given mass, age and metallicity. It is also vectorized, so computing photometry for large number of stars, should not be slow.

Author: Sergey Koposov (2020-2025) skoposov __AT__ ed __DOT__ ac __DOT__ uk

## Instructions 

* Install minimint. You can either pip install the released version or install from github

```
pip install minimint
```

* Download and prepare isochrone files 
```
minimint.download_and_prepare()
```

That will download all the MIST files, and process the evolutionary tracks and bolometric corrections by creating the necessary files for the package.

`minimint.download_and_prepare()` by default creates bolometric corrections for these filters
'DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper','UBVRIplus', 'WISE'
If you need additional filters, you can specify them using the filters parameter

``` minimint.download_and_prepare(filters=['JWST','WISE','DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper','UBVRIplus'])```
Check which filters are available on the MIST website http://waps.cfa.harvard.edu/MIST/model_grids.html
The downloading will take some time (20-30 min) and will use significant amount of disk space (10-30 Gb).

If you want to put those processed isochrone files in a location different from the site-packages folder of minimint, you can use the outp_prefix parameter of `download_and_prepare`. You then will need to either specify the location each time when you construct the interpolators or with the MINIMINT_DATA_PATH environment variable.

## Usage 

In order to create an interpolator object for two filters (your can provide a list of any numbers of filters)

```ii = minimint.Interpolator(['DECam_g','DECam_r'])```

The interpolator is a callable, so you can call it on mass, log10(age), feh 

``` ii(mass, logage, feh)``` 
 
This returns a dictionary with photometry, logg, logteff and logl.
mass, logage and feh could arrays. In this case the result will be dictionary of arrays.

You also can use the interpolator to find the maximum valid mass on the isochrone.

```ii.getMaxMass(logage, feh)```

## Examples 

See the [notebook](examples/Example.ipynb) in the examples folder

## Synthetic stellar populations
If you are interested in synthetic stellar populations you will need
an implementation of the IMF. For this you may want to use https://github.com/keflavich/imf

## Acknowledgement

If you are using this package please cite it through zenodo link
https://doi.org/10.5281/zenodo.4002971
Also make sure that you acknowledge MIST isochrones https://waps.cfa.harvard.edu/MIST/
