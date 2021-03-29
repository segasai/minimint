[![Build Status](https://github.com/segasai/minimint/workflows/Minimint/badge.svg)](https://github.com/segasai/minimint/actions?query=workflow%3AMinimint)
[![Coverage Status](https://coveralls.io/repos/github/segasai/minimint/badge.svg?branch=master)](https://coveralls.io/github/segasai/minimint?branch=master)

Minimint (MIni Mist INTerpolation)

Software to do simple interpolation of MIST isochrones.

Author: Sergey Koposov (2020-2021) skoposov __AT__ ed __DOT__ ac __DOT__ uk

# Instructions 

* Install minimint  (clone the repo and do pip install) 

```
pip install minimint
```

* Download and prepare isochrone files 
```
minimint.download_and_prepare()
```

That will download everything and process the evolutionary tracks and bolometric corrections by creating the necessary  files for the package.

minimint.download_and_prepare() by by default creates bolometric corrections for these filters
'DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper','UBVRIplus', 'WISE'
If you need additional filters, you can specify them using the filters parameter

``` minimint.download_and_prepare(filters=['JWST','WISE','DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper','UBVRIplus'])```
Check which filters are available on the MIST website http://waps.cfa.harvard.edu/MIST/model_grids.html
This will take some time (20-30 min) and will use some space (10-30 Gb).


Now you can use the package. In order to create an interpolator object:

```i = minimint.Interpolator(['DECam_g','DECam_r'])```

The interpolator is a callable, so you can call it on mass, log10(age), feh 

``` ii(mass, logage,feh)``` 
 
This returns a dictionary with photometry and logg, logteff, logl and photometry.

You also can use the interpolator to find the maximum valid mass on the isochrone.

```ii.getMaxMass(logage, feh)```

## Examples 

See the [notebook](examples/Example.ipynb) in the examples folder

## Synthetic stellar populations
If you are interested in synthetic stellar populations you will need
the implementation of the IMF. For this you may want to use https://github.com/keflavich/imf
