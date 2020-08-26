[![Build Status](https://travis-ci.com/segasai/minimint.svg?branch=master)](https://travis-ci.com/segasai/minimint)
[![Coverage Status](https://coveralls.io/repos/github/segasai/minimint/badge.svg?branch=master)](https://coveralls.io/github/segasai/minimint?branch=master)

Minimint (MIni Mist INTerpolation)

Software to do simple interpolation of MIST isochrones

Author: Sergey Koposov (2020) skoposov __AT__ ed __DOT__ ac __DOT__ uk

# Instructions 

* Install minimint  (clone the repo and do pip install) 

```
git clone https://github.com/segasai/minimint.git
pip install ./minimint
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
This will take some time (20-30 min) and will use some space (10-30Gb).


Now you can use the package. In order to create an interpolator object:

```i = minimint.Interpolator(['DECam_g','DECam_r'])```

The interpolator is a callable, so you can call it on mass, log10(age), feh 

``` ii(mass, logage,feh)``` 
 
This returns a dictionary with photometry and logg, logteff, logl and photometry.

## Examples 

See the notebook in the examples/ folder
