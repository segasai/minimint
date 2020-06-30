[![Build Status](https://travis-ci.com/segasai/minimint.svg?branch=master)](https://travis-ci.com/segasai/minimint)
[![Coverage Status](https://coveralls.io/repos/github/segasai/minimint/badge.svg?branch=master)](https://coveralls.io/github/segasai/minimint?branch=master)

Minimint (MIni Mist INTerpolation)

Software to do simple interpolation of MIST isochrones

Author: Sergey Koposov (2020) skoposov __AT__ cmu.edu

# Instructions 

* Install minimint  (clone the repo and do pip install) 
* Download and unpack EEP track files from MIST and the bolometric corrections from http://waps.cfa.harvard.edu/MIST/model_grids.html
* Create the necessary preprocessed files by running 

``` minimint.prepare('FOLDER_WITH_EEP', 'FOLDER_WITH_BC')```

That will process the evolutionary tracks and bolometric corrections by creating the necessary 
files for the package.

Now you can use the package. In order to create an interpolator object:

```i = minimint.Interpolator(['DECam_g','DECam_r'])```

The interpolator is a callable, so you can call it on mass, log10(age), feh 

``` ii(mass, logage,feh)``` 
 
This returns a dictionary with photometry and logg, logteff, logl and photometry.

## Examples 

See the notebook in the examples/ folder
