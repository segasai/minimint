[![Build Status](https://travis-ci.com/segasai/minimint.svg?branch=master)](https://travis-ci.com/segasai/minimint)

Minimint Software to do simple interpolation of MIST isochrones
(Mini MIST Interpolation)
Author Sergey Koposov (2020) skoposov __AT__ cmu.edu

# Instructions 

* Install minimint  (clone the repo and do pip install) 
* Download and unpack EEP track files from MIST and the bolometric corrections from http://waps.cfa.harvard.edu/MIST/model_grids.html
* Create the necessary preprocessed files by running 

``` minimint.prepare('FOLDER_WITH_EEP', 'FOLDER_WITH_BC')```

That will process the evolutionary tracks and bolometric corrections by creating the necessary 
files for the package.

Now you can use the package 

To create an interpolator 

```i = minimint.Interpolator(['DECam_g','DECam_r'])```

Then you just call it 
``` ii(mass, logage,feh)``` 
 
And you get a dictionary with photometry and logg,logteff,logl

## Examples 

See the notebook in the examples/ folder
