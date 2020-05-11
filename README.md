[![Build Status](https://travis-ci.com/segasai/minimint.svg?branch=master)](https://travis-ci.com/segasai/minimint)

Minimint Software to do simple interpolation of MIST isochrones
(Mini MIST Interpolation)

# Preparation 

* Install minimint  (clone the repo and do pip install) 
* Download the EEP files from MIST and the bolometric corrections
* Create the necessary preprocessed files by running 

``` minimint.prepare('FOLDER_WITH_EEP', 'FOLDER_WITH_BC')```
That will process the evolutionary tracks and bolometric corrections

Now you can use it. 

To create an interpolator 

```i = minimint.Interpolator(['DECam_g','DECam_r'])```

Then you just call it 
``` ii(mass, logage,feh)``` 

And you get a dictionary with photometry and logg,logteff,logl

