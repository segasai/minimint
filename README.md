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

* Download and unpack EEP track files from MIST and the bolometric corrections for filter systems that you need from http://waps.cfa.harvard.edu/MIST/model_grids.html
i.e. 
```
wget http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_UBVRIplus.txz
wget http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_DECam.txz
...
wget http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_feh_m4.00_afe_p0.0_vvcrit0.4_EEPS.txz
wget http://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_feh_m3.50_afe_p0.0_vvcrit0.4_EEPS.txz
...
```

* Create the necessary preprocessed files by running 

``` minimint.prepare('FOLDER_WITH_EEP', 'FOLDER_WITH_BC')```

That will process the evolutionary tracks and bolometric corrections by creating the necessary  files for the package.

minimint.prepare by default creates bolometric corrections for these 
'DECam', 'GALEX', 'PanSTARRS', 'SDSSugriz', 'SkyMapper','UBVRIplus', 'WISE'
If you need additional filters, you can specify them using the filters parameter

``` minimint.prepare('FOLDER_WITH_EEP', 'FOLDER_WITH_BC', filters=['JWST','WISE'])```


Now you can use the package. In order to create an interpolator object:

```i = minimint.Interpolator(['DECam_g','DECam_r'])```

The interpolator is a callable, so you can call it on mass, log10(age), feh 

``` ii(mass, logage,feh)``` 
 
This returns a dictionary with photometry and logg, logteff, logl and photometry.

## Examples 

See the notebook in the examples/ folder
