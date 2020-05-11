To create an interpolator 

ii = mist_interpolator.FullInterpolator(['DECam_g','DECam_r'],'data/')          

Then you just call it 

ii(mass, logage,feh)

And you get a dictionary with photometry and logg,logteff,logl


to prepare you need the folder with EEPs 
and folder with unzipped bolometric correction
mist_interpolation.prepare('folder_with EEPS', 'data/'x)
bolom.prepare('mist_bolom/','data/') 