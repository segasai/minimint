# Version 0.6.0
* switch to cubic interpolation for isochrones and bolometric corrections
* add `linear=True` option to keep legacy poly-linear interpolation behavior
* improved robustness at grid boundaries

# Version 0.5.0
* switch to pyproject.toml
* allow to chose v/vcrit value when fetching MIST tracks
* fix getLogAgeFromEEP which was probably failing if some points were incorrect
