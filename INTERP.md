# How Minimint Interpolation Works (Current Implementation)

This document describes the interpolation logic currently implemented in `minimint`.

## Overview

`minimint.Interpolator` combines two stages:

1. `TheoryInterpolator`: maps `(mass, logage, [Fe/H]) -> (logg, logl, logteff, phase)`
2. `BCInterpolator`: maps `(logteff, logg, [Fe/H], A_V) -> bolometric corrections`, then computes magnitudes

The runtime is vectorized over stars.

## Stage 1: TheoryInterpolator

MIST tracks are tabulated on `(feh, mass, EEP)`, while user input includes `logage`, so the code first inverts age to fractional EEP.

### 1) Spatial coefficients in `(feh, mass)`

- Always compute bilinear weights (`C11..C22`) for the enclosing `(feh, mass)` cell.
- If `spatial_order=3`, also compute cubic Hermite/Catmull-Rom-style weights in `feh` and `mass` (4-point stencil per axis).

### 2) Age -> EEP inversion

- A custom binary search (`_binary_search`) is used over EEP index.
- The helper `getAge(EEP)` is evaluated at each iteration:
  - `spatial_order=1`: bilinear in `(feh, mass)` + lookup at EEP
  - `spatial_order=3`: bicubic in `(feh, mass)` + lookup at EEP, with linear fallback if any cubic stencil value is non-finite
- Once bracketing EEP indices `(eep1, eep2)` are found, the fractional position `eep_frac` is solved with `utils.solve_steffen_t(...)` using age values at `EEP-1, EEP, EEP+1, EEP+2`.

This keeps age/EEP interpolation monotonic while still enabling smooth cubic spatial behavior.

### 3) Final theory quantities at target age

For each good point:

- Build EEP samples `eep-1, eep0, eep1, eep+2`
- Evaluate spatial interpolation at those EEP samples:
  - `logg`, `logl`, `logteff`: cubic spatial (`spatial_order=3`) with linear fallback on invalid cubic stencils
  - `phase`: linear spatial interpolation
- Interpolate in EEP:
  - `logg`, `logl`, `logteff`: `utils.steffen_interp(...)`
  - `phase`: linear in EEP between `eep1` and `eep2`

### 4) Boundary helpers

- `_isvalid` uses the same age interpolation mode as runtime (`spatial_order`-aware) to avoid boundary inconsistencies.
- `getMaxMass`:
  - Binary-searches mass grid for a valid/invalid bracket
  - Refines with bisection using finite/non-finite behavior of `self(m, logage, feh)`

## Stage 2: BCInterpolator

Bolometric corrections are on a 4D regular grid `(logteff, logg, [Fe/H], A_V)`.

Current runtime uses 4D cubic interpolation:

- Per axis, compute cubic weights/indices (`_get_cubic_coeffs`)
- Evaluate tensor-product cubic interpolant (`_interpolator_4cubic`)
- Mark out-of-bounds points as `NaN`

This is smoother than linear BC interpolation and avoids piecewise-linear color artifacts along isochrones.

## Magnitude formula

For each filter `lambda`:

`M_lambda = 4.74 - 2.5 * logl - BC_lambda`

## NaNs and validity

- Points outside the physical track support are returned as `NaN`.
- Cubic paths include linear fallback when a required cubic stencil contains non-finite values.
