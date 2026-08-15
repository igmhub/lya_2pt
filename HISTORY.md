# History

## Unreleased

## v0.3.0 — 2026-08-14

### Added

- Add `r-mu` coordinate grids for correlation and distortion calculations,
  including INI configuration, FITS export metadata, documentation, and
  regression coverage.

### Fixed

- Restore specialized scalar rp/rt Numba kernels and add dedicated r/mu
  kernels to retain legacy rp/rt performance.
