# History

## Unreleased

### Added

- Propagate DESI blinding metadata through correlation and distortion products,
  and apply the ``desi_dr3`` correlation template during rp/rt export.

## v0.3.0 — 2026-08-14

### Added

- Add `r-mu` coordinate grids for correlation and distortion calculations,
  including INI configuration, FITS export metadata, documentation, and
  regression coverage.

### Fixed

- Restore specialized scalar rp/rt Numba kernels and add dedicated r/mu
  kernels to retain legacy rp/rt performance.
