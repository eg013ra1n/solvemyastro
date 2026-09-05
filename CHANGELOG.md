# Changelog

All notable changes to solvemyastro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-06

First tagged release. The crate has been developed as a submodule of Athenaeum
and consumed by commit; this tag names the state Athenaeum v0.5.5 ships, so a
release checkout points at a version rather than a bare SHA. Everything below
is what the crate does today, not a list of changes against a previous tag —
there was none.

### Solving

- `solve()` — blind and hinted astrometric solving in-process, returning a
  `SolveSolution` with a full WCS. No external binary, no subprocess. Hints
  (`SolveHints`) narrow the search by position, radius and either field of view
  or pixel scale; without them the solver walks a field-of-view ladder.
- **SIP distortion**, fit in pixel space with annealed sigma-clipped weighted
  least squares on the precise profile. Sub-pixel residuals on 13 of the 14
  bench frames.
- **Precise profile**, opt-in: PSF-refined centroids (`CentroidRefinement`), a
  gated iterative affine refit and a PROSAC RANSAC robust seed, with a
  refinement-helped guard so the precise path can never return a worse
  solution than the fast one.
- Failure is reported, not guessed at: `SolveFailure` / `FailureClass` /
  `BestAttempt` say which stage refused and how close the best trial came, and
  a residual vector-field dump separates centroid error from uncorrected
  distortion.

### Star selection

- Selection is separate from detection — detection stays in `rustafits`. This
  crate applies an SNR gate, ranks by SNR and caps the pool, because extended
  nebula and galaxy knots carry large flux at low SNR and would otherwise
  crowd the top of a flux-ranked list.
- **Streaks are not quad material.** Detections above `MAX_ECCENTRICITY` (0.8)
  are dropped before ranking, in every selector. Healthy frames and
  trailed-but-solvable ones carry 8–10 % above that line; hopeless ones carry
  98–99 %.
- A frame the cut empties — 90 % or more of at least 100 detections removed —
  is refused immediately, naming the count, instead of clearing the four-star
  minimum with a handful of survivors and walking the whole field-of-view
  ladder twice to reach the same answer minutes later.

### Catalog

- `.smac` star cache built from a local Gaia DR3 mirror (`build-cache`),
  memory-mapped at solve time, with HEALPix cone queries.
- **Additive density tiers** (`Caches::Layered`, `LayeredCones`): a deeper tier
  adds stars to a shallower one instead of replacing it, so a user who
  downloads more depth keeps what they already had.
- `build_cache_parallel` writes the cache with parallel positioned writes,
  portably across platforms.

### Registration

- `register` / `register_with_config` align one frame to another and return a
  `Registration`, with the inlier tolerance derived from the configured arcsec
  value and reflected transforms (a meridian flip) flagged rather than silently
  accepted.

### Interfaces

- CLI: `solve` (with `--json`), `build-cache`, `cache-info`.
- `tracing` as a facade only — no subscriber is installed in library code, so
  events flow to whatever the embedding process configured.

### Notes

- `rustafits` is a path dependency, so the crate builds inside the Athenaeum
  workspace layout and is not published to crates.io.
- Behaviour is held by a corpus gate (`corpus_bench`, `corpus_layered_tiers`)
  against a frozen baseline, with a seeded RNG so repeated runs are identical.
