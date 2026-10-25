# Changelog

All notable changes to solvemyastro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-10-25

### Quad index

- **Precomputed quad index** (`index` module, `QuadIndex`, `quads.smaq` format
  v3) for blind and scale-free solves. An image quad is looked up directly in a
  table of catalog quads, so the cost no longer grows with the sky area or the
  scale range left to search: the 37 real bench frames solve blind in 0.2–4.7 s
  each (43 s total, 330 s without the index), every solution matching the
  hinted one.
- `solve_with_index()` takes one or more index packs. A base pack (depth 2–7,
  fields from ~1.1°, 147 MB) and an optional narrow pack (depth 8, fields from
  ~0.55°, 236 MB) can be shipped and installed separately. Fully hinted frames
  never use the index and solve exactly as with `solve()`.
- `solvemyastro build-index --tiers <dir> --out <dir>` builds a pack from the
  density-tier catalog; the output is deterministic, so a published pack is
  verified by its SHA-256. `solvemyastro solve` takes `--tiers` and a
  repeatable `--index`.
- When the index finds nothing, the spiral runs only where the index did not
  look, so a frame nothing can solve fails in ~20–30 s instead of minutes.
- `index::FORMAT_VERSION` (3) is public, for a pack manifest's `format` field.
- `QUAD_INDEX.md` documents building, shipping and wiring in the packs.

### Verification

- **Bright-star completeness gate.** After the SIP fit the brightest catalog
  stars in the frame are scored, found for the WCS and missing against it; a
  candidate below `log(1e9)` is rejected and the search continues. Two chance
  alignments on a dense 51″/px Milky Way field that used to pass now fail.
- A seed rejected by verification is re-fitted about its own centre before it
  is given up.

### Search

- The hinted ladder reaches 0.15° FOV for long-focus frames; the catalog cone
  radius is FOV/1.5 so the hinted rung matches.
- A sparse-catalog cell is balanced in the primary pass.
- Spiral chunks grow 16 → 1024 instead of a fixed 1024.

### Performance

- Grid nearest-neighbour lookup for catalog-to-detection matching.
- `PixelCache` decodes pixels only to the requested magnitude; the cone test
  is trig-free with an exact fallback at the boundary.
- The brightest cone stars are selected with `select_nth_unstable` instead of
  sorting all of them.

### Testing

- Bench frames moved out of the repository to
  `https://artfrom.space/builds/platesolve_bench/platesolve_bench.zip`;
  `scripts/fetch_bench_data.sh` downloads them with the tier catalog.
- `synth_bench` and the `synth_fields` example: synthetic star fields from the
  tier catalog, hinted, position-only, scale-only and blind.
- `_DSC2522` and `_DSC5767` join the corpus gate with known truth.

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
