# Changelog

All notable changes to solvemyastro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — unreleased

### Verification

- One decision replaces the bright-star completeness test: a Bayes factor of
  the final (SIP) WCS over the image stars, after D. Lang, D. W. Hogg, K. Mierle,
  M. Blanton, S. Roweis, The Astronomical Journal 139, 1782–1800 (2010),
  arXiv:0910.2233, §2.4. The image stars, in order of signal-to-noise and minus
  the hypothesis's own stars (the ones that chose or fitted it), are scanned
  once. Each star scores against a uniform background with a Gaussian around the
  nearest unclaimed reference star (a reference explains at most one star; a
  star beyond 3σ of every free reference is a distractor; the uniform part of
  the foreground weighs `distractor_fraction` = 0.25; σ is the matching
  tolerance, never fitted); `log10_odds` is the running maximum of the
  sum. A catalog star with no detection costs nothing: it only enlarges the
  reference count `N_R`, which is how a dense field asks for more evidence.
- The references are the verification ladder's list, deepened over G ≤ 12, 14
  and `catalog_mag_limit` until the frame holds clamp(|T|/4, 8, 100) of them on
  the final WCS (|T| the image stars), so a sparse narrow field is scored
  against a list that is not nearly empty.
- The bar is `log10_odds ≥ 9`: a prior of 10⁻⁶ for a given hypothesis and the
  cost ratio of 1999 to 1 for a false against a true positive give odds of 10⁹.
  Beyond 10⁶ hypotheses verified in one solve (all passes) the bar rises by
  `log10(H / 10⁶)`. Under the background the scan reaches 10⁹ with probability
  at most 10⁻⁹ per hypothesis (a non-negative martingale of mean 1).
- The bright-star completeness test and the gate's inlier-ratio rule are gone;
  so is the 0.3.0 known issue (the in-field count read far too low) with the
  function that had it. The seed pre-filter at ln 10⁶ and every search stage are
  unchanged.
- The decision also refuses a candidate whose final WCS matches fewer than
  `min_final_inliers` image stars (one detection per reference, the acceptance
  gate's floor), so the search goes on instead of returning a solution the gate
  would refuse.
- A solve whose best candidate scored below the bar fails with "the best match
  could be a chance alignment (odds 10^x to 1, 10^y needed)".
- `diag::best_finite_log10_odds` (crate-internal) folds the candidates' scores
  into `FailureClass::NotConfident`.

### Breaking

- `SolveSolution`: new `log10_odds`, `reference_stars_in_frame` and
  `hypotheses_verified`. `matched_stars` is now counted on the final WCS, one
  detection per reference at 1σ; `inlier_ratio` is matched over the in-frame
  reference stars, a description and no longer a quality bar.
- `SolveConfig`: new `accept_log10_odds` (9.0, the decision's bar) and
  `min_final_inliers` (6, the linear-inlier floor a candidate needs to reach the
  decision).
- `accept`: `GateMetrics` gains `log10_odds` and loses `expected_in_fov` and
  `inlier_ratio`; `Gate::min_inlier_ratio`, `DENSE_FIELD_MIN_EXPECTED` and
  `expected_bright_in_fov` are removed; `Gate::refusal` and `GateRefusal`
  (a named reason, with a `Display`) are new, and `Gate::accepts` is
  `refusal().is_none()`. Raising the odds bar changes cached verdicts: an
  embedding app bumps its solve-recipe version.
- `diag`: `FailureClass::NotConfident { best_log10_odds }` (code
  `NOT_CONFIDENT`) and `FailureClass::TooFewMatched { best_matched, floor }`
  (code `TOO_FEW_MATCHED`, the strongest candidate met the bar but matched too few
  stars), so `FailureClass` is no longer `Eq`; a failure's class and message are
  the refusal of the candidate with the highest finite score; `SolveFailure` gains
  `hypotheses_verified`, `prefilter_passed` and `verifies`; `VerifyTrace` gains
  the decision's numbers (`path`, `recentered`, `hypotheses_so_far`,
  `own_star_count`, `image_star_count`, `inliers_outside_own`, `tol_px`,
  `reference_stars`, `reference_mag_limit`, `test_stars`, `scan_matched`,
  `matched_stars`, `peak_rank`, `log10_odds`, `log10_odds_quad4`,
  `required_log10_odds`, `matched_floor`, `final_wcs`); `SearchPath` is new.
- `index::Hypothesis::star_idx`: the image stars of the hypothesis's quad.
- `verify`: `BRIGHT_MATCH_PROB`, `BRIGHT_LOG_ODDS_MIN` and
  `bright_star_log_odds` are removed; `ACCEPT_LOG10_ODDS`, `SCAN_CUTOFF_SIGMA`,
  `HYPOTHESIS_BUDGET`, `ScanInput`, `ScanResult`, `scan_log10_odds` and
  `required_log10_odds` are new.
- New entry points `solve_traced` and `solve_bounded` (the second confines the
  scale to a window, as the search's step B2 does).
- The CLI's `solve --json` keys are eleven: `ra`, `dec`, `scale`, `rot`,
  `matched`, `rms_px`, `inlier_ratio`, `log10_odds`, `reference_stars`,
  `hypotheses`, `gate`. `expected_in_fov` is gone. The text output prints the
  odds, and a refusal names the check that failed.
- `tests/corpus_layered_baseline.json` is re-frozen with the new match count,
  ratio and odds; every recorded WCS is unchanged. The legacy
  `corpus_baseline.json` keeps 0.3.0's meanings (it needs a deep star cache this
  release was not built with) unless it is regenerated.

## [0.3.0] — 2026-11-26

### Catalog building moves here

- The Gaia catalog builder leaves Athenaeum: `gaia` (ingest of the
  `GaiaSource_*.csv.gz` files into bins, with a coverage record and an optional
  `_MD5SUM.txt` check), `bins`, `tiers` (the four density tiers), `package` (the
  publish tree), `manifest`, `layout`, `healpix`, `accept` (the acceptance gate)
  and the index presets (`index::Preset`, `open_preset_layers`) are modules of
  this crate. Ingest, tiers and packages are behind the `catalog` feature; the
  base layer has none of their dependencies.
- A bin sorts by a total key: magnitude, then position and proper motion, so
  any input order and any `--jobs` give the same bytes. Records with equal
  magnitudes used to come out in the order threads delivered them.
- `build_tiers` builds each tier in `tier_<d>.partial/` and renames it to
  `tier_<d>/` only when the tier is finished, so a killed run leaves no folder
  that looks like a tier; a failed tier removes its folder, and the next run
  removes one a killed run left. A bin that exists and cannot be read fails the
  build and names the file.
- `ingest` reads a file the `_MD5SUM.txt` list does not name (with a warning that
  names it); `--verify` refuses the run instead, naming it.
- `package_index` refuses pack names that differ by case only.

### The `solvemyastro` command

- clap-based, behind the `cli` feature: `ingest`, `build-tiers`, `build-cache`,
  `build-bright-cache`, `build-index` (with `--preset base|narrow`), `package`,
  `solve` and `info` (`cache-info` stays as an alias). Global `--log <file>`
  (JSON lines) and `-q`. Exit codes: 0 ok, 1 failure or refused solve, 2 bad
  arguments.
- `solve` runs the acceptance gate: a refused solution prints
  `rejected: ...` on stderr and nothing on stdout (`--no-gate` prints it). `--json`
  gains `rms_px`, `inlier_ratio`, `expected_in_fov` and `gate`.
- `info` reports a bins folder's complete and partial cells by file range, and
  each pack of an index folder (the app's `smac_index`), and reads a lone
  `manifest.json` that is an index manifest (the app's cached one) as such.
- Known issue: the acceptance gate's in-field radius takes the square root of
  the detected star count for the image side, so `expected_in_fov` comes out far
  too low and the dense-field ratio check rarely arms. A fix changes verdicts and
  is planned.
- `cargo install solvemyastro --features cli --locked` installs the command; a
  plain `cargo install solvemyastro` no longer builds it (the `cli` feature is
  required).
- `rustafits` is a version dependency, so the crate builds outside Athenaeum.

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
