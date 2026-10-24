//! solvemyastro — standalone astrometric plate solver.
//!
//! Produces **precise WCS including SIP distortion**, suitable for later
//! frame-to-frame registration. Pure in-process; depends only on
//! `rustafits` (`astroimage`) for FITS/XISF decode + raw star detection.
//!
//! Phase 0 = API surface only. Solve/cache/registration bodies land in
//! later phases (see the plan); their public *shapes* are fixed here so
//! the integration and registration seams don't move.

#![forbid(unsafe_op_in_unsafe_fn)]

use std::path::Path;

use anyhow::Result;

pub mod cache;
pub mod diag;
pub mod index;
pub(crate) mod math;
pub(crate) mod nn;
pub mod orchestrate;
pub mod project;
pub mod quad;
pub mod register;
pub(crate) mod refine;
pub mod select;
pub mod sip;
pub mod verify;
pub mod wcs;
pub mod wcs_fit;

pub use cache::{LayeredCones, PixelCache, StarCache, StarRecord};
pub use diag::{BestAttempt, FailureClass, SolveFailure};
pub use index::QuadIndex;
pub use register::{register, register_with_config, Registration};
pub use wcs::{SipCoefficients, WcsSolution};

/// Star caches passed to [`solve`]. Two modes:
///
/// * [`Caches::Legacy`] — a `deep` catalog (G<19 Gaia, full depth — needed by
///   the verify stage so the Bayesian log-odds gate has correct
///   `NR = catalog stars in FOV`) plus an optional `bright` sub-catalog (hybrid
///   floor + density top-up) used only for the per-cell quad-matching cone, with
///   automatic fallback to the deep cache when a cone returns fewer than
///   [`SolveConfig::bright_fallback_threshold`] stars. This is the pre-tier
///   behaviour; construct with [`Caches::deep_only`] / [`Caches::tiered`].
/// * [`Caches::Layered`] — an ordered stack of additive density-tier layers
///   (disjoint magnitude bands, base → deepest). Quad matching uses the base
///   layer with fallback to the full union; verify uses the full union.
///   Construct with [`Caches::layered`].
///
/// Both variants are byte-cheap (just references).
#[derive(Clone, Copy)]
pub enum Caches<'a> {
    /// Legacy deep (+ optional bright) pair.
    Legacy {
        /// Deep catalog (Gaia G<19). Used for: verify-stage inlier counting,
        /// quad matching when no `bright` is configured OR when the bright
        /// cone is too sparse.
        deep: &'a StarCache,
        /// Optional bright sub-catalog (hybrid floor + density top-up).
        /// When present, quad matching tries this first and falls back to
        /// `deep` only when the bright cone has < threshold stars.
        bright: Option<&'a StarCache>,
    },
    /// Additive density-tier stack (base → deepest, disjoint magnitude bands).
    Layered {
        /// `layers[0]` is the fast base; the union of all layers is the full depth.
        layers: &'a [&'a StarCache],
    },
}

impl<'a> Caches<'a> {
    /// Single-cache mode — quad matching and verify both use `deep`.
    /// Byte-identical to pre-bright-cache behaviour.
    pub fn deep_only(deep: &'a StarCache) -> Self {
        Self::Legacy { deep, bright: None }
    }

    /// Legacy tiered mode — quad matching tries `bright` first, falls back to
    /// `deep` on under-population. Verify always uses `deep`.
    pub fn tiered(deep: &'a StarCache, bright: &'a StarCache) -> Self {
        Self::Legacy { deep, bright: Some(bright) }
    }

    /// Additive density-tier mode — `layers[0]` is the base, the union of all
    /// layers is the full depth. A 1-element stack is byte-equivalent to
    /// [`Caches::deep_only`].
    pub fn layered(layers: &'a [&'a StarCache]) -> Self {
        Self::Layered { layers }
    }
}

/// A catalog star with epoch-corrected sky position.
#[derive(Clone, Copy, Debug)]
pub struct CatalogStar {
    /// Right ascension, degrees (epoch-corrected).
    pub ra: f64,
    /// Declination, degrees (epoch-corrected).
    pub dec: f64,
    /// Apparent magnitude (catalog native band).
    pub mag: f32,
}

/// Centroid refinement mode for the detection stage.
///
/// Controls whether PSF-refined centroids (Moffat LM fit via `rustafits`) are
/// used for quad-building. The refined path produces sub-pixel centroids
/// (~0.05 px at SNR≥20) at the cost of a per-star Moffat fit.
///
/// # Default
///
/// [`CentroidRefinement::Off`] — the fast pass-1 centroid path, byte-identical
/// to the pre-Phase-1 behaviour. Refinement must be **explicitly opted in**.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CentroidRefinement {
    /// Pass-1 intensity-weighted centroids only. Default. Zero extra cost.
    #[default]
    Off,
    /// PSF refinement via Moffat LM fit (7-parameter, fixed β=3).
    ///
    /// After `detect_fast`, each star with SNR≥10 gets a Moffat fit on a
    /// stamp window. Per-star gates (non-physical parameters, centre shift
    /// > 2 px) fall back to the pass-1 centroid. Global field gates (median
    /// FWHM < 1.5 px OR median SNR < 10) disable refinement for the whole
    /// frame and revert to pass-1 positions.
    Auto,
}

/// Optional priors from the image's metadata. All `None` ⇒ true blind.
#[derive(Clone, Copy, Debug, Default)]
pub struct SolveHints {
    /// Approx field-centre RA, degrees.
    pub ra: Option<f64>,
    /// Approx field-centre Dec, degrees.
    pub dec: Option<f64>,
    /// Approx field of view (long axis), degrees.
    pub fov_deg: Option<f64>,
    /// Approx pixel scale, arcsec/px.
    pub pixel_scale_arcsec: Option<f64>,
    /// Positional search radius around (ra,dec), degrees.
    pub search_radius_deg: Option<f64>,
    /// Observation epoch (Julian year, e.g. 2024.04) used to propagate catalog
    /// proper motions from the catalog epoch (Gaia 2016.0) to when the frame
    /// was shot. `None` ⇒ solver default (~2024.0). Pass the frame's DATE-OBS.
    pub epoch: Option<f64>,
}

/// Solver tuning.
#[derive(Clone, Debug)]
pub struct SolveConfig {
    /// Continuous quad-ratio tolerance for quad matching.
    pub quad_tolerance: f64,
    /// Catalog read magnitude ceiling (covers the Gaia G<19 build).
    pub catalog_mag_limit: f32,
    /// Fit SIP distortion after the linear solve (precise WCS).
    pub fit_sip: bool,
    /// SIP polynomial order when `fit_sip` (2 or 3).
    pub sip_order: u8,
    /// Phase 1 instrumentation — emit a per-stage timing breakdown at the
    /// end of `solve()`. Default `false` (production runs stay quiet).
    /// Set to `true` from `corpus_bench` and ad-hoc profiling to see where
    /// wall-clock actually goes.
    pub print_timing: bool,
    /// Use the Bayesian log-odds verification gate (astrometry.net style).
    /// Default `true`. When `false`, falls back to the legacy 5%-of-FOV
    /// inlier floor (kept for emergency revert; documented to over-reject
    /// valid solves in dense fields).
    pub use_bayesian_verify: bool,
    /// Bayesian log-odds acceptance threshold. Default `log(1e6) ≈ 13.8`.
    /// See [`crate::verify::DEFAULT_LOG_ODDS_MIN`].
    pub log_odds_threshold: f64,
    /// Distractor fraction `D` for the Bayesian gate (expected proportion
    /// of catalog stars in FOV that have no matching detection). Default
    /// `0.25`. See [`crate::verify::DEFAULT_DISTRACTOR_FRACTION`].
    pub distractor_fraction: f64,
    /// Minimum bright-cache cone stars before the quad-matching path
    /// falls back to the deep cache for the same cone. Only honoured
    /// when a bright sub-catalog is configured ([`Caches::tiered`]) — or,
    /// in [`Caches::layered`] mode, the base-layer cone before the union
    /// fallback. Default `30` — enough for
    /// `minimum_quads = 3 + n/140` to clear with ~5-7 quads. Set to
    /// `0` to never fall back (trust the bright cache absolutely);
    /// set to `usize::MAX` to always fall back (effectively disabling
    /// the bright path).
    pub bright_fallback_threshold: usize,
    /// Enable the scale-recovery fallback ladder (B1 harvest + B2 bounded
    /// sweep) on hinted-solve failure with a position prior. Default `true`.
    /// Set `false` to restore the prior degrade-to-blind-position behaviour.
    pub scale_fallback_enabled: bool,
    /// Minimum inliers in the failed attempt's best candidate before B1 trusts
    /// its recovered scale. Default `2`. Safe to keep low: B1's retry must pass
    /// full verification, so a wrong estimate only costs one extra pass.
    pub b1_min_inliers: usize,
    /// B2 bounded-sweep lower scale factor (× the hint). Default `0.125` (÷8) —
    /// covers focal-length-as-aperture errors, which always make the true scale
    /// finer than the hint.
    pub b2_scale_lo_factor: f64,
    /// B2 bounded-sweep upper scale factor (× the hint). Default `2.0` (×2).
    pub b2_scale_hi_factor: f64,
    /// PSF centroid refinement mode for the detection stage.
    ///
    /// Default [`CentroidRefinement::Off`] — the fast pass-1 path, byte-identical
    /// to pre-Phase-1 behaviour. Set to [`CentroidRefinement::Auto`] for the
    /// "precise" profile (Moffat-refined centroids fed into quad building).
    /// When [`CentroidRefinement::Auto`] is chosen but the global field gates
    /// reject refinement (undersampled FWHM < 1.5 px, or median SNR < 10),
    /// the solver logs a warning and uses pass-1 positions instead.
    pub centroid_refinement: CentroidRefinement,
    /// RNG seed for the PROSAC RANSAC robust-seed path (Phase 3).
    ///
    /// The PROSAC path only fires on weak seeds (seed_inliers < 30 OR
    /// quad_match_pairs < 10); it is completely bypassed on easy frames.
    /// Using a fixed seed makes the corpus gate deterministic across repeated
    /// runs. Change this to explore different PROSAC sample paths.
    ///
    /// Default `0x5F3759DF_5F3759DF` (an arbitrary non-zero constant).
    pub ransac_seed: u64,
    /// Registration inlier tolerance in arcseconds. `None` keeps the legacy
    /// fixed 4 px tolerance. When set, the pixel tolerance is derived from the
    /// reference WCS pixel scale and clamped to [1.0, 12.0] px.
    pub register_inlier_tol_arcsec: Option<f64>,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            quad_tolerance: 0.007,
            catalog_mag_limit: 19.0,
            fit_sip: true,
            sip_order: 3,
            print_timing: false,
            use_bayesian_verify: true,
            log_odds_threshold: verify::DEFAULT_LOG_ODDS_MIN,
            distractor_fraction: verify::DEFAULT_DISTRACTOR_FRACTION,
            bright_fallback_threshold: 30,
            scale_fallback_enabled: true,
            b1_min_inliers: 2,
            b2_scale_lo_factor: 0.125,
            b2_scale_hi_factor: 2.0,
            centroid_refinement: CentroidRefinement::Off,
            ransac_seed: 0x5F3759DF_5F3759DF,
            register_inlier_tol_arcsec: None,
        }
    }
}

/// The result of a successful solve. Superset of Athenaeum's
/// `PlateSolveRecord` inputs so the adapter maps 1:1.
#[derive(Clone, Debug)]
pub struct SolveSolution {
    /// Linear (CD matrix + tangent point) WCS, with optional SIP distortion
    /// polynomial. Use `wcs.pixel_to_sky` / `wcs.sky_to_pixel` to evaluate.
    pub wcs: WcsSolution,
    /// Plate scale, arcsec per pixel. Derived from the CD matrix:
    /// `sqrt(|det(CD)|) * 3600`. Constant across the field — the
    /// per-position scale variation due to SIP is not captured here.
    pub pixel_scale_arcsec: f64,
    /// Position angle of catalog north (+Dec) relative to the image +Y axis,
    /// degrees in `[0, 360)`. Positive = east-of-north (counter-clockwise on
    /// a standard image where +X is right and +Y is up).
    pub field_rotation_deg: f64,
    /// Number of catalog stars matched to detections within the inlier
    /// tolerance after the final affine refit (linear stage). Does not
    /// re-count after the SIP refit.
    pub matched_stars: usize,
    /// Total stars produced by `detect_fast` before SNR-gating / selection.
    /// Useful as a denominator for `matched_stars` and for diagnosing low-SNR
    /// fields.
    pub total_detected: usize,
    /// RMS of inlier residuals in image pixels. If SIP was applied, this is
    /// the post-SIP RMS; otherwise it is the linear-fit RMS.
    pub rms_residual_px: f64,
    /// Same as `rms_residual_px`, expressed in arcsec
    /// (`rms_residual_px * pixel_scale_arcsec`).
    pub rms_residual_arcsec: f64,
    /// Fraction `matched_stars / total_detected`. Healthy solves are
    /// typically in `[0.1, 0.7]`; very low values suggest a faint /
    /// crowded field or a partial match.
    pub inlier_ratio: f64,
    /// Identifier of the catalog used. Currently `"gaia_dr3"`. Reserved for
    /// future support of alternative catalogs (lower-density fallback, etc.).
    pub catalog_used: String,
    /// Solver-implementation identifier (e.g. `"solvemyastro"`). Opaque
    /// string; do not parse — it is for human-facing diagnostic logging.
    pub algorithm: String,
    /// Wall-clock time from `solve()` entry to return, milliseconds. Includes
    /// rustafits detection, blind search, and any SIP fit.
    pub solve_time_ms: u64,
    /// Median fitted PSF FWHM across PSF-refined stars (pixels).
    /// `Some(fwhm)` only when [`SolveConfig::centroid_refinement`] is
    /// [`CentroidRefinement::Auto`] and the global field gates accepted
    /// refinement. `None` when refinement is off or the field was rejected
    /// by the undersampled/low-SNR gate.
    pub psf_fwhm_px: Option<f64>,
}

/// B1: decide the scale to retry at, given a failed attempt. Returns the
/// recovered `best_attempt` scale only when it is physically plausible, backed
/// by enough inliers, and meaningfully different from the supplied hint (so we
/// don't re-search where step A already did). Correctness-safe: the caller's
/// retry must still pass full verification.
fn harvest_scale(
    failure: &SolveFailure,
    min_inliers: usize,
    supplied_scale_arcsec: Option<f64>,
) -> Option<f64> {
    let ba = failure.best_attempt.as_ref()?;
    let s = ba.scale_arcsec_per_px;
    if !s.is_finite() || s < 0.05 || s > 120.0 {
        return None;
    }
    if ba.inliers < min_inliers {
        return None;
    }
    if let Some(hint) = supplied_scale_arcsec {
        if hint > 0.0 && (s - hint).abs() / hint <= 0.15 {
            return None;
        }
    }
    Some(s)
}

/// B2: derive the bounded scale band `[lo, hi]` (arcsec/px) from the supplied
/// hint. Returns `None` when no scale hint was given — step A already ran the
/// full position-constrained ladder in that case, so a sweep adds nothing.
fn b2_bounds(hints: &SolveHints, cfg: &SolveConfig) -> Option<(f64, f64)> {
    let hint = hints.pixel_scale_arcsec?;
    if !hint.is_finite() || hint <= 0.0 {
        return None;
    }
    Some((hint * cfg.b2_scale_lo_factor, hint * cfg.b2_scale_hi_factor))
}

/// Solve an image to a precise (SIP) WCS.
///
/// `image` is a path to a FITS/XISF file (decoded + detected via rustafits
/// internally). Implements the full blind orchestration: autoFOV ladder
/// × square-spiral sky search × gnomonic-consistent quad matching ×
/// second-solve × optional SIP distortion fit.
///
/// `cancel`: optional cooperative cancellation flag. When set to `true` by
/// the caller, the solver returns `Err("cancelled")` at the next check point
/// (between FOV rungs, between rayon trials in a rung, before each candidate
/// verify). Pass `None` if cancellation is not needed.
pub fn solve(
    image: &Path,
    hints: &SolveHints,
    caches: &Caches<'_>,
    cfg: &SolveConfig,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<SolveSolution> {
    solve_with_index(image, hints, caches, None, cfg, cancel)
}

/// [`solve`] with a [`QuadIndex`]. When the hints leave the position or the
/// scale open (blind, position-only, scale-only), the index is searched
/// before the spiral: its cost does not grow with the sky area or scale
/// range left to cover. Fully hinted frames solve exactly as with [`solve`].
pub fn solve_with_index(
    image: &Path,
    hints: &SolveHints,
    caches: &Caches<'_>,
    index: Option<&QuadIndex>,
    cfg: &SolveConfig,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<SolveSolution> {
    // Step A — hinted (or blind) solve.
    let result = orchestrate::solve(image, hints, caches, index, cfg, cancel);
    if result.is_ok() {
        return result;
    }

    let has_position = hints.ra.is_some() || hints.dec.is_some();
    let cancelled = || cancel
        .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
        .unwrap_or(false);

    // Scale-recovery fallback (B1 harvest, then B2 bounded sweep) — position
    // kept. Only fires after step A failed, so correctly-hinted frames pay
    // nothing. Each retry must still pass full verification.
    if cfg.scale_fallback_enabled && has_position && !cancelled() {
        // B1 — retry once at the scale recovered from step A's best attempt.
        let harvested = result
            .as_ref()
            .err()
            .and_then(|e| e.downcast_ref::<SolveFailure>())
            .and_then(|f| harvest_scale(f, cfg.b1_min_inliers, hints.pixel_scale_arcsec));
        if let Some(scale) = harvested {
            tracing::warn!(
                scale_arcsec_px = scale,
                "hinted solve failed — retrying at recovered scale (position kept)"
            );
            let b1_hints = SolveHints {
                pixel_scale_arcsec: Some(scale),
                fov_deg: None,
                ..*hints
            };
            if let Ok(sol) = orchestrate::solve(image, &b1_hints, caches, index, cfg, cancel) {
                return Ok(sol);
            }
        }

        // B2 — bounded scale sweep at the known position.
        if !cancelled() {
            if let Some((lo, hi)) = b2_bounds(hints, cfg) {
                tracing::warn!(
                    scale_lo_arcsec_px = lo,
                    scale_hi_arcsec_px = hi,
                    "hinted solve failed — retrying with bounded scale sweep (position kept)"
                );
                let sweep_hints = SolveHints {
                    pixel_scale_arcsec: None,
                    fov_deg: None,
                    ..*hints
                };
                if let Ok(sol) =
                    orchestrate::solve_inner(image, &sweep_hints, caches, index, cfg, cancel, Some((lo, hi)))
                {
                    return Ok(sol);
                }
            }
        }
    }

    // Step C — existing degrade-to-blind: clear the position prior, keep scale.
    if has_position && !cancelled() {
        tracing::warn!(
            "hinted solve failed — retrying with the position prior cleared (blind position, scale kept)"
        );
        let blind_hints = SolveHints { ra: None, dec: None, ..*hints };
        if let Ok(sol) = orchestrate::solve(image, &blind_hints, caches, index, cfg, cancel) {
            return Ok(sol);
        }
    }

    // All rungs failed — surface step A's original (richest) error.
    result
}


#[cfg(test)]
mod fallback_config_tests {
    use super::SolveConfig;

    #[test]
    fn scale_fallback_defaults() {
        let c = SolveConfig::default();
        assert!(c.scale_fallback_enabled);
        assert_eq!(c.b1_min_inliers, 2);
        assert_eq!(c.b2_scale_lo_factor, 0.125);
        assert_eq!(c.b2_scale_hi_factor, 2.0);
    }
}

#[cfg(test)]
mod harvest_tests {
    use super::harvest_scale;
    use crate::diag::{BestAttempt, FailureClass, SolveFailure};

    fn failure_with(scale: f64, inliers: usize) -> SolveFailure {
        SolveFailure {
            class: FailureClass::VerifyGap { best: inliers, required: 6 },
            message: String::new(),
            best_attempt: Some(BestAttempt {
                pass_balance: true, pass_idx: 1, rung_idx: 4, rung_total: 6,
                seed_ra: 36.7, seed_dec: 60.8, scale_arcsec_per_px: scale,
                cat_in_fov: 237, inliers, required: 6, rms_px: 0.0,
                seed_rms_px: 2.7, log_odds: 12.9,
                dist_from_truth_deg: None, scale_ratio_to_truth: None,
            }),
        }
    }

    #[test]
    fn harvests_when_scale_diverges_and_inliers_ok() {
        // true 0.879 vs supplied 3.52 (4x) -> harvest.
        assert_eq!(harvest_scale(&failure_with(0.879, 2), 2, Some(3.52)), Some(0.879));
    }
    #[test]
    fn rejects_low_inliers() {
        assert_eq!(harvest_scale(&failure_with(0.879, 1), 2, Some(3.52)), None);
    }
    #[test]
    fn rejects_when_scale_matches_hint() {
        // recovered scale within 15% of the supplied hint -> A already searched there.
        assert_eq!(harvest_scale(&failure_with(3.50, 5), 2, Some(3.52)), None);
    }
    #[test]
    fn rejects_unphysical_scale() {
        assert_eq!(harvest_scale(&failure_with(0.0, 5), 2, Some(3.52)), None);
        assert_eq!(harvest_scale(&failure_with(500.0, 5), 2, Some(3.52)), None);
    }
}

#[cfg(test)]
mod b2_bounds_tests {
    use super::b2_bounds;
    use crate::{SolveConfig, SolveHints};

    #[test]
    fn bounds_from_pixel_scale_hint() {
        let cfg = SolveConfig::default(); // ÷8 .. ×2
        let hints = SolveHints { pixel_scale_arcsec: Some(3.52), ..Default::default() };
        let (lo, hi) = b2_bounds(&hints, &cfg).expect("bounds");
        assert!((lo - 3.52 * 0.125).abs() < 1e-9);
        assert!((hi - 3.52 * 2.0).abs() < 1e-9);
    }
    #[test]
    fn none_when_no_scale_hint() {
        // No scale hint -> step A already ran the full position-constrained
        // ladder, so B2 adds nothing.
        let cfg = SolveConfig::default();
        let hints = SolveHints { ra: Some(1.0), dec: Some(2.0), ..Default::default() };
        assert_eq!(b2_bounds(&hints, &cfg), None);
    }
}
