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
pub(crate) mod math;
pub mod orchestrate;
pub mod project;
pub mod quad;
pub mod select;
pub mod sip;
pub mod verify;
pub mod wcs;
pub mod wcs_fit;

pub use cache::{PixelCache, StarCache, StarRecord};
pub use diag::{BestAttempt, FailureClass, SolveFailure};
pub use wcs::{SipCoefficients, WcsSolution};

/// Bundle of star caches passed to [`solve`]. Always carries a `deep`
/// catalog (G<19 Gaia, full depth — needed by the verify stage so the
/// Bayesian log-odds gate has correct `NR = catalog stars in FOV`).
/// Optionally carries a `bright` sub-catalog (hybrid floor + density
/// top-up) used only for the per-cell quad-matching cone — faster
/// per-pixel decode + smaller working set, with automatic fallback to
/// the deep cache when a cone returns fewer than
/// [`SolveConfig::bright_fallback_threshold`] stars.
///
/// Construct with [`Caches::deep_only`] (current single-cache behaviour)
/// or [`Caches::tiered`] (bright + deep). Both variants are byte-cheap
/// (just two references).
#[derive(Clone, Copy)]
pub struct Caches<'a> {
    /// Deep catalog (Gaia G<19). Used for: verify-stage inlier counting,
    /// quad matching when no `bright` is configured OR when the bright
    /// cone is too sparse.
    pub deep: &'a StarCache,
    /// Optional bright sub-catalog (hybrid floor + density top-up).
    /// When present, quad matching tries this first and falls back to
    /// `deep` only when the bright cone has < threshold stars.
    pub bright: Option<&'a StarCache>,
}

impl<'a> Caches<'a> {
    /// Single-cache mode — quad matching and verify both use `deep`.
    /// Byte-identical to pre-bright-cache behaviour.
    pub fn deep_only(deep: &'a StarCache) -> Self {
        Self { deep, bright: None }
    }

    /// Tiered mode — quad matching tries `bright` first, falls back to
    /// `deep` on under-population. Verify always uses `deep`.
    pub fn tiered(deep: &'a StarCache, bright: &'a StarCache) -> Self {
        Self { deep, bright: Some(bright) }
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
    /// when [`Caches::bright`] is `Some`. Default `30` — enough for
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
    /// future support of alternative catalogs (Tycho-2 fallback, etc.).
    pub catalog_used: String,
    /// Solver-implementation identifier (e.g. `"solvemyastro"`). Opaque
    /// string; do not parse — it is for human-facing diagnostic logging.
    pub algorithm: String,
    /// Wall-clock time from `solve()` entry to return, milliseconds. Includes
    /// rustafits detection, blind search, and any SIP fit.
    pub solve_time_ms: u64,
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
    let result = orchestrate::solve(image, hints, caches, cfg, cancel);

    // Degrade-to-blind: if a positional prior was present and the hinted solve
    // failed, the prior may be wrong (stale/mis-synced mount, or a leaked
    // neighbour coordinate). Retry once with the position cleared — the scale
    // hint is kept, so a known-scale wrong-pointing frame stays fast. Only
    // fires on failure, so correctly-hinted frames pay nothing.
    if result.is_err() && (hints.ra.is_some() || hints.dec.is_some()) {
        let cancelled = cancel
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false);
        if !cancelled {
            eprintln!(
                "[solvemyastro] hinted solve failed — retrying with the position prior cleared \
                 (blind position, scale kept)"
            );
            let blind_hints = SolveHints {
                ra: None,
                dec: None,
                ..*hints
            };
            if let Ok(sol) = orchestrate::solve(image, &blind_hints, caches, cfg, cancel) {
                return Ok(sol);
            }
            // Both failed — fall through to surface the original (richer) error.
        }
    }
    result
}

/// Future seam (NOT in scope for this plan): register a sub-frame's
/// detections to an already-solved reference WCS and return the sub's
/// propagated/refined WCS. Shape is fixed now so the later registration
/// layer is not architecturally blocked.
pub fn register(
    _reference: &WcsSolution,
    _sub_detections: &[(f64, f64)],
) -> Result<WcsSolution> {
    anyhow::bail!("solvemyastro::register not implemented (future registration layer)")
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
