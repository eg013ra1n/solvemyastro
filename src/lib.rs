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
pub(crate) mod math;
pub mod orchestrate;
pub mod project;
pub mod quad;
pub mod select;
pub mod sip;
pub mod wcs;
pub mod wcs_fit;

pub use cache::{PixelCache, StarCache, StarRecord};
pub use wcs::{SipCoefficients, WcsSolution};

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
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            quad_tolerance: 0.007,
            catalog_mag_limit: 19.0,
            fit_sip: true,
            sip_order: 3,
            print_timing: false,
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
    cache: &StarCache,
    cfg: &SolveConfig,
    cancel: Option<&std::sync::atomic::AtomicBool>,
) -> Result<SolveSolution> {
    orchestrate::solve(image, hints, cache, cfg, cancel)
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
