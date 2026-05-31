//! Blind plate-solve orchestration.
//!
//! Implements the full solve pipeline:
//! 1. Star detection via `astroimage::ImageAnalyzer::detect_fast`.
//! 2. SNR-gated star selection (`select_fast_stars`).
//! 3. autoFOV ladder (hinted ≤ 3 rungs, blind 9.5→0.38° ÷1.5).
//! 4. Square-spiral sky-cell search.
//! 5. Per-cell: cone → gnomonic project → build quads → match → fit affine.
//! 6. Seed WCS → inlier count → second-solve (full-field affine refit).
//! 7. Optional SIP fit (`sip.rs`).

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Result;
use rayon::prelude::*;

/// Per-stage timing accumulator for the solve pipeline. All fields are
/// nanoseconds. Updated with `Relaxed` ordering from rayon workers; printed
/// once at solve end when `SolveConfig::print_timing` is true.
///
/// Phase 1 instrumentation — the data drives the speed-fix targeting in
/// Phase 3 (kd-tree on `match_quads` vs `count_inliers`).
#[derive(Default, Debug)]
pub(crate) struct TimingProfile {
    pub cone_ns: AtomicU64,
    pub cone_calls: AtomicU64,
    pub project_ns: AtomicU64,
    pub build_quads_cat_ns: AtomicU64,
    pub match_quads_ns: AtomicU64,
    pub match_quads_calls: AtomicU64,
    pub fit_affine_ns: AtomicU64,
    pub fit_affine_calls: AtomicU64,
    pub count_inliers_ns: AtomicU64,
    pub count_inliers_calls: AtomicU64,
    pub refit_affine_ns: AtomicU64,
    pub fit_sip_ns: AtomicU64,
}

impl TimingProfile {
    fn add(target: &AtomicU64, dur: std::time::Duration) {
        target.fetch_add(dur.as_nanos() as u64, Ordering::Relaxed);
    }
    fn print_summary(&self, total: std::time::Duration, frame: &str) {
        let total_ms = total.as_millis() as u64;
        let ms = |ns: u64| ns / 1_000_000;
        let pct = |ns: u64| if total_ms > 0 { ms(ns) * 100 / total_ms } else { 0 };
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        eprintln!(
            "[solvemyastro timing] {frame}: total {total_ms} ms — \
             cone {cone}ms/{ccalls} ({cone_pct}%), \
             project {proj}ms, \
             build_quads_cat {bqc}ms, \
             match_quads {mq}ms/{mqcalls} ({mq_pct}%), \
             fit_affine {fa}ms/{facalls}, \
             count_inliers {ci}ms/{cicalls} ({ci_pct}%), \
             refit_affine {ra}ms, \
             fit_sip {sip}ms",
            cone = ms(g(&self.cone_ns)), ccalls = g(&self.cone_calls), cone_pct = pct(g(&self.cone_ns)),
            proj = ms(g(&self.project_ns)),
            bqc = ms(g(&self.build_quads_cat_ns)),
            mq = ms(g(&self.match_quads_ns)), mqcalls = g(&self.match_quads_calls), mq_pct = pct(g(&self.match_quads_ns)),
            fa = ms(g(&self.fit_affine_ns)), facalls = g(&self.fit_affine_calls),
            ci = ms(g(&self.count_inliers_ns)), cicalls = g(&self.count_inliers_calls), ci_pct = pct(g(&self.count_inliers_ns)),
            ra = ms(g(&self.refit_affine_ns)),
            sip = ms(g(&self.fit_sip_ns)),
        );
    }
}

use crate::{
    diag::{CellStop, CellTrace, KnownTruth, PassTrace, RungTrace, SeedInfo, SolveTrace, TraceOutcome, VerifyTrace, known_truth_for},
    math::great_circle_deg,
    project::project_catalog,
    quad::{build_quads, fit_affine, group_size_for, match_quads_stats},
    select::select_fast_stars,
    wcs::WcsSolution,
    wcs_fit::affine_to_wcs,
    CatalogStar, PixelCache, SolveConfig, SolveHints, SolveSolution, StarCache,
};

// ---------------------------------------------------------------------------
// Blind-search constants
// ---------------------------------------------------------------------------

/// Blind autoFOV top rung.
const FOV_MAX_DEG: f64 = 9.5;
/// Blind autoFOV bottom rung floor.
const FOV_MIN_DEG: f64 = 0.38;
/// Per-rung divisor.
const FOV_DIV: f64 = 1.5;
/// Full-sky search radius when blind.
const FULL_SKY_RADIUS_DEG: f64 = 180.0;
/// Default hinted search radius (degrees) when none specified.
const DEFAULT_HINTED_RADIUS_DEG: f64 = 10.0;
/// Oversize factor for catalog cone relative to the rung FOV.
/// Read a slightly larger cone so edge stars are included.
const CONE_OVERSIZE: f64 = 1.0;
/// Maximum catalog stars per trial (deep, no magnitude-budget starvation).
const MAX_CAT_STARS: usize = 6000;
/// Inlier match radius in arcsec (adaptive: this is the base; see adaptive_tol).
const BASE_MATCH_ARCSEC: f64 = 3.0;
/// Minimum inlier floor (absolute) regardless of density calculation.
const MIN_ABSOLUTE_INLIERS: usize = 6;
/// Adaptive tolerance maximum (px) so we don't accept garbage on tiny scales.
const MAX_MATCH_PX: f64 = 20.0;
/// Magnitude limit for the verification cone (bright stars only — mirrors
/// athenaeum refine.rs VERIFY_MAG_LIMIT). We use the deep catalog for quad
/// matching, but only bright stars for inlier acceptance to avoid the
/// `expected / 15%` floor being inflated by thousands of faint Gaia stars.
const VERIFY_MAG_LIMIT: f32 = 12.0;

// ---------------------------------------------------------------------------
// FOV ladder
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct FovRung {
    fov_deg: f64,
    #[allow(dead_code)]
    scale_arcsec: f64,
}

fn fov_rungs(long_px: usize, hints: &SolveHints) -> Vec<FovRung> {
    let hint_scale = hints.pixel_scale_arcsec.or_else(|| {
        hints.fov_deg.map(|fov| fov * 3600.0 / long_px.max(1) as f64)
    });

    let mk = |fov: f64| FovRung {
        fov_deg: fov,
        scale_arcsec: fov * 3600.0 / long_px.max(1) as f64,
    };

    if let Some(s) = hint_scale {
        if s.is_finite() && s > 0.0 {
            let fov = s * long_px.max(1) as f64 / 3600.0;
            // Ladder centred on the hint, expanding outward in FOV_DIV steps.
            // Try up to 4 steps in each direction (covers 1.5^4 ≈ 5× scale range),
            // clamped to [FOV_MIN_DEG, FOV_MAX_DEG].
            // This handles stacked/drizzled images where the stored FOCALLEN or CDELT1
            // may differ from the actual per-pixel scale by 2–4×.
            // Centre-first, expanding outward (centre, ÷1.5, ×1.5, ÷1.5², …).
            // The hinted scale is the most likely rung, so the first-verify
            // early-exit fires on it immediately instead of after several
            // dense wrong-scale spirals (the finest rungs have the most cells).
            let mut rungs = Vec::new();
            rungs.push(mk(fov.clamp(FOV_MIN_DEG, FOV_MAX_DEG)));
            let mut up = fov;
            let mut down = fov;
            for _ in 0..4 {
                down /= FOV_DIV;
                up *= FOV_DIV;
                if down >= FOV_MIN_DEG {
                    rungs.push(mk(down));
                }
                if up <= FOV_MAX_DEG {
                    rungs.push(mk(up));
                }
            }
            return rungs;
        }
    }
    // Blind: push-then-check so first rung ≤ fov_min is still tried.
    let mut rungs = Vec::new();
    let mut fov = FOV_MAX_DEG;
    loop {
        rungs.push(mk(fov));
        if fov <= FOV_MIN_DEG {
            break;
        }
        fov /= FOV_DIV;
    }
    rungs
}

/// FOV ladder over an explicit `[lo_arcsec, hi_arcsec]` scale band (B2 of the
/// scale-recovery fallback). Rungs descend from the coarse (hi) bound by
/// `FOV_DIV`, clamped to `[FOV_MIN_DEG, FOV_MAX_DEG]`. Always includes a rung at
/// (or just below) the fine bound so a finer-than-hint true scale is reached.
fn fov_rungs_bounded(long_px: usize, lo_arcsec: f64, hi_arcsec: f64) -> Vec<FovRung> {
    let lp = long_px.max(1) as f64;
    let mk = |fov: f64| FovRung { fov_deg: fov, scale_arcsec: fov * 3600.0 / lp };
    let lo_fov = (lo_arcsec * lp / 3600.0).clamp(FOV_MIN_DEG, FOV_MAX_DEG);
    let hi_fov = (hi_arcsec * lp / 3600.0).clamp(FOV_MIN_DEG, FOV_MAX_DEG);
    let (lo_fov, hi_fov) = if lo_fov <= hi_fov { (lo_fov, hi_fov) } else { (hi_fov, lo_fov) };

    let mut rungs = Vec::new();
    let mut fov = hi_fov;
    while fov > lo_fov {
        rungs.push(mk(fov));
        fov /= FOV_DIV;
    }
    rungs.push(mk(lo_fov)); // guarantee the fine end is tried
    rungs
}

// ---------------------------------------------------------------------------
// Square-spiral sky-cell generator
// ---------------------------------------------------------------------------

/// Generate sky cells in square-spiral order around the seed centre.
fn spiral_cells(center: Option<(f64, f64)>, radius_deg: f64, step_deg: f64) -> Vec<(f64, f64)> {
    let (ra0, dec0) = center.unwrap_or((0.0, 0.0));
    let radius = if center.is_some() { radius_deg.max(0.0) } else { FULL_SKY_RADIUS_DEG };
    let step = step_deg.max(1e-6);
    let max_distance = ((radius / (step + 1e-5)).round() as i64).max(0);

    let mut out: Vec<(f64, f64)> = Vec::new();
    let (mut sx, mut sy): (i64, i64) = (0, 0);
    let (mut dx, mut dy): (i64, i64) = (0, -1);

    loop {
        let dec = step * sy as f64 + dec0;
        if dec.abs() <= 90.0 {
            let cosd = dec.to_radians().cos();
            let ra = if cosd.abs() > 1e-6 {
                ra0 + step * sx as f64 / cosd
            } else {
                ra0
            };
            let ra = ra.rem_euclid(360.0);
            if great_circle_deg(ra, dec, ra0, dec0) <= radius + step * 0.5 {
                out.push((ra, dec));
            }
        }
        if sx > max_distance {
            break;
        }
        // Spiral turning-point recurrence.
        if sx == sy || (sx < 0 && sx == -sy) || (sx > 0 && sx == 1 - sy) {
            let t = dx;
            dx = -dy;
            dy = t;
        }
        sx += dx;
        sy += dy;
    }
    out
}

// ---------------------------------------------------------------------------
// Density-aware minimum-quads floor
// ---------------------------------------------------------------------------

fn minimum_quads(n: usize) -> usize {
    (3 + n / 140).max(3)
}

/// Catalog cone for the quad-matching pass with bright-cache auto-fallback.
///
/// When a bright sub-catalog is configured (`bright_cache.is_some()`), try
/// it first — it's smaller (hybrid floor+top-up per HEALPix cell) so the
/// per-pixel decode + PM-propagation in `PixelCache` is much faster. If
/// the bright cone returns fewer than `fallback_threshold` stars (rare
/// with the hybrid build — only HEALPix singular caps or other boundary
/// degeneracies), retry the same cone against the deep catalog and use
/// that. Either way returns one set, not a union.
///
/// When `bright_cache` is `None`, this is byte-identical to a direct
/// `cache.cone_cached(...)` call.
#[allow(clippy::too_many_arguments)]
fn cone_for_quad_match(
    deep_cache: &StarCache,
    deep_pc: &PixelCache,
    bright_cache: Option<&StarCache>,
    bright_pc: Option<&PixelCache>,
    ra: f64,
    dec: f64,
    radius_deg: f64,
    mag_limit: f32,
    obs_epoch: f64,
    fallback_threshold: usize,
) -> Result<Vec<CatalogStar>> {
    if let (Some(bc), Some(bpc)) = (bright_cache, bright_pc) {
        let bright_stars = bc.cone_cached(bpc, ra, dec, radius_deg, mag_limit, obs_epoch)?;
        if bright_stars.len() >= fallback_threshold {
            return Ok(bright_stars);
        }
        // Bright cone too sparse — fall back to the deep cache.
    }
    deep_cache.cone_cached(deep_pc, ra, dec, radius_deg, mag_limit, obs_epoch)
}

// ---------------------------------------------------------------------------
// Per-cell trial result
// ---------------------------------------------------------------------------

struct TrialResult {
    wcs: WcsSolution,
    #[allow(dead_code)]
    tangent: (f64, f64),
    scale_arcsec: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_trial(
    image_quads: &[crate::quad::Quad],
    image_pts: &[(f64, f64)],
    image_center: (f64, f64),
    cell: (f64, f64),
    cat_stars: &[CatalogStar],
    group_size: usize,
    cfg: &SolveConfig,
    min_quads: usize,
    balance: bool,
    timing: &TimingProfile,
    hint: Option<(f64, f64)>,
    ct: &mut CellTrace,
) -> Option<TrialResult> {
    let (ra0, dec0) = cell;

    // In `balance` mode (the fallback pass) density-cap the image quad pool
    // to the catalog's depth. `build_quads` forms quads from spatial
    // nearest-neighbours, so when the image detects fainter than the catalog
    // the two pools don't correspond and matching starves. The catalog cone
    // covers the frame rectangle scaled by `aspect / (π·OVERSIZE²)`, so the
    // catalog count expected within the frame is `n_cat·aspect/(π·OVERSIZE²)`
    // — cap the image pool to that. Catalog is never reduced. When the image
    // is not the deeper pool the cap is ≥ n_img and this is a no-op, so the
    // fallback pass can only ever help, never regress.
    let balanced: Option<(Vec<crate::quad::Quad>, usize, usize)> = if balance {
        let dim_a = 2.0 * image_center.0 + 1.0;
        let dim_b = 2.0 * image_center.1 + 1.0;
        let aspect = dim_a.min(dim_b) / dim_a.max(dim_b);
        let est = (cat_stars.len() as f64 * aspect
            / (std::f64::consts::PI * CONE_OVERSIZE * CONE_OVERSIZE))
            .round() as usize;
        let img_keep = est.clamp(4, image_pts.len());
        if img_keep >= image_pts.len() {
            None // image not the deeper pool — nothing to cap
        } else {
            let gs = group_size_for(img_keep);
            let q = build_quads(&image_pts[..img_keep], img_keep, gs);
            Some((q, gs, minimum_quads(img_keep)))
        }
    } else {
        None
    };
    let (iquads, gs, mq): (&[crate::quad::Quad], usize, usize) = match &balanced {
        Some((q, g, m)) => (q.as_slice(), *g, *m),
        None => (image_quads, group_size, min_quads),
    };
    if iquads.len() < 3 {
        ct.stop = CellStop::CatQuadsEmpty; // image side; rare, reported under same bucket
        return None;
    }

    let t_proj = Instant::now();
    let projected = project_catalog(cat_stars, ra0, dec0);
    TimingProfile::add(&timing.project_ns, t_proj.elapsed());
    if projected.len() < 4 {
        ct.stop = CellStop::ConeTooSparse;
        return None;
    }
    let t_bq = Instant::now();
    let cat_quads = build_quads(&projected, projected.len(), gs);
    TimingProfile::add(&timing.build_quads_cat_ns, t_bq.elapsed());
    ct.cat_quads = cat_quads.len();
    if cat_quads.is_empty() {
        ct.stop = CellStop::CatQuadsEmpty;
        return None;
    }
    let t_mq = Instant::now();
    let (matches, raw_match_count) =
        match_quads_stats(iquads, &cat_quads, cfg.quad_tolerance);
    TimingProfile::add(&timing.match_quads_ns, t_mq.elapsed());
    timing.match_quads_calls.fetch_add(1, Ordering::Relaxed);
    ct.match_raw = raw_match_count;
    ct.match_filtered = matches.len();
    if matches.len() < mq {
        ct.stop = CellStop::MatchBelowFloor;
        return None;
    }
    let t_fa = Instant::now();
    let affine = match fit_affine(&matches, iquads, &cat_quads) {
        Some(a) => a,
        None => {
            ct.stop = CellStop::AffineFailed;
            return None;
        }
    };
    TimingProfile::add(&timing.fit_affine_ns, t_fa.elapsed());
    timing.fit_affine_calls.fetch_add(1, Ordering::Relaxed);
    let wcs = affine_to_wcs(&affine, image_center, (ra0, dec0));

    // Sanity: pixel scale must be in physical range (0.05"–120"/px).
    let scale = wcs.pixel_scale_arcsec();
    if !scale.is_finite() || scale < 0.05 || scale > 120.0 {
        ct.stop = CellStop::ScaleSanityFailed;
        return None;
    }

    let (seed_ra, seed_dec) = wcs.pixel_to_sky(image_center.0, image_center.1);
    let dist_from_hint_deg = hint.map(|(hr, hd)| great_circle_deg(hr, hd, seed_ra, seed_dec));
    ct.seed = Some(SeedInfo {
        ra: seed_ra,
        dec: seed_dec,
        scale_arcsec_per_px: scale,
        dist_from_hint_deg,
    });
    ct.stop = CellStop::SeedEmitted;
    Some(TrialResult { wcs, tangent: (ra0, dec0), scale_arcsec: scale })
}

// ---------------------------------------------------------------------------
// Inlier counting for verification + second solve
// ---------------------------------------------------------------------------

/// Count catalog→detection inlier pairs given a WCS.
/// Returns (count, residual_sum_sq, pairs).
fn count_inliers(
    wcs: &WcsSolution,
    cat_stars: &[CatalogStar],
    image_pts: &[(f64, f64)],
    width: f64,
    height: f64,
    tol_px: f64,
) -> (usize, f64, Vec<((f64, f64), (f64, f64))>) {
    let mut count = 0usize;
    let mut rss = 0.0_f64;
    let mut pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();

    for cat in cat_stars {
        let (px, py) = wcs.sky_to_pixel(cat.ra, cat.dec);
        if !px.is_finite() || !py.is_finite() {
            continue;
        }
        if px < 0.0 || px >= width || py < 0.0 || py >= height {
            continue;
        }
        // Find the nearest detected star. Compare squared distances — no sqrt
        // in the hot loop; only the ordering matters, and `rss` wants the
        // squared residual anyway.
        let mut best_d2 = f64::MAX;
        let mut best_img = (0.0_f64, 0.0_f64);
        for &(ix, iy) in image_pts {
            let d2 = (ix - px).powi(2) + (iy - py).powi(2);
            if d2 < best_d2 {
                best_d2 = d2;
                best_img = (ix, iy);
            }
        }
        if best_d2 <= tol_px * tol_px {
            count += 1;
            rss += best_d2;
            pairs.push(((px, py), best_img));
        }
    }
    (count, rss, pairs)
}

/// Full-field affine refit over inlier pairs (catalog-predicted pixel → detected pixel).
/// Returns None if too few pairs.
fn refit_affine(
    pairs: &[((f64, f64), (f64, f64))],
    _image_center: (f64, f64),
    wcs: &WcsSolution,
) -> Option<WcsSolution> {
    if pairs.len() < 6 {
        return None;
    }

    // Build correspondences: catalog sky → image detection pixel.
    // We do this as a similarity fit in the tangent plane.
    // For each pair: project catalog sky through wcs.crval tangent point → arcsec,
    // fit detected pixel vs projected arcsec, recover new WCS.
    use nalgebra::{DMatrix, DVector};

    let (ra0, dec0) = wcs.crval;
    let n = pairs.len();

    // pairs: ((pred_px, pred_py), (det_px, det_py))
    // In arcsec tangent plane about crval:
    // xi_as = CD11*(det_x - crpix_x)*3600 + CD12*(det_y - crpix_y)*3600
    // We fit: detected pixel (u=det_x - crpix_x, v=det_y - crpix_y) → tangent arcsec
    // xi_as = a1*u + b1*v + c1, eta_as = a2*u + b2*v + c2

    let mut mat = DMatrix::zeros(n, 3);
    let mut bx = DVector::zeros(n);
    let mut by = DVector::zeros(n);

    for (row, &((_, _), (dx, dy))) in pairs.iter().enumerate() {
        let u = dx - wcs.crpix.0;
        let v = dy - wcs.crpix.1;
        // Catalog sky position for this pair: sky_to_tangent of the catalog star.
        // We use the predicted pixel → sky → tangent chain.
        let (sky_ra, sky_dec) = wcs.pixel_to_sky(pairs[row].0.0, pairs[row].0.1);
        let (xi_deg, eta_deg) = crate::wcs::tan::sky_to_tangent(sky_ra, sky_dec, ra0, dec0);
        let xi_as = xi_deg * 3600.0;
        let eta_as = eta_deg * 3600.0;

        mat[(row, 0)] = u;
        mat[(row, 1)] = v;
        mat[(row, 2)] = 1.0;
        bx[row] = xi_as;
        by[row] = eta_as;
    }

    let svd = mat.svd(true, true);
    let eps = 1e-12;
    let px = svd.solve(&bx, eps).ok()?;
    let py = svd.solve(&by, eps).ok()?;

    // Sanity: xy scale ratio.
    let sx2 = px[0].powi(2) + px[1].powi(2);
    let sy2 = py[0].powi(2) + py[1].powi(2);
    if sy2 < 1e-30 {
        return None;
    }
    let ratio = sx2 / sy2;
    if !(0.7..=1.3).contains(&ratio) {
        // Looser than affine fit (that uses 0.9–1.1) to accommodate distortion
        return None;
    }

    // Build WCS from the refit affine (origin = crpix, tangent = crval).
    // The affine has origin at crpix (u = x - crpix_x), so:
    // xi_as = a1*(x - crpix_x) + b1*(y - crpix_y) + c1
    // When c1 ≈ 0 (as expected if the crval is well-aligned), this is clean.
    // We fold c1 back into crval by evaluating at (crpix_x, crpix_y):
    let xi_at_crpix = px[2];
    let eta_at_crpix = py[2];
    let (new_crval_ra, new_crval_dec) = crate::wcs::tan::tangent_to_sky(
        xi_at_crpix / 3600.0,
        eta_at_crpix / 3600.0,
        ra0,
        dec0,
    );

    let cd = [
        [px[0] / 3600.0, px[1] / 3600.0],
        [py[0] / 3600.0, py[1] / 3600.0],
    ];

    Some(WcsSolution {
        crpix: wcs.crpix,
        crval: (new_crval_ra, new_crval_dec),
        cd,
        sip_forward: None,
        sip_reverse: None,
    })
}

// ---------------------------------------------------------------------------
// Adaptive tolerance
// ---------------------------------------------------------------------------

fn adaptive_tol_px(scale_arcsec: f64, base_arcsec: f64) -> f64 {
    if scale_arcsec < 1e-6 {
        return 5.0;
    }
    (base_arcsec / scale_arcsec).min(MAX_MATCH_PX).max(1.5)
}

// ---------------------------------------------------------------------------
// Required inlier floor (density-aware)
// ---------------------------------------------------------------------------

fn required_inliers(expected_bright_in_fov: usize, n_detected: usize) -> usize {
    // The requirement is: max(
    //   MIN_ABSOLUTE_INLIERS,
    //   min(
    //     5% of bright catalog stars in FOV,    ← anti-false-positive
    //     5% of detected image stars             ← anti-dense-field over-rejection
    //   )
    // )
    //
    // Rationale: In very dense Milky Way / wide-field frames, 5% of thousands of
    // bright catalog stars exceeds all detectable matches. Capping at 5% of
    // DETECTED stars (which is the image depth, not the catalog depth) prevents
    // over-rejection while still demanding a real positional match.
    // Both floors are at 5% so the formula is symmetric: roughly 5% of
    // available reference points must corroborate.
    let cat_floor = (expected_bright_in_fov as f64 * 0.05) as usize;
    let det_cap = (n_detected as f64 * 0.05) as usize;
    let dynamic = cat_floor.min(det_cap);
    MIN_ABSOLUTE_INLIERS.max(dynamic)
}

// ---------------------------------------------------------------------------
// Main solve entry point
// ---------------------------------------------------------------------------

/// Public solve entry — hint-derived/blind FOV ladder (no scale-bound override).
pub fn solve(
    image: &Path,
    hints: &SolveHints,
    caches: &crate::Caches<'_>,
    cfg: &SolveConfig,
    cancel: Option<&AtomicBool>,
) -> Result<SolveSolution> {
    solve_inner(image, hints, caches, cfg, cancel, None)
}

/// Internal entry: `scale_bounds = Some((lo_arcsec, hi_arcsec))` drives a
/// bounded FOV ladder (B2); `None` uses the hint-derived/blind ladder.
pub(crate) fn solve_inner(
    image: &Path,
    hints: &SolveHints,
    caches: &crate::Caches<'_>,
    cfg: &SolveConfig,
    cancel: Option<&AtomicBool>,
    scale_bounds: Option<(f64, f64)>,
) -> Result<SolveSolution> {
    // The deep cache is the only one used by every stage; the optional
    // bright cache is consulted for quad matching with auto-fallback.
    let cache = caches.deep;
    let t0 = Instant::now();
    let is_cancelled = || cancel.is_some_and(|f| f.load(Ordering::Relaxed));

    // 1. Detect stars.
    let result = astroimage::ImageAnalyzer::new()
        .with_detection_sigma(5.0)
        .with_max_stars(600)
        .detect_fast(image)
        .map_err(|e| anyhow::anyhow!("detect_fast failed: {e}"))?;

    let width = result.width as f64;
    let height = result.height as f64;
    let long_px = (result.width.max(result.height)) as usize;
    let image_center = ((width - 1.0) * 0.5, (height - 1.0) * 0.5);
    let total_detected = result.stars.len();

    if total_detected < 4 {
        anyhow::bail!("too few stars detected ({total_detected}, need ≥4)");
    }

    // 2. Select quad pool: SNR > 10, SNR-ranked descending, cap at 600.
    let image_pts = select_fast_stars(&result.stars, 600);
    let n = image_pts.len();
    if n < 4 {
        anyhow::bail!("too few stars after selection ({n}, need ≥4)");
    }

    // 3. Build image quads once.
    let group_size = group_size_for(n);
    let image_quads = build_quads(&image_pts, n, group_size);
    if image_quads.len() < 3 {
        anyhow::bail!("too few image quads ({}, need ≥3)", image_quads.len());
    }
    let min_quads = minimum_quads(n);

    // Catalog epoch: assume current Gaia (2016.0); observation epoch from hints or now.
    let _cat_epoch = cache.catalog_epoch();
    // Observation epoch for proper-motion propagation: the frame's DATE-OBS
    // (Julian year, via hints.epoch), else a ~2024 default. Wrong-by-construction
    // before this — every frame was propagated to a hardcoded 2024.04.
    let obs_epoch = hints.epoch.unwrap_or(2024.04_f64);

    // 4. autoFOV ladder.
    let rungs = match scale_bounds {
        Some((lo, hi)) => fov_rungs_bounded(long_px, lo, hi),
        None => fov_rungs(long_px, hints),
    };

    // Search center + radius.
    let center = match (hints.ra, hints.dec) {
        (Some(ra), Some(dec)) => Some((ra, dec)),
        _ => None,
    };
    let search_radius = hints.search_radius_deg.unwrap_or(DEFAULT_HINTED_RADIUS_DEG);

    eprintln!(
        "[solvemyastro] {}: {} stars ({}→{} after selection), {} quads (g{}), {} rungs, {}",
        image.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        total_detected, total_detected, n, image_quads.len(), group_size,
        rungs.len(),
        if center.is_some() { "hinted" } else { "blind" }
    );

    // 5. Search. Primary pass uses the full detection pool (unchanged
    //    behaviour). If it finds nothing, a fallback pass retries with
    //    density-balanced quad pools — for frames whose detections
    //    out-resolve the catalog. Frames that solve never reach the
    //    fallback, so it cannot regress them.
    let timing = TimingProfile::default();
    let pixel_cache = PixelCache::new(obs_epoch);
    // Bright cache mirrors the deep cache's per-solve PixelCache pattern
    // — separate instance because each `cone_cached` call is bound to one
    // PixelCache instance + epoch. Only constructed when a bright cache
    // is provided; else `None` and the fallback path is never taken.
    let bright_pixel_cache: Option<PixelCache> =
        caches.bright.map(|_| PixelCache::new(obs_epoch));
    let frame_name = image
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut trace = SolveTrace {
        frame: frame_name.clone(),
        n_detected: total_detected,
        n_selected: n,
        snr_floor: 10.0,
        n_image_quads: image_quads.len(),
        group_size,
        hint_center: center,
        hint_radius_deg: search_radius,
        passes: Vec::with_capacity(2),
        outcome: TraceOutcome::Pending,
        total_ms: 0,
    };
    let truth = known_truth_for(&frame_name);
    let ctx = SearchCtx {
        image_pts: &image_pts,
        image_quads: &image_quads,
        group_size,
        min_quads,
        image_center,
        width,
        height,
        long_px,
        total_detected,
        rungs: &rungs,
        center,
        search_radius,
        cache,
        pixel_cache: &pixel_cache,
        bright_cache: caches.bright,
        bright_pixel_cache: bright_pixel_cache.as_ref(),
        cfg,
        obs_epoch,
        t0,
        cancel,
        timing: &timing,
        truth,
    };
    let mut primary_pass = PassTrace::new(false);
    let primary_result = run_search(&ctx, false, &mut primary_pass);
    trace.passes.push(primary_pass);
    match primary_result {
        Err(e) => {
            trace.outcome = TraceOutcome::Failed(e.to_string());
            trace.total_ms = t0.elapsed().as_millis() as u64;
            eprintln!("{trace}");
            return Err(e);
        }
        Ok(Some(sol)) => {
            trace.outcome = TraceOutcome::Solved {
                ra: sol.wcs.crval.0, dec: sol.wcs.crval.1,
                inliers: sol.matched_stars, rms_px: sol.rms_residual_px,
            };
            trace.total_ms = t0.elapsed().as_millis() as u64;
            if cfg.print_timing {
                timing.print_summary(t0.elapsed(), &frame_name);
            }
            eprintln!("{trace}");
            return Ok(sol);
        }
        Ok(None) => {}
    }
    if is_cancelled() {
        anyhow::bail!("cancelled");
    }
    eprintln!(
        "[solvemyastro] primary search found nothing — \
         retrying with density-balanced quad pools"
    );
    let mut fallback_pass = PassTrace::new(true);
    let fallback_result = run_search(&ctx, true, &mut fallback_pass);
    trace.passes.push(fallback_pass);
    match fallback_result {
        Err(e) => {
            trace.outcome = TraceOutcome::Failed(e.to_string());
            trace.total_ms = t0.elapsed().as_millis() as u64;
            eprintln!("{trace}");
            return Err(e);
        }
        Ok(Some(sol)) => {
            trace.outcome = TraceOutcome::Solved {
                ra: sol.wcs.crval.0, dec: sol.wcs.crval.1,
                inliers: sol.matched_stars, rms_px: sol.rms_residual_px,
            };
            trace.total_ms = t0.elapsed().as_millis() as u64;
            if cfg.print_timing {
                timing.print_summary(t0.elapsed(), &frame_name);
            }
            eprintln!("{trace}");
            return Ok(sol);
        }
        Ok(None) => {}
    }
    if cfg.print_timing {
        timing.print_summary(t0.elapsed(), &frame_name);
    }
    let msg = format!(
        "no trial passed verification across {} FOV rung(s) (incl. density-balanced retry)",
        rungs.len()
    );
    trace.outcome = TraceOutcome::Failed(msg.clone());
    trace.total_ms = t0.elapsed().as_millis() as u64;
    eprintln!("{trace}");
    // Return the structured failure so callers can render a precise reason
    // (the trace is otherwise only printed to stderr and dropped here).
    Err(anyhow::Error::new(crate::diag::SolveFailure {
        class: trace.classify(),
        best_attempt: trace.best_attempt(),
        message: msg,
    }))
}

/// Bundled inputs for one search pass — avoids a 17-argument function.
struct SearchCtx<'a> {
    image_pts: &'a [(f64, f64)],
    image_quads: &'a [crate::quad::Quad],
    group_size: usize,
    min_quads: usize,
    image_center: (f64, f64),
    width: f64,
    height: f64,
    long_px: usize,
    total_detected: usize,
    rungs: &'a [FovRung],
    center: Option<(f64, f64)>,
    search_radius: f64,
    cache: &'a StarCache,
    /// Phase 2: per-solve PM-propagated pixel cache, shared across rayon
    /// trial workers. Created once in `solve()`; reused for every cone
    /// query (primary and verify passes alike).
    pixel_cache: &'a PixelCache,
    /// Optional bright sub-catalog for the quad-matching cone fast path.
    /// `Some` when `Caches::tiered(..)` was passed to `solve()`.
    bright_cache: Option<&'a StarCache>,
    /// Per-solve PM-propagated pixel cache for the bright sub-catalog
    /// (mirror of `pixel_cache`). `Some` iff `bright_cache.is_some()`.
    bright_pixel_cache: Option<&'a PixelCache>,
    cfg: &'a SolveConfig,
    obs_epoch: f64,
    t0: Instant,
    cancel: Option<&'a AtomicBool>,
    timing: &'a TimingProfile,
    /// Optional known-correct WCS for this frame; populated by `solve()` via
    /// `known_truth_for(frame_name)`. Used only by the diagnostic trace to
    /// answer "did any trial seed land near the true answer?" — never
    /// influences solver decisions.
    truth: Option<KnownTruth>,
}

/// One search pass over the autoFOV ladder × square spiral.
///
/// `balance = false` is the primary pass — the full pre-built image quad
/// pool, byte-identical to the original single-pass solver. `balance =
/// true` is the fallback — each trial density-caps the image pool against
/// its catalog cone (see `run_trial`). Returns `Ok(Some)` on a verified
/// solve, `Ok(None)` if no trial passed any rung, `Err` on cancellation.
fn run_search(
    ctx: &SearchCtx,
    balance: bool,
    pass_trace: &mut PassTrace,
) -> Result<Option<SolveSolution>> {
    let &SearchCtx {
        image_pts,
        image_quads,
        group_size,
        min_quads,
        image_center,
        width,
        height,
        long_px,
        total_detected,
        rungs,
        center,
        search_radius,
        cache,
        pixel_cache,
        bright_cache,
        bright_pixel_cache,
        cfg,
        obs_epoch,
        t0,
        cancel,
        timing,
        truth,
    } = ctx;
    let is_cancelled = || cancel.is_some_and(|f| f.load(Ordering::Relaxed));

    // 5a. Spiral search — rayon-parallelised per rung.
    let total_rungs = rungs.len();
    for (rung_idx, rung) in rungs.iter().enumerate() {
        if is_cancelled() {
            anyhow::bail!("cancelled");
        }
        let cells = spiral_cells(center, search_radius, rung.fov_deg);
        let cone_radius = (rung.fov_deg * CONE_OVERSIZE).min(89.0);
        let mut rt = RungTrace::new(rung_idx + 1, total_rungs, rung.fov_deg, cells.len());

        // Spiral-pruning chunk size. The spiral visits cells in order of
        // increasing distance from the hint, so the correct cell is
        // usually near the start. We process the spiral in chunks of
        // `SPIRAL_CHUNK`: each chunk runs all its cells in parallel via
        // rayon, then sequentially verifies any candidates. On the first
        // verified solve we return immediately — skipping every remaining
        // chunk in this rung.
        //
        // Tuning trade-off: smaller chunks give finer early-exit
        // granularity (faster on hinted frames) but introduce more
        // per-chunk barrier overhead (slower on sparse-field frames that
        // must scan all chunks of the right-scale rung). 1024 was picked
        // by corpus_bench A/B: it keeps the big hinted-frame wins
        // (~30-70% reduction) while limiting sparse-field overhead.
        const SPIRAL_CHUNK: usize = 1024;

        // Early-exit flag: set once ANY trial passes verification.
        // Workers in subsequent chunks check it at cell start.
        let any_verified = AtomicBool::new(false);
        let cells_mutex: Mutex<Vec<CellTrace>> = Mutex::new(Vec::with_capacity(cells.len()));

        // Iterate the spiral in chunks. Each chunk does the par_iter +
        // verify pair; on the first verified solve we return immediately.
        // Empty / no-candidate chunks fall through to the next chunk.
        let mut total_candidates_in_rung = 0usize;
        'chunks: for chunk in cells.chunks(SPIRAL_CHUNK) {
            if is_cancelled() {
                anyhow::bail!("cancelled");
            }
            if any_verified.load(Ordering::Relaxed) {
                break 'chunks; // Another chunk already solved (defensive — set on success below).
            }

            let chunk_results: Vec<TrialResult> = chunk
                .par_iter()
                .filter_map(|&cell| {
                    if any_verified.load(Ordering::Relaxed)
                        || cancel.is_some_and(|f| f.load(Ordering::Relaxed))
                    {
                        return None;
                    }
                    let mut ct = CellTrace { ra: cell.0, dec: cell.1, ..Default::default() };
                    let t_cone = Instant::now();
                    let stars = match cone_for_quad_match(
                        cache, pixel_cache,
                        bright_cache, bright_pixel_cache,
                        cell.0, cell.1, cone_radius,
                        cfg.catalog_mag_limit, obs_epoch,
                        cfg.bright_fallback_threshold,
                    ) {
                        Ok(s) => s,
                        Err(_) => {
                            ct.stop = CellStop::ConeTooSparse;
                            cells_mutex.lock().unwrap().push(ct);
                            return None;
                        }
                    };
                    TimingProfile::add(&timing.cone_ns, t_cone.elapsed());
                    timing.cone_calls.fetch_add(1, Ordering::Relaxed);
                    ct.cat_raw = stars.len();
                    if stars.len() < 4 {
                        ct.stop = CellStop::ConeTooSparse;
                        cells_mutex.lock().unwrap().push(ct);
                        return None;
                    }
                    let pool_cap = if balance {
                        MAX_CAT_STARS
                    } else {
                        image_pts.len().min(MAX_CAT_STARS).max(50)
                    };
                    let stars = if stars.len() > pool_cap {
                        let mut s = stars;
                        s.sort_unstable_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap_or(std::cmp::Ordering::Equal));
                        s.truncate(pool_cap);
                        s
                    } else {
                        stars
                    };
                    ct.cat_after_cap = stars.len();
                    let trial = run_trial(
                        image_quads,
                        image_pts,
                        image_center,
                        cell,
                        &stars,
                        group_size,
                        cfg,
                        min_quads,
                        balance,
                        timing,
                        center,
                        &mut ct,
                    );
                    cells_mutex.lock().unwrap().push(ct);
                    trial
                })
                .collect();

            total_candidates_in_rung += chunk_results.len();
            if chunk_results.is_empty() {
                continue 'chunks;
            }

            // Verify this chunk's candidates inline. On the first verified
            // solve, we `return Ok(Some(...))` immediately — skipping all
            // remaining chunks in this rung AND all remaining rungs. This
            // is the spiral-pruning early-exit. Sequential within-chunk
            // verify is fine: candidates per chunk are rare (false quad
            // matches survive the median-ratio filter only occasionally).
            let results = chunk_results;
        'candidates: for trial in results {
            if is_cancelled() {
                anyhow::bail!("cancelled");
            }

        // 6. Verification: cone at seed centre, count inliers.
            let seed_scale = trial.scale_arcsec;
            let image_fov_deg = long_px as f64 * seed_scale / 3600.0;
            let verify_radius = (image_fov_deg * 0.75).min(88.0);

            if !verify_radius.is_finite() || verify_radius <= 0.0 {
                rt.verifies.push(VerifyTrace {
                    seed_ra: 0.0, seed_dec: 0.0, scale_arcsec_per_px: seed_scale,
                    cat_in_fov: 0, seed_inliers: 0, required: 0,
                    seed_rms_px: 0.0,
                    log_odds: f64::NEG_INFINITY,
                    final_inliers: 0, final_rms_px: 0.0,
                    dist_from_truth_deg: None, scale_ratio_to_truth: None,
                    accepted: false,
                    reject_reason: Some("verify_radius_invalid".into()),
                });
                continue 'candidates;
            }

            let (seed_ra, seed_dec) = trial.wcs.pixel_to_sky(image_center.0, image_center.1);
            if !seed_ra.is_finite() || !seed_dec.is_finite() {
                rt.verifies.push(VerifyTrace {
                    seed_ra: 0.0, seed_dec: 0.0, scale_arcsec_per_px: seed_scale,
                    cat_in_fov: 0, seed_inliers: 0, required: 0,
                    seed_rms_px: 0.0,
                    log_odds: f64::NEG_INFINITY,
                    final_inliers: 0, final_rms_px: 0.0,
                    dist_from_truth_deg: None, scale_ratio_to_truth: None,
                    accepted: false,
                    reject_reason: Some("seed_ra_dec_nonfinite".into()),
                });
                continue 'candidates;
            }

            // Truth comparison (diagnostic only): if we have a known-correct
            // WCS for this frame, record per-trial distance + scale ratio so
            // the funnel can show "did we find the right WCS and reject it?".
            let (dist_from_truth_deg, scale_ratio_to_truth) = match truth {
                Some(t) => (
                    Some(great_circle_deg(t.ra, t.dec, seed_ra, seed_dec)),
                    Some(seed_scale / t.scale_arcsec_per_px),
                ),
                None => (None, None),
            };
            let mut v = VerifyTrace {
                seed_ra, seed_dec, scale_arcsec_per_px: seed_scale,
                cat_in_fov: 0, seed_inliers: 0, required: 0,
                seed_rms_px: 0.0,
                log_odds: f64::NEG_INFINITY,
                final_inliers: 0, final_rms_px: 0.0,
                dist_from_truth_deg, scale_ratio_to_truth,
                accepted: false, reject_reason: None,
            };

            // Verification uses bright stars to keep the expected-count denominator sensible.
            // For sparse narrow-FOV fields, step up the mag limit until we have enough
            // reference stars. Phase 1 fix (was: break at first tier with ≥4 stars):
            // the ≥4 floor was too lax — a mislocated cone (false-positive quad seed
            // points to a different sky region) can return 4-15 bright stars in the
            // wrong location and short-circuit the escalation, never reaching the
            // deeper catalog. New floor is `image_pts.len() / 4` (capped at 100,
            // floored at 8) so a healthy density forces continue-to-next-tier.
            // We *always* keep the deepest non-empty result so even if no tier hits
            // the floor we still try verification rather than silently dropping.
            let density_floor = (image_pts.len() / 4).clamp(8, 100);
            let verify_stars = {
                let mut stars = Vec::new();
                let mut hit_floor = false;
                for &mag_lim in &[VERIFY_MAG_LIMIT, 14.0_f32, cfg.catalog_mag_limit] {
                    let t = Instant::now();
                    let res = cache.cone_cached(pixel_cache, seed_ra, seed_dec, verify_radius, mag_lim, obs_epoch);
                    TimingProfile::add(&timing.cone_ns, t.elapsed());
                    timing.cone_calls.fetch_add(1, Ordering::Relaxed);
                    if let Ok(s) = res {
                        if !s.is_empty() {
                            stars = s;
                            if stars.len() >= density_floor {
                                hit_floor = true;
                                break;
                            }
                        }
                    }
                }
                // If no tier hit the floor and even the deepest came back near-empty,
                // the seed-WCS is very likely mislocated. We keep whatever stars we
                // got (≥4 required below) so the trial can still be rejected on
                // inlier count rather than silently abandoned at this stage.
                let _ = hit_floor;
                stars
            };
            v.cat_in_fov = verify_stars.len();
            if verify_stars.len() < 4 {
                v.reject_reason = Some("verify_cone_too_sparse".into());
                rt.verifies.push(v);
                continue 'candidates;
            }

            let tol_px = adaptive_tol_px(seed_scale, BASE_MATCH_ARCSEC);
            let t_ci = Instant::now();
            let (seed_inliers, seed_rss, seed_pairs) = count_inliers(
                &trial.wcs,
                &verify_stars,
                &image_pts,
                width,
                height,
                tol_px,
            );
            TimingProfile::add(&timing.count_inliers_ns, t_ci.elapsed());
            timing.count_inliers_calls.fetch_add(1, Ordering::Relaxed);

            v.seed_inliers = seed_inliers;
            v.seed_rms_px = if seed_inliers > 0 {
                (seed_rss / seed_inliers as f64).sqrt()
            } else {
                0.0
            };
            rt.best_inliers_seen = rt.best_inliers_seen.max(seed_inliers);
            let required = required_inliers(verify_stars.len(), image_pts.len());
            v.required = required;

            // Verification gate: Bayesian log-odds (astrometry.net style) by
            // default; legacy 5%-of-FOV-inlier floor available behind
            // `use_bayesian_verify=false` for emergency revert.
            let gate_decision = if cfg.use_bayesian_verify {
                let inlier_distances: Vec<f64> = seed_pairs.iter()
                    .map(|&((px, py), (dx, dy))| ((px - dx).powi(2) + (py - dy).powi(2)).sqrt())
                    .collect();
                let area_px2 = width * height;
                let lo = crate::verify::log_odds(
                    &inlier_distances,
                    tol_px,
                    verify_stars.len(),
                    area_px2,
                    cfg.distractor_fraction,
                );
                v.log_odds = lo;
                let gate_cfg = crate::verify::BayesianGateConfig {
                    log_odds_threshold: cfg.log_odds_threshold,
                    distractor: cfg.distractor_fraction,
                    ..Default::default()
                };
                crate::verify::accept_bayesian(lo, seed_inliers, v.seed_rms_px, tol_px, &gate_cfg)
            } else if seed_inliers < required {
                crate::verify::AcceptanceDecision::Rejected(
                    format!("seed_inliers {seed_inliers} < required {required} (legacy gate)"),
                )
            } else {
                crate::verify::AcceptanceDecision::Accepted
            };
            if let crate::verify::AcceptanceDecision::Rejected(reason) = gate_decision {
                eprintln!(
                    "[solvemyastro] trial rejected: {reason} \
                     (inliers={seed_inliers}/{required} log_odds={:.1} seed_rms={:.2}px cat_in_fov={})",
                    v.log_odds, v.seed_rms_px, verify_stars.len(),
                );
                v.reject_reason = Some(reason);
                rt.verifies.push(v);
                continue 'candidates;
            }

            // 7. Second solve — full-field affine refit over inlier pairs.
            let final_wcs = if seed_pairs.len() >= 6 {
                // Use looser tolerance for refit gathering.
                let refit_tol = (tol_px * 2.0).min(MAX_MATCH_PX);
                let t = Instant::now();
                let (_, _, refit_pairs) = count_inliers(
                    &trial.wcs,
                    &verify_stars,
                    &image_pts,
                    width,
                    height,
                    refit_tol,
                );
                TimingProfile::add(&timing.count_inliers_ns, t.elapsed());
                timing.count_inliers_calls.fetch_add(1, Ordering::Relaxed);
                let t_ra = Instant::now();
                let refit = refit_affine(&refit_pairs, image_center, &trial.wcs);
                TimingProfile::add(&timing.refit_affine_ns, t_ra.elapsed());
                match refit {
                    Some(refitted) => {
                        // Verify the refit is an improvement.
                        let t = Instant::now();
                        let (ri, _, _) = count_inliers(&refitted, &verify_stars, &image_pts, width, height, tol_px);
                        TimingProfile::add(&timing.count_inliers_ns, t.elapsed());
                        timing.count_inliers_calls.fetch_add(1, Ordering::Relaxed);
                        if ri >= seed_inliers {
                            refitted
                        } else {
                            trial.wcs.clone()
                        }
                    }
                    None => trial.wcs.clone(),
                }
            } else {
                trial.wcs.clone()
            };

            // Recount inliers with final WCS.
            let t = Instant::now();
            let (final_inliers, final_rss, _) = count_inliers(
                &final_wcs,
                &verify_stars,
                &image_pts,
                width,
                height,
                tol_px,
            );
            TimingProfile::add(&timing.count_inliers_ns, t.elapsed());
            timing.count_inliers_calls.fetch_add(1, Ordering::Relaxed);

            if final_inliers < MIN_ABSOLUTE_INLIERS {
                v.final_inliers = final_inliers;
                v.reject_reason = Some(format!("final_inliers {final_inliers} < MIN_ABSOLUTE_INLIERS {MIN_ABSOLUTE_INLIERS}"));
                rt.verifies.push(v);
                continue 'candidates;
            }

            let final_scale = final_wcs.pixel_scale_arcsec();
            let rms_px = if final_inliers > 0 { (final_rss / final_inliers as f64).sqrt() } else { 0.0 };
            let rms_arcsec = rms_px * final_scale;
            let inlier_ratio = final_inliers as f64 / verify_stars.len() as f64;

            v.accepted = true;
            v.final_inliers = final_inliers;
            v.final_rms_px = rms_px;
            rt.verifies.push(v);

            let (cra, cdec) = final_wcs.pixel_to_sky(image_center.0, image_center.1);
            eprintln!(
                "[solvemyastro] SOLVED  RA={cra:.4} Dec={cdec:.4} \
                 scale={final_scale:.3}\"/px rot={:.1}° inliers={final_inliers} rms={rms_px:.2}px",
                final_wcs.field_rotation_deg()
            );

            // 8. SIP fit (optional).
            let (sip_wcs, sip_rms_px, sip_rms_arcsec) = if cfg.fit_sip {
                let t_sip = Instant::now();
                let sip_result = crate::sip::fit_sip(
                    &final_wcs,
                    &verify_stars,
                    &image_pts,
                    width,
                    height,
                    tol_px,
                    cfg.sip_order,
                );
                TimingProfile::add(&timing.fit_sip_ns, t_sip.elapsed());
                match sip_result {
                    Some(w) => {
                        let t = Instant::now();
                        let (si, srss, sip_pairs) = count_inliers(&w, &verify_stars, &image_pts, width, height, tol_px);
                        TimingProfile::add(&timing.count_inliers_ns, t.elapsed());
                        timing.count_inliers_calls.fetch_add(1, Ordering::Relaxed);
                        let srms = if si > 0 { (srss / si as f64).sqrt() } else { f64::MAX };
                        // DIAGNOSTIC (P2b/0008): residual distribution — a few
                        // large residuals ⇒ nearest-neighbour mismatch; a broad
                        // ~rms spread ⇒ centroid systematic / distortion order.
                        let mut resid: Vec<f64> = sip_pairs
                            .iter()
                            .map(|&((px, py), (dx, dy))| ((px - dx).powi(2) + (py - dy).powi(2)).sqrt())
                            .collect();
                        // Residuals are sqrt of (finite px,py - finite dx,dy)^2; non-NaN by upstream
                        // invariants (count_inliers filters NaN positions). Use defensive comparison
                        // so a stray NaN cannot panic this rayon worker.
                        resid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let med = resid.get(resid.len() / 2).copied().unwrap_or(0.0);
                        let p90 = resid.get(resid.len() * 9 / 10).copied().unwrap_or(0.0);
                        let mx = resid.last().copied().unwrap_or(0.0);
                        let gt2 = resid.iter().filter(|&&r| r > 2.0).count();
                        // Keep SIP only if it does not worsen the fit (with the
                        // correct pixel-space math it should strictly improve;
                        // this guards against a pathological field).
                        if srms <= rms_px {
                            eprintln!(
                                "[solvemyastro] SIP applied: linear rms={rms_px:.2}px → sip rms={srms:.2}px ({si} pairs) | med={med:.2} p90={p90:.2} max={mx:.2} >2px={gt2}"
                            );
                            (w, srms, srms * final_scale)
                        } else {
                            eprintln!(
                                "[solvemyastro] SIP rejected: rms {srms:.2}px > linear {rms_px:.2}px"
                            );
                            (final_wcs.clone(), rms_px, rms_arcsec)
                        }
                    }
                    None => (final_wcs.clone(), rms_px, rms_arcsec),
                }
            } else {
                (final_wcs.clone(), rms_px, rms_arcsec)
            };

            let solve_time_ms = t0.elapsed().as_millis() as u64;

            // Mark verified so remaining parallel trials in this rung stop early.
            any_verified.store(true, Ordering::Relaxed);

            // Successful return: snapshot this rung into pass_trace before
            // exiting so the funnel includes the rung that ultimately solved.
            pass_trace.rungs.push(rt);

            return Ok(Some(SolveSolution {
                pixel_scale_arcsec: sip_wcs.pixel_scale_arcsec(),
                field_rotation_deg: sip_wcs.field_rotation_deg(),
                matched_stars: final_inliers,
                total_detected,
                rms_residual_px: sip_rms_px,
                rms_residual_arcsec: sip_rms_arcsec,
                inlier_ratio,
                catalog_used: "gaia_dr3".to_string(),
                algorithm: "solvemyastro".to_string(),
                solve_time_ms,
                wcs: sip_wcs,
            }));
        } // end 'candidates loop
            // This chunk's candidates didn't verify — fall through to the
            // next chunk in the same rung.
        } // end 'chunks loop

        // Either every chunk failed to verify or we broke out due to
        // any_verified being set externally. Snapshot trace counts and
        // push the rung before moving on.
        rt.cells = cells_mutex.into_inner().unwrap();
        rt.candidates_emitted = total_candidates_in_rung;
        pass_trace.rungs.push(rt);
    } // end rung loop

    if is_cancelled() {
        anyhow::bail!("cancelled");
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bounded_ladder_tests {
    use super::{fov_rungs_bounded, FOV_MIN_DEG, FOV_MAX_DEG};

    #[test]
    fn bounded_span_is_clamped_and_descending() {
        // long_px=5496, hint 3.52"/px -> bounds ÷8..×2 = 0.44..7.04"/px.
        let lo = 0.44_f64;
        let hi = 7.04_f64;
        let rungs = fov_rungs_bounded(5496, lo, hi);
        assert!(!rungs.is_empty());
        // Descending in FOV.
        for w in rungs.windows(2) {
            assert!(w[0].fov_deg > w[1].fov_deg, "rungs must descend");
        }
        // Every rung inside the global clamp.
        for r in &rungs {
            assert!(r.fov_deg >= FOV_MIN_DEG - 1e-9 && r.fov_deg <= FOV_MAX_DEG + 1e-9);
        }
        // Reaches near the finer (lo) end: smallest scale ≤ ~1.5× lo.
        let smallest = rungs.last().unwrap().scale_arcsec;
        assert!(smallest <= lo * 1.5 + 1e-6, "must reach the fine end: {smallest}");
    }
}
