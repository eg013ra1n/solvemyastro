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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use anyhow::Result;
use rayon::prelude::*;

use crate::{
    math::great_circle_deg,
    project::project_catalog,
    quad::{build_quads, fit_affine, group_size_for, match_quads},
    select::select_fast_stars,
    wcs::WcsSolution,
    wcs_fit::affine_to_wcs,
    CatalogStar, SolveConfig, SolveHints, SolveSolution, StarCache,
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
            let mut rungs = Vec::new();
            // Centre rung.
            rungs.push(mk(fov.clamp(FOV_MIN_DEG, FOV_MAX_DEG)));
            let mut up = fov;
            let mut down = fov;
            for _ in 0..4 {
                down /= FOV_DIV;
                up *= FOV_DIV;
                if down >= FOV_MIN_DEG {
                    rungs.insert(0, mk(down));
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
        return None;
    }

    let projected = project_catalog(cat_stars, ra0, dec0);
    if projected.len() < 4 {
        return None;
    }
    let cat_quads = build_quads(&projected, projected.len(), gs);
    if cat_quads.is_empty() {
        return None;
    }
    let matches = match_quads(iquads, &cat_quads, cfg.quad_tolerance);
    if matches.len() < mq {
        return None;
    }
    let affine = fit_affine(&matches, iquads, &cat_quads)?;
    let wcs = affine_to_wcs(&affine, image_center, (ra0, dec0));

    // Sanity: pixel scale must be in physical range (0.05"–120"/px).
    let scale = wcs.pixel_scale_arcsec();
    if !scale.is_finite() || scale < 0.05 || scale > 120.0 {
        return None;
    }

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
        // Find the nearest detected star.
        let mut best_dist = f64::MAX;
        let mut best_img = (0.0_f64, 0.0_f64);
        for &(ix, iy) in image_pts {
            let d = ((ix - px).powi(2) + (iy - py).powi(2)).sqrt();
            if d < best_dist {
                best_dist = d;
                best_img = (ix, iy);
            }
        }
        if best_dist <= tol_px {
            count += 1;
            rss += best_dist.powi(2);
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

/// Solve an image to a WCS. Called from `lib.rs::solve`.
///
/// `cancel`: optional cooperative cancellation flag — checked at three
/// responsive points (rung top, inside each rayon trial, before each
/// candidate verify). On observed cancel, returns `Err("cancelled")`.
pub fn solve(
    image: &Path,
    hints: &SolveHints,
    cache: &StarCache,
    cfg: &SolveConfig,
    cancel: Option<&AtomicBool>,
) -> Result<SolveSolution> {
    let t0 = Instant::now();
    let is_cancelled = || cancel.is_some_and(|f| f.load(Ordering::Relaxed));

    // 1. Detect stars.
    let result = astroimage::ImageAnalyzer::new()
        .with_detection_sigma(5.0)
        .with_max_stars(600)
        .with_saturation_fraction(1.0)
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
    // Observation epoch: ~2024 (hardcoded reasonable default; can be refined later).
    let obs_epoch = 2024.04_f64;

    // 4. autoFOV ladder.
    let rungs = fov_rungs(long_px, hints);

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
        cfg,
        obs_epoch,
        t0,
        cancel,
    };
    if let Some(sol) = run_search(&ctx, false)? {
        return Ok(sol);
    }
    if is_cancelled() {
        anyhow::bail!("cancelled");
    }
    eprintln!(
        "[solvemyastro] primary search found nothing — \
         retrying with density-balanced quad pools"
    );
    if let Some(sol) = run_search(&ctx, true)? {
        return Ok(sol);
    }
    anyhow::bail!(
        "no trial passed verification across {} FOV rung(s) \
         (incl. density-balanced retry)",
        rungs.len()
    )
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
    cfg: &'a SolveConfig,
    obs_epoch: f64,
    t0: Instant,
    cancel: Option<&'a AtomicBool>,
}

/// One search pass over the autoFOV ladder × square spiral.
///
/// `balance = false` is the primary pass — the full pre-built image quad
/// pool, byte-identical to the original single-pass solver. `balance =
/// true` is the fallback — each trial density-caps the image pool against
/// its catalog cone (see `run_trial`). Returns `Ok(Some)` on a verified
/// solve, `Ok(None)` if no trial passed any rung, `Err` on cancellation.
fn run_search(ctx: &SearchCtx, balance: bool) -> Result<Option<SolveSolution>> {
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
        cfg,
        obs_epoch,
        t0,
        cancel,
    } = ctx;
    let is_cancelled = || cancel.is_some_and(|f| f.load(Ordering::Relaxed));

    // 5a. Spiral search — rayon-parallelised per rung.
    for rung in rungs {
        if is_cancelled() {
            anyhow::bail!("cancelled");
        }
        let cells = spiral_cells(center, search_radius, rung.fov_deg);
        let cone_radius = (rung.fov_deg * CONE_OVERSIZE).min(89.0);

        // Early-exit flag: set once ANY trial passes verification (set below, not here).
        // We collect ALL trial results per rung so that false quad matches don't block
        // the correct cell — each candidate is verified in turn.
        let any_verified = AtomicBool::new(false);

        // Collect trial results in parallel (rayon `par_iter` + filter_map).
        // No early-exit within the quad-match phase: all cells that produce a
        // plausible affine are kept and verified sequentially below.
        let results: Vec<TrialResult> = cells
            .par_iter()
            .filter_map(|&cell| {
                if any_verified.load(Ordering::Relaxed)
                    || cancel.is_some_and(|f| f.load(Ordering::Relaxed))
                {
                    return None; // Another thread verified, or cancelled.
                }
                let stars = cache
                    .cone(cell.0, cell.1, cone_radius, cfg.catalog_mag_limit, obs_epoch)
                    .ok()?;
                if stars.len() < 4 {
                    return None;
                }
                // ASTAP-style brightness-rank-aligned cap on the catalog quad
                // pool. Without it, a deep catalog (e.g. Gaia G<19, ~1200
                // stars per typical narrow-field cone) builds 6–10× more
                // catalog quads than the image side does, and `match_quads`'
                // median-scale-ratio outlier filter gets pulled toward the
                // false-positive mode — true matches are then dropped as
                // outliers and the affine fit returns garbage.
                //
                // Two-tier cap:
                //
                //   - **Primary pass** (`balance=false`): tight cap at
                //     image depth (count-matched). Restores correct
                //     matching for filtered/narrowband frames where the
                //     image detects far fewer correspondences than the
                //     deep catalog offers.
                //   - **Fallback pass** (`balance=true`): loose cap at
                //     `MAX_CAT_STARS`. For sparse fields (M51-class
                //     narrowband, ~140 detections in a 30′ FOV) the tight
                //     cap cuts too aggressively into the catalog and the
                //     few true matches drop below the inlier floor; the
                //     loose cap lets the existing image-side balance logic
                //     (in `run_trial`) recover with a deeper catalog.
                //
                // Stars are mag-sorted (brightest first) before truncation
                // so the surviving set is the bright-rank-aligned subset
                // most likely to correspond to the image's bright
                // detections, regardless of filter/band.
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
                run_trial(
                    image_quads,
                    image_pts,
                    image_center,
                    cell,
                    &stars,
                    group_size,
                    cfg,
                    min_quads,
                    balance,
                )
            })
            .collect();

        if results.is_empty() {
            continue;
        }

        // Verify each trial candidate in turn; take the first that passes.
        // Iterating sequentially is fine: verification is cheap (one cone lookup +
        // O(n²) nearest-neighbour search), and we expect at most a handful of
        // candidates per rung (false quad matches are rare after the affine sanity check).
        'candidates: for trial in results {
            if is_cancelled() {
                anyhow::bail!("cancelled");
            }

        // 6. Verification: cone at seed centre, count inliers.
            let seed_scale = trial.scale_arcsec;
            let image_fov_deg = long_px as f64 * seed_scale / 3600.0;
            let verify_radius = (image_fov_deg * 0.75).min(88.0);

            if !verify_radius.is_finite() || verify_radius <= 0.0 {
                continue 'candidates;
            }

            let (seed_ra, seed_dec) = trial.wcs.pixel_to_sky(image_center.0, image_center.1);
            if !seed_ra.is_finite() || !seed_dec.is_finite() {
                continue 'candidates;
            }

            // Verification uses bright stars to keep the expected-count denominator sensible.
            // For sparse narrow-FOV fields (< 4 bright stars in the tiny FOV), step up the
            // mag limit in 2-magnitude steps until we have enough reference stars.
            // The deeper the fallback, the more stars we use — but we scale the expected
            // count via `required_inliers` which caps on n_detected anyway.
            let verify_stars = {
                let mut stars = Vec::new();
                for &mag_lim in &[VERIFY_MAG_LIMIT, 14.0_f32, cfg.catalog_mag_limit] {
                    match cache.cone(seed_ra, seed_dec, verify_radius, mag_lim, obs_epoch) {
                        Ok(s) if s.len() >= 4 => { stars = s; break; }
                        Ok(_) | Err(_) => {}
                    }
                }
                stars
            };
            if verify_stars.len() < 4 {
                continue 'candidates;
            }

            let tol_px = adaptive_tol_px(seed_scale, BASE_MATCH_ARCSEC);
            let (seed_inliers, _seed_rss, seed_pairs) = count_inliers(
                &trial.wcs,
                &verify_stars,
                &image_pts,
                width,
                height,
                tol_px,
            );

            let required = required_inliers(verify_stars.len(), image_pts.len());
            if seed_inliers < required {
                eprintln!(
                    "[solvemyastro] trial rejected: {seed_inliers} inliers < {required} required \
                     (expected={} stars in FOV)",
                    verify_stars.len()
                );
                continue 'candidates;
            }

            // 7. Second solve — full-field affine refit over inlier pairs.
            let final_wcs = if seed_pairs.len() >= 6 {
                // Use looser tolerance for refit gathering.
                let refit_tol = (tol_px * 2.0).min(MAX_MATCH_PX);
                let (_, _, refit_pairs) = count_inliers(
                    &trial.wcs,
                    &verify_stars,
                    &image_pts,
                    width,
                    height,
                    refit_tol,
                );
                match refit_affine(&refit_pairs, image_center, &trial.wcs) {
                    Some(refitted) => {
                        // Verify the refit is an improvement.
                        let (ri, _, _) = count_inliers(&refitted, &verify_stars, &image_pts, width, height, tol_px);
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
            let (final_inliers, final_rss, _) = count_inliers(
                &final_wcs,
                &verify_stars,
                &image_pts,
                width,
                height,
                tol_px,
            );

            if final_inliers < MIN_ABSOLUTE_INLIERS {
                continue 'candidates;
            }

            let final_scale = final_wcs.pixel_scale_arcsec();
            let rms_px = if final_inliers > 0 { (final_rss / final_inliers as f64).sqrt() } else { 0.0 };
            let rms_arcsec = rms_px * final_scale;
            let inlier_ratio = final_inliers as f64 / verify_stars.len() as f64;

            let (cra, cdec) = final_wcs.pixel_to_sky(image_center.0, image_center.1);
            eprintln!(
                "[solvemyastro] SOLVED  RA={cra:.4} Dec={cdec:.4} \
                 scale={final_scale:.3}\"/px rot={:.1}° inliers={final_inliers} rms={rms_px:.2}px",
                final_wcs.field_rotation_deg()
            );

            // 8. SIP fit (optional).
            let (sip_wcs, sip_rms_px, sip_rms_arcsec) = if cfg.fit_sip {
                match crate::sip::fit_sip(
                    &final_wcs,
                    &verify_stars,
                    &image_pts,
                    width,
                    height,
                    tol_px,
                    cfg.sip_order,
                ) {
                    Some(w) => {
                        let (si, srss, sip_pairs) = count_inliers(&w, &verify_stars, &image_pts, width, height, tol_px);
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
    } // end rung loop

    if is_cancelled() {
        anyhow::bail!("cancelled");
    }
    Ok(None)
}
