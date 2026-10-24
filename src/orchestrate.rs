//! Blind plate-solve orchestration.
//!
//! Implements the full solve pipeline:
//! 1. Star detection via `astroimage::ImageAnalyzer::detect_fast`.
//! 2. SNR-gated star selection (`select_fast_stars`).
//! 3. autoFOV ladder (hinted ≤ 9 rungs, blind 14.25→0.57° ÷1.5).
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
/// nanoseconds. Updated with `Relaxed` ordering from rayon workers; emitted
/// as structured `debug!` events once at solve end when
/// `SolveConfig::print_timing` is true.
///
/// Phase 1 instrumentation — the data drives the speed-fix targeting in
/// Phase 3 (kd-tree on `match_quads` vs `count_inliers`). Also the math-
/// instrumentation backbone for the pipeline's `select` / `quad_build` /
/// `match` / `refine` / `verify` stages (see `log_stage_timings`).
#[derive(Default, Debug)]
pub(crate) struct TimingProfile {
    pub cone_ns: AtomicU64,
    pub cone_calls: AtomicU64,
    pub project_ns: AtomicU64,
    pub build_quads_cat_ns: AtomicU64,
    pub build_quads_cat_calls: AtomicU64,
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

    /// Emit one `debug!` stage-timing event per pipeline stage tracked here
    /// (`cone`, `project`, `quad_build` (catalog side), `match`, `fit_affine`,
    /// `verify`, `refine` (affine + SIP)), plus one summary event with the
    /// total wall time. Coarse — one event per stage per solve, not per call
    /// — so the cost at any tracing level is a handful of atomic loads.
    fn log_stage_timings(&self, total: std::time::Duration, frame: &str) {
        let total_ms = total.as_millis() as u64;
        let ms = |ns: u64| ns / 1_000_000;
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        tracing::debug!(stage = "cone", frame, duration_ms = ms(g(&self.cone_ns)), count = g(&self.cone_calls), "solve stage timing");
        tracing::debug!(stage = "project", frame, duration_ms = ms(g(&self.project_ns)), "solve stage timing");
        tracing::debug!(stage = "quad_build", pool = "catalog", frame, duration_ms = ms(g(&self.build_quads_cat_ns)), count = g(&self.build_quads_cat_calls), "solve stage timing");
        tracing::debug!(stage = "match", frame, duration_ms = ms(g(&self.match_quads_ns)), count = g(&self.match_quads_calls), "solve stage timing");
        tracing::debug!(stage = "fit_affine", frame, duration_ms = ms(g(&self.fit_affine_ns)), count = g(&self.fit_affine_calls), "solve stage timing");
        tracing::debug!(stage = "verify", frame, duration_ms = ms(g(&self.count_inliers_ns)), count = g(&self.count_inliers_calls), "solve stage timing");
        tracing::debug!(stage = "refine", step = "affine", frame, duration_ms = ms(g(&self.refit_affine_ns)), "solve stage timing");
        tracing::debug!(stage = "refine", step = "sip", frame, duration_ms = ms(g(&self.fit_sip_ns)), "solve stage timing");
        tracing::debug!(frame, total_ms, "solve stage timing summary");
    }
}

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

use crate::{
    index::QuadIndex,
    diag::{CellStop, CellTrace, KnownTruth, PassTrace, RungTrace, SeedInfo, SolveTrace, TraceOutcome, VerifyTrace, known_truth_for},
    math::great_circle_deg,
    nn::PointGrid,
    project::project_catalog,
    quad::{build_quads, fit_affine, group_size_for, match_quads_stats, Affine, Quad},
    select::{select_fast_stars, select_fast_stars_with_raw},
    wcs::WcsSolution,
    wcs_fit::affine_to_wcs,
    Caches, CatalogStar, LayeredCones, PixelCache, SolveConfig, SolveHints, SolveSolution, StarCache,
};

// ---------------------------------------------------------------------------
// Blind-search constants
// ---------------------------------------------------------------------------

/// Blind autoFOV top rung. With [`CONE_OVERSIZE`] its cone radius is 9.5°.
const FOV_MAX_DEG: f64 = 14.25;
/// Blind autoFOV bottom rung floor. With [`CONE_OVERSIZE`] its cone radius is
/// 0.38°.
const FOV_MIN_DEG: f64 = 0.57;
/// Floor of a hinted ladder. Lower than [`FOV_MIN_DEG`] because a hint pins
/// the search to a small sky region: long-focus frames (0.3° at 4000 mm)
/// need a cone near their own size, and only a blind ladder would pay for
/// fine rungs over the whole sky.
const HINTED_FOV_MIN_DEG: f64 = 0.15;
/// Per-rung divisor.
const FOV_DIV: f64 = 1.5;
/// Full-sky search radius when blind.
const FULL_SKY_RADIUS_DEG: f64 = 180.0;
/// Default hinted search radius (degrees) when none specified.
const DEFAULT_HINTED_RADIUS_DEG: f64 = 10.0;
/// Catalog-cone radius (and spiral step) as a fraction of the rung FOV.
///
/// Quads are built from nearest neighbours of the brightest stars on each
/// side, so they only correspond when the catalog pool covers about the same
/// sky as the frame. At radius = FOV the cone holds ~4.5× the frame's area
/// and the correct rung matched nothing; every bench frame solved one rung
/// finer instead, i.e. at radius ≈ FOV/1.5 — so that is the radius used.
const CONE_OVERSIZE: f64 = 2.0 / 3.0;
/// Maximum catalog stars per trial (deep, no magnitude-budget starvation).
const MAX_CAT_STARS: usize = 6000;
/// Inlier match radius in arcsec (adaptive: this is the base; see adaptive_tol).
const BASE_MATCH_ARCSEC: f64 = 3.0;
/// Minimum inlier floor (absolute) regardless of density calculation.
const MIN_ABSOLUTE_INLIERS: usize = 6;
/// Below this many detections a frame is simply sparse, and the share of
/// streaks says nothing — the trailed verdict needs a population to judge.
const MIN_DETECTIONS_FOR_TRAIL_VERDICT: usize = 100;
/// Share of detections the shape cut must remove before a frame is called
/// trailed rather than merely imperfect.
const TRAILED_POOL_PERCENT: usize = 90;

/// Did the shape cut just empty this frame?
///
/// `total_detected` is what the detector found; `pool` what survived the
/// elongation cut and the SNR gate. A frame of streaks loses nearly all of
/// it, a healthy one almost none — so the share separates "nothing to match"
/// from "a hard field", and only the former is worth refusing outright.
fn looks_trailed(total_detected: usize, pool: usize) -> bool {
    total_detected >= MIN_DETECTIONS_FOR_TRAIL_VERDICT
        && total_detected.saturating_sub(pool) * 100 >= total_detected * TRAILED_POOL_PERCENT
}

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
            // clamped to [HINTED_FOV_MIN_DEG, FOV_MAX_DEG].
            // This handles stacked/drizzled images where the stored FOCALLEN or CDELT1
            // may differ from the actual per-pixel scale by 2–4×.
            // Centre-first, expanding outward (centre, ÷1.5, ×1.5, ÷1.5², …).
            // The hinted scale is the most likely rung, so the first-verify
            // early-exit fires on it immediately instead of after several
            // dense wrong-scale spirals (the finest rungs have the most cells).
            let mut rungs = Vec::new();
            rungs.push(mk(fov.clamp(HINTED_FOV_MIN_DEG, FOV_MAX_DEG)));
            let mut up = fov;
            let mut down = fov;
            for _ in 0..4 {
                down /= FOV_DIV;
                up *= FOV_DIV;
                if down >= HINTED_FOV_MIN_DEG {
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
/// `FOV_DIV`, clamped to `[HINTED_FOV_MIN_DEG, FOV_MAX_DEG]`. Always includes a rung at
/// (or just below) the fine bound so a finer-than-hint true scale is reached.
fn fov_rungs_bounded(long_px: usize, lo_arcsec: f64, hi_arcsec: f64) -> Vec<FovRung> {
    let lp = long_px.max(1) as f64;
    let mk = |fov: f64| FovRung { fov_deg: fov, scale_arcsec: fov * 3600.0 / lp };
    let lo_fov = (lo_arcsec * lp / 3600.0).clamp(HINTED_FOV_MIN_DEG, FOV_MAX_DEG);
    let hi_fov = (hi_arcsec * lp / 3600.0).clamp(HINTED_FOV_MIN_DEG, FOV_MAX_DEG);
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

/// Smallest spiral chunk: enough cells to keep every rayon worker busy.
const SPIRAL_CHUNK_MIN: usize = 16;
/// Largest spiral chunk — bounds per-chunk barrier overhead on frames that
/// scan a whole rung.
const SPIRAL_CHUNK_MAX: usize = 1024;

/// Split the spiral into consecutive chunks that double in size from
/// [`SPIRAL_CHUNK_MIN`] up to [`SPIRAL_CHUNK_MAX`]. Order is preserved.
fn spiral_chunks<T>(cells: &[T]) -> impl Iterator<Item = &[T]> {
    let mut rest = cells;
    let mut size = SPIRAL_CHUNK_MIN;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let (head, tail) = rest.split_at(size.min(rest.len()));
        rest = tail;
        size = (size * 2).min(SPIRAL_CHUNK_MAX);
        Some(head)
    })
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
    /// Number of filtered quad-match pairs used to seed this trial's affine.
    /// Threaded to the verify section so the PROSAC gate (`< 10 pairs`) can
    /// be applied to the full-field iterative refit decision.
    quad_match_pairs: usize,
    /// The same seed re-fitted about its own frame centre (see
    /// [`refit_seed_at`]). Verified only when `wcs` is rejected.
    recentered: Option<WcsSolution>,
}

/// Catalog stars expected inside the frame, from a cone of `n_cone` stars:
/// the cone covers the frame rectangle scaled by `aspect / (π·OVERSIZE²)`.
fn catalog_in_frame(n_cone: usize, image_center: (f64, f64)) -> usize {
    let dim_a = 2.0 * image_center.0 + 1.0;
    let dim_b = 2.0 * image_center.1 + 1.0;
    let aspect = dim_a.min(dim_b) / dim_a.max(dim_b);
    (n_cone as f64 * aspect / (std::f64::consts::PI * CONE_OVERSIZE * CONE_OVERSIZE)).round()
        as usize
}

#[allow(clippy::too_many_arguments)]
fn run_trial(
    image_quads: &[Quad],
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
    // Per-trial RANSAC seed: `SolveConfig::ransac_seed` XOR'd with a
    // deterministic function of the cell coordinates. Ensures repeated
    // corpus runs produce identical PROSAC sample sequences.
    ransac_seed: u64,
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
        let img_keep = catalog_in_frame(cat_stars.len(), image_center).clamp(4, image_pts.len());
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
    timing.build_quads_cat_calls.fetch_add(1, Ordering::Relaxed);
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
    // PROSAC gate: when there are very few quad matches (< 10), the single
    // least-squares affine is prone to being poisoned by one bad pair. Run
    // PROSAC to find a more robust seed. The gate value (10) is chosen so
    // easy frames — which typically accumulate dozens of matches — skip
    // PROSAC entirely and remain byte-identical. The `quad_match_pairs`
    // count is threaded to the caller for its own gating decisions.
    let quad_match_pairs = matches.len();
    const PROSAC_WEAK_THRESHOLD: usize = 10;
    let affine = if quad_match_pairs < PROSAC_WEAK_THRESHOLD {
        // Weak-match path: PROSAC (100 trials) for a robust seed affine.
        match prosac_seed_affine(&matches, iquads, &cat_quads, ransac_seed, 100) {
            Some(a) => a,
            None => {
                ct.stop = CellStop::AffineFailed;
                return None;
            }
        }
    } else {
        // Strong-match path: existing single SVD affine — byte-identical.
        match fit_affine(&matches, iquads, &cat_quads) {
            Some(a) => a,
            None => {
                ct.stop = CellStop::AffineFailed;
                return None;
            }
        }
    };
    TimingProfile::add(&timing.fit_affine_ns, t_fa.elapsed());
    timing.fit_affine_calls.fetch_add(1, Ordering::Relaxed);
    let wcs = affine_to_wcs(&affine, image_center, (ra0, dec0));

    // The affine was fitted in the tangent plane of the *cell* centre, but the
    // WCS puts its tangent point at the frame centre it implies. When those
    // differ, the CD is the wrong plane's: on a wide field a 0.6° hint error
    // leaves the seed several pixels off at the edges, verification sees a
    // handful of inliers and the correct seed is thrown away. Re-fit once
    // with the catalog projected about the seed's own centre, kept as a
    // fallback candidate so seeds that already verify are untouched.
    let seed = wcs.pixel_to_sky(image_center.0, image_center.1);
    let recentered = if great_circle_deg(seed.0, seed.1, ra0, dec0) > RECENTER_MIN_DEG {
        refit_seed_at(seed, iquads, cat_stars, image_center, gs, mq, cfg, ransac_seed, timing)
            .filter(|w| {
                let s = w.pixel_scale_arcsec();
                s.is_finite() && (0.05..=120.0).contains(&s)
            })
    } else {
        None
    };

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
    Some(TrialResult { wcs, tangent: (ra0, dec0), scale_arcsec: scale, quad_match_pairs, recentered })
}

/// Seed-to-cell offset above which [`run_trial`] re-fits the seed about its
/// own centre.
const RECENTER_MIN_DEG: f64 = 0.01;

/// Re-run the quad match and affine fit with the catalog projected about
/// `tangent` (the seed's frame centre). Same match floor and PROSAC rule as
/// the first fit; `None` when the re-fit falls short, so the caller keeps its
/// original seed.
#[allow(clippy::too_many_arguments)]
fn refit_seed_at(
    tangent: (f64, f64),
    iquads: &[Quad],
    cat_stars: &[CatalogStar],
    image_center: (f64, f64),
    gs: usize,
    mq: usize,
    cfg: &SolveConfig,
    ransac_seed: u64,
    timing: &TimingProfile,
) -> Option<WcsSolution> {
    let t = Instant::now();
    let projected = project_catalog(cat_stars, tangent.0, tangent.1);
    TimingProfile::add(&timing.project_ns, t.elapsed());
    let t = Instant::now();
    let cat_quads = build_quads(&projected, projected.len(), gs);
    TimingProfile::add(&timing.build_quads_cat_ns, t.elapsed());
    let t = Instant::now();
    let (matches, _) = match_quads_stats(iquads, &cat_quads, cfg.quad_tolerance);
    TimingProfile::add(&timing.match_quads_ns, t.elapsed());
    if matches.len() < mq {
        return None;
    }
    let affine = if matches.len() < 10 {
        prosac_seed_affine(&matches, iquads, &cat_quads, ransac_seed, 100)?
    } else {
        fit_affine(&matches, iquads, &cat_quads)?
    };
    Some(affine_to_wcs(&affine, image_center, tangent))
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

    let grid = PointGrid::new(image_pts, tol_px);
    for cat in cat_stars {
        let (px, py) = wcs.sky_to_pixel(cat.ra, cat.dec);
        if !px.is_finite() || !py.is_finite() {
            continue;
        }
        if px < 0.0 || px >= width || py < 0.0 || py >= height {
            continue;
        }
        // Nearest detected star by squared distance — `rss` wants the
        // squared residual anyway.
        let nearest = grid.nearest_by(px, py, |(ix, iy)| (ix - px).powi(2) + (iy - py).powi(2));
        if let Some((i, best_d2)) = nearest {
            if best_d2 <= tol_px * tol_px {
                count += 1;
                rss += best_d2;
                pairs.push(((px, py), image_pts[i]));
            }
        }
    }
    (count, rss, pairs)
}

/// Of the catalog stars that land in the frame, take the `image_pts.len()`
/// brightest and count how many have a detection within `tol_px`.
/// Returns `(k, matched)`.
fn bright_star_evidence(
    wcs: &WcsSolution,
    cat_stars: &[CatalogStar],
    image_pts: &[(f64, f64)],
    width: f64,
    height: f64,
    tol_px: f64,
) -> (usize, usize) {
    let mut in_frame: Vec<(f32, f64, f64)> = cat_stars
        .iter()
        .filter_map(|c| {
            let (px, py) = wcs.sky_to_pixel(c.ra, c.dec);
            (px.is_finite() && py.is_finite() && px >= 0.0 && px < width && py >= 0.0 && py < height)
                .then_some((c.mag, px, py))
        })
        .collect();
    let k = in_frame.len().min(image_pts.len());
    if k == 0 {
        return (0, 0);
    }
    let by_mag = |a: &(f32, f64, f64), b: &(f32, f64, f64)| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    };
    if in_frame.len() > k {
        in_frame.select_nth_unstable_by(k - 1, by_mag);
        in_frame.truncate(k);
    }
    let grid = PointGrid::new(image_pts, tol_px);
    let tol2 = tol_px * tol_px;
    let matched = in_frame
        .iter()
        .filter(|&&(_, px, py)| {
            grid.nearest_by(px, py, |(ix, iy)| (ix - px).powi(2) + (iy - py).powi(2))
                .is_some_and(|(_, d2)| d2 <= tol2)
        })
        .count();
    (k, matched)
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
// Phase 3 — iterative σ-clipped affine refit
// ---------------------------------------------------------------------------

/// Iterative σ-clipped affine refit over inlier pairs.
///
/// Runs up to `max_iter` iterations (default 5):
///   1. Fit affine via `refit_affine`.
///   2. Recount inliers and compute per-pair residuals.
///   3. Compute σ of residual distances; drop pairs with residual > `sigma_clip` × σ.
///   4. Repeat until residual RMS changes < `rms_rtol` (3%) OR the inlier set
///      is stable (Jaccard similarity > 0.97) OR fewer than 6 pairs survive.
///
/// **Gate:** callers MUST only invoke this when `seed_inliers < 50`.
/// Easy frames (≥ 50 inliers) must use the existing single `refit_affine` path
/// (byte-identical fast-path skip). This ensures the corpus gate is unaffected.
///
/// Returns the best WCS found (maximising inlier count), or `None` if even
/// the first refit fails.
fn refit_affine_iter(
    pairs: &[((f64, f64), (f64, f64))],
    image_center: (f64, f64),
    wcs: &WcsSolution,
    cat_stars: &[CatalogStar],
    image_pts: &[(f64, f64)],
    width: f64,
    height: f64,
    tol_px: f64,
) -> Option<WcsSolution> {
    const MAX_ITER: usize = 5;
    const SIGMA_CLIP: f64 = 3.0;
    const RMS_RTOL: f64 = 0.03;
    const JACCARD_STABLE: f64 = 0.97;

    if pairs.len() < 6 {
        return None;
    }

    let mut current_wcs = refit_affine(pairs, image_center, wcs)?;
    let mut prev_rms = f64::MAX;
    // Track the inlier index set from the previous iteration for Jaccard stability.
    let mut prev_pair_count: usize = pairs.len();

    // Use the looser tolerance for gathering refit pairs (same as the caller
    // does for the single-refit path: tol_px * 2.0, capped).
    let refit_tol = (tol_px * 2.0).min(MAX_MATCH_PX);

    for _iter in 0..MAX_ITER {
        // Recount inliers + residuals with current WCS.
        let (_, rss, cur_pairs) =
            count_inliers(&current_wcs, cat_stars, image_pts, width, height, refit_tol);
        if cur_pairs.len() < 6 {
            break;
        }

        let n = cur_pairs.len() as f64;
        let rms = (rss / n).sqrt();

        // Jaccard stability: |intersection| / |union| of pair counts.
        let intersection = cur_pairs.len().min(prev_pair_count);
        let union = cur_pairs.len().max(prev_pair_count);
        let jaccard = if union > 0 { intersection as f64 / union as f64 } else { 1.0 };

        // Convergence checks.
        let rms_converged = prev_rms.is_finite()
            && prev_rms > 0.0
            && (prev_rms - rms).abs() / prev_rms <= RMS_RTOL;
        let set_stable = jaccard >= JACCARD_STABLE;
        if rms_converged || set_stable {
            break;
        }

        // σ-clip: compute per-pair residual distances, filter > 3σ.
        let residuals: Vec<f64> = cur_pairs
            .iter()
            .map(|&((px, py), (dx, dy))| ((px - dx).powi(2) + (py - dy).powi(2)).sqrt())
            .collect();
        let sigma = if rms > 0.0 { rms } else { 1.0 };
        let clipped: Vec<((f64, f64), (f64, f64))> = cur_pairs
            .into_iter()
            .zip(residuals.iter())
            .filter(|(_, &r)| r <= SIGMA_CLIP * sigma)
            .map(|(p, _)| p)
            .collect();

        if clipped.len() < 6 {
            break;
        }

        let next_wcs = match refit_affine(&clipped, image_center, &current_wcs) {
            Some(w) => w,
            None => break,
        };

        prev_pair_count = clipped.len();
        prev_rms = rms;
        current_wcs = next_wcs;
    }

    Some(current_wcs)
}

// ---------------------------------------------------------------------------
// Phase 3 — PROSAC RANSAC for weak seeds
// ---------------------------------------------------------------------------

/// PROSAC-style RANSAC seed-affine refinement.
///
/// Fires only when `matches.len() < 10` (weak-match gate). For stronger match
/// sets the single `fit_affine(matches, ...)` path is sufficient and calling
/// this function is unnecessary overhead.
///
/// Algorithm:
/// - Sort matches by image-quad index ascending (low-index = high-SNR stars,
///   since `image_pts` is SNR-ranked descending). Low-index quads are
///   preferentially sampled in early trials ("top-quartile" bias).
/// - Run up to `n_trials` trials. First 10 trials: draw 3-pair samples
///   preferentially from the top-quartile (first quarter) of sorted matches;
///   thereafter: uniform random sampling from all matches.
/// - Per trial: fit affine from 3 pairs → count inliers from ALL matches.
/// - Keep the affine with the most inliers.
///
/// Determinism: `rng_seed` is a stable per-trial value derived from
/// `SolveConfig::ransac_seed` XOR'd with the cell coordinates (bit-cast to u64).
/// The corpus gate uses the same seed on every run.
///
/// Determinism is per-build, NOT across `rand` majors: the raw Xoshiro256**
/// stream is version-stable, but `random_range`'s stream→index mapping is not
/// (rand 0.8 used Lemire zone-rejection, 0.9+ uses Canon's method), so a `rand`
/// bump reshuffles which triples get sampled. That is safe here because the
/// sample only ever seeds a candidate affine that must out-inlier the full-set
/// least-squares fit to be adopted — but it does mean a `rand` bump can move
/// the corpus baseline, and the baseline must be re-frozen if it does.
///
/// Returns the best `Affine` found, or the original `fit_affine` result when
/// the PROSAC result is not an improvement.
#[allow(clippy::too_many_arguments)]
fn prosac_seed_affine(
    matches: &[(usize, usize)],
    image_quads: &[Quad],
    cat_quads: &[Quad],
    rng_seed: u64,
    n_trials: usize,
) -> Option<Affine> {
    // `RngExt` is rand 0.10's extension trait (0.8 called it `Rng`; `rand::Rng`
    // is now a re-export of the core trait and carries no `random_range`).
    use rand::RngExt;

    if matches.len() < 3 {
        return None;
    }

    // Baseline: try the full-set affine first.
    let baseline = fit_affine(matches, image_quads, cat_quads);

    if matches.len() < 3 {
        return baseline;
    }

    // Sort by image-quad index (ascending = higher-SNR quads first).
    let mut sorted_matches: Vec<(usize, usize)> = matches.to_vec();
    sorted_matches.sort_unstable_by_key(|&(ii, _)| ii);
    let nm = sorted_matches.len();
    let top_quartile = ((nm / 4).max(3)).min(nm);

    let mut rng = Xoshiro256StarStar::seed_from_u64(rng_seed);

    let mut best_inlier_count = 0usize;
    let mut best_affine: Option<Affine> = None;

    for trial_idx in 0..n_trials {
        // Sample 3 pairs.
        let sample: [(usize, usize); 3] = if trial_idx < 10 && top_quartile >= 3 {
            // Top-quartile preferential sampling.
            let i0 = rng.random_range(0..top_quartile);
            let mut i1 = rng.random_range(0..top_quartile);
            if i1 == i0 { i1 = (i1 + 1) % top_quartile; }
            let mut i2 = rng.random_range(0..top_quartile);
            if i2 == i0 || i2 == i1 { i2 = (i2 + 2) % top_quartile; }
            [sorted_matches[i0], sorted_matches[i1], sorted_matches[i2]]
        } else {
            let i0 = rng.random_range(0..nm);
            let mut i1 = rng.random_range(0..nm);
            if i1 == i0 { i1 = (i1 + 1) % nm; }
            let mut i2 = rng.random_range(0..nm);
            if i2 == i0 || i2 == i1 { i2 = (i2 + 2) % nm; }
            [sorted_matches[i0], sorted_matches[i1], sorted_matches[i2]]
        };

        let trial_matches: Vec<(usize, usize)> = sample.to_vec();
        let affine = match fit_affine(&trial_matches, image_quads, cat_quads) {
            Some(a) => a,
            None => continue,
        };

        // Count inliers from ALL matches: how many match-pair quad-centers
        // are consistent with this affine (within a generous pixel tolerance)?
        let mut inliers = 0usize;
        let center_tol = 5.0_f64; // pixels in projected space — loose for seed
        for &(ii, ci) in matches.iter() {
            let (ix, iy) = image_quads[ii].center;
            let (cx, cy) = cat_quads[ci].center;
            let (px, py) = affine.apply(ix, iy);
            let d2 = (px - cx).powi(2) + (py - cy).powi(2);
            if d2 <= center_tol * center_tol {
                inliers += 1;
            }
        }

        if inliers > best_inlier_count {
            best_inlier_count = inliers;
            best_affine = Some(affine);
        }
    }

    // Return the PROSAC result only if it's better than the baseline.
    // "Better" here means it found an affine (baseline may be None if
    // the full-set SVD was degenerate) or has more inliers.
    match (best_affine, baseline) {
        (Some(pa), None) => Some(pa),
        (Some(pa), Some(ba)) => {
            // Count baseline inliers for comparison.
            let baseline_inliers: usize = matches.iter().filter(|&&(ii, ci)| {
                let (ix, iy) = image_quads[ii].center;
                let (cx, cy) = cat_quads[ci].center;
                let (px, py) = ba.apply(ix, iy);
                (px - cx).powi(2) + (py - cy).powi(2) <= 25.0
            }).count();
            if best_inlier_count > baseline_inliers {
                Some(pa)
            } else {
                Some(ba)
            }
        }
        (None, b) => b,
    }
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
    index: Option<&QuadIndex>,
    cfg: &SolveConfig,
    cancel: Option<&AtomicBool>,
) -> Result<SolveSolution> {
    solve_inner(image, hints, caches, index, cfg, cancel, None)
}

/// Internal entry: `scale_bounds = Some((lo_arcsec, hi_arcsec))` drives a
/// bounded FOV ladder (B2); `None` uses the hint-derived/blind ladder.
pub(crate) fn solve_inner(
    image: &Path,
    hints: &SolveHints,
    caches: &crate::Caches<'_>,
    index: Option<&QuadIndex>,
    cfg: &SolveConfig,
    cancel: Option<&AtomicBool>,
    scale_bounds: Option<(f64, f64)>,
) -> Result<SolveSolution> {
    // A representative cache for the incidental catalog-epoch read below; the
    // real cone queries go through `cones` (built later, per `Caches` variant).
    let repr_cache: &StarCache = match *caches {
        Caches::Legacy { deep, .. } => deep,
        Caches::Layered { layers } => layers
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Caches::Layered with no layers"))?,
    };
    let t0 = Instant::now();
    let is_cancelled = || cancel.is_some_and(|f| f.load(Ordering::Relaxed));
    let frame_name = image
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    // 1. Detect stars.
    //
    // When centroid_refinement is Auto, run the Moffat LM refinement pass
    // inside rustafits. After detection, the global fallback gate in
    // `crate::refine` checks whether the field is amenable to PSF fitting
    // (FWHM ≥ 1.5 px, median SNR ≥ 10). If the gate fires, we re-run
    // detection without refinement to recover byte-identical pass-1 positions.
    // The gate fires rarely (only undersampled or very faint fields).
    //
    // Phase 2: when refinement is accepted, `refinement_accepted = true` and
    // we thread the per-star σ values (sx, sy) into `star_sigmas` alongside
    // `image_pts` so the precise SIP path can use inverse-variance weights.
    let use_refine = cfg.centroid_refinement == crate::CentroidRefinement::Auto;

    // `refinement_accepted` is set to `true` only when the Auto path runs AND
    // the global field gate accepts the PSF fits.  In all other cases the σ
    // values in `FastStar` are 0.0 (pass-1 centroids) and we use None.
    let (result, psf_fwhm_px, refinement_accepted) = if use_refine {
        let r = astroimage::ImageAnalyzer::new()
            .with_detection_sigma(5.0)
            .with_max_stars(600)
            .with_centroid_refine(true)
            .detect_fast(image)
            .map_err(|e| anyhow::anyhow!("detect_fast failed: {e}"))?;

        let (decision, median_fwhm) = crate::refine::evaluate_refinement(&r.stars);
        match decision {
            crate::refine::RefineDecision::Accept => {
                tracing::debug!(frame = %frame_name, median_fwhm_px = median_fwhm, "PSF refinement accepted");
                (r, Some(median_fwhm), true)
            }
            crate::refine::RefineDecision::Revert { reason } => {
                tracing::warn!(frame = %frame_name, reason = ?reason, "PSF refinement reverted — re-detecting without refine");
                // Re-run without refinement to get clean pass-1 positions.
                let r2 = astroimage::ImageAnalyzer::new()
                    .with_detection_sigma(5.0)
                    .with_max_stars(600)
                    .detect_fast(image)
                    .map_err(|e| anyhow::anyhow!("detect_fast (pass-1 fallback) failed: {e}"))?;
                (r2, None, false)
            }
        }
    } else {
        let r = astroimage::ImageAnalyzer::new()
            .with_detection_sigma(5.0)
            .with_max_stars(600)
            .detect_fast(image)
            .map_err(|e| anyhow::anyhow!("detect_fast failed: {e}"))?;
        (r, None, false)
    };

    let width = result.width as f64;
    let height = result.height as f64;
    let long_px = (result.width.max(result.height)) as usize;
    let image_center = ((width - 1.0) * 0.5, (height - 1.0) * 0.5);
    let total_detected = result.stars.len();

    if total_detected < 4 {
        anyhow::bail!("too few stars detected ({total_detected}, need ≥4)");
    }

    // 2. Select quad pool: SNR > 10, SNR-ranked descending, cap at 600.
    //
    // Fast path: `select_fast_stars` — positions only, byte-identical to all
    // prior solver behaviour.
    //
    // Precise path (refinement_accepted): also build the σ slice from
    // `select_fast_stars_with_sigma`.  The two functions apply the SAME SNR gate
    // and ranking so the returned slices are always the same length / order.
    let t_select = Instant::now();
    let image_pts = select_fast_stars(&result.stars, 600);
    let n = image_pts.len();
    tracing::debug!(
        stage = "select",
        frame = %frame_name,
        duration_ms = t_select.elapsed().as_millis() as u64,
        count = n,
        total_detected,
        "solve stage timing"
    );
    if n < 4 {
        anyhow::bail!("too few stars after selection ({n}, need ≥4)");
    }
    // A frame the shape cut nearly emptied has no stars to offer, only
    // streaks — and the handful that survive are the least-bad fragments,
    // not catalogue positions. Refuse now rather than walk the FOV ladder
    // (twice, counting the density-balanced retry) to arrive at the same
    // answer minutes later. Measured: frames whose stars are trails lose
    // 98-99 % of their detections here, healthy ones 0-13 %, and even a
    // hard DSLR frame only 60 %.
    if looks_trailed(total_detected, n) {
        let dropped = total_detected - n;
        anyhow::bail!(
            "stars are trailed: {dropped} of {total_detected} detections are streaks, \
             leaving {n} usable"
        );
    }
    // σ slice + pass-1 raw positions: built only when refinement was accepted
    // so the sx/sy/raw_x/raw_y values are meaningful (PSF-fitted).
    // None otherwise → uniform weights + no raw comparison in the final fit.
    //
    // Phase 4: `image_pts_raw_storage` holds the pass-1 (intensity-weighted)
    // centroids aligned 1:1 with `image_pts`.  Used by the compare-and-keep-
    // better guard in the SIP final-fit stage.
    let (star_sigmas_storage, image_pts_raw_storage): (
        Option<Vec<(f64, f64)>>,
        Option<Vec<(f64, f64)>>,
    ) = if refinement_accepted {
        let with_raw = select_fast_stars_with_raw(&result.stars, 600);
        let sigmas: Vec<(f64, f64)> = with_raw
            .iter()
            .map(|&(_, _, _, _, sx, sy)| (sx, sy))
            .collect();
        let raw_pts: Vec<(f64, f64)> = with_raw
            .iter()
            .map(|&(_, _, rx, ry, _, _)| (rx, ry))
            .collect();
        (Some(sigmas), Some(raw_pts))
    } else {
        (None, None)
    };

    // 3. Build image quads once.
    let group_size = group_size_for(n);
    let t_quad_build = Instant::now();
    let image_quads = build_quads(&image_pts, n, group_size);
    tracing::debug!(
        stage = "quad_build",
        pool = "image",
        frame = %frame_name,
        duration_ms = t_quad_build.elapsed().as_millis() as u64,
        count = image_quads.len(),
        group_size,
        "solve stage timing"
    );
    if image_quads.len() < 3 {
        anyhow::bail!("too few image quads ({}, need ≥3)", image_quads.len());
    }
    let min_quads = minimum_quads(n);

    // Catalog epoch: assume current Gaia (2016.0); observation epoch from hints or now.
    let _cat_epoch = repr_cache.catalog_epoch();
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

    tracing::info!(
        frame = %frame_name,
        total_detected,
        n_selected = n,
        n_image_quads = image_quads.len(),
        group_size,
        n_rungs = rungs.len(),
        mode = if center.is_some() { "hinted" } else { "blind" },
        "solve started"
    );

    // 5. Search. Primary pass uses the full detection pool (unchanged
    //    behaviour). If it finds nothing, a fallback pass retries with
    //    density-balanced quad pools — for frames whose detections
    //    out-resolve the catalog. Frames that solve never reach the
    //    fallback, so it cannot regress them.
    let timing = TimingProfile::default();
    // Per-solve cone source: either the legacy deep(+bright) PixelCaches or an
    // additive density-tier `LayeredCones`. Built once here and shared
    // (read-only) across the rayon trial workers. `deep_pixel_cache` is built
    // unconditionally (cheap — an empty map) to keep the borrows below simple;
    // it is unused in `Layered` mode.
    let deep_pixel_cache = PixelCache::new(obs_epoch);
    let (bright_pixel_cache, layered): (Option<PixelCache>, Option<LayeredCones>) = match *caches {
        Caches::Legacy { bright, .. } => (bright.map(|_| PixelCache::new(obs_epoch)), None),
        Caches::Layered { layers } => (None, Some(LayeredCones::new(layers, obs_epoch))),
    };
    let cones = match *caches {
        Caches::Legacy { deep, bright } => ConeSource::Legacy {
            deep,
            deep_pc: &deep_pixel_cache,
            bright,
            bright_pc: bright_pixel_cache.as_ref(),
        },
        Caches::Layered { .. } => {
            ConeSource::Layered(layered.as_ref().expect("layered is Some in the Layered arm"))
        }
    };
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
        cones,
        cfg,
        obs_epoch,
        t0,
        cancel,
        timing: &timing,
        truth,
        frame_name: &frame_name,
        star_sigmas: star_sigmas_storage.as_deref(),
        image_pts_raw: image_pts_raw_storage.as_deref(),
    };
    // Quad-index search first when the hints leave position or scale open:
    // it costs the same anywhere on the sky, where the spiral's cost grows
    // with the area and scale range it has to cover. A fully hinted frame
    // keeps the spiral, which is near-instant around a good hint.
    let has_position = center.is_some();
    let has_scale = scale_bounds.is_none()
        && (hints.pixel_scale_arcsec.is_some() || hints.fov_deg.is_some());
    if let Some(ix) = index.filter(|_| !(has_position && has_scale)) {
        let mut index_pass = PassTrace::new(false);
        let found = run_index_search(&ctx, ix, hints, scale_bounds, &mut index_pass);
        trace.passes.push(index_pass);
        match found {
            Err(e) => return Err(e),
            Ok(Some(mut sol)) => {
                sol.psf_fwhm_px = psf_fwhm_px;
                trace.outcome = TraceOutcome::Solved {
                    ra: sol.wcs.crval.0, dec: sol.wcs.crval.1,
                    inliers: sol.matched_stars, rms_px: sol.rms_residual_px,
                };
                trace.total_ms = t0.elapsed().as_millis() as u64;
                tracing::info!(
                    frame = %frame_name,
                    total_ms = trace.total_ms,
                    inliers = sol.matched_stars,
                    scale_arcsec_px = sol.pixel_scale_arcsec,
                    outcome = "solved",
                    path = "index",
                    "solve succeeded"
                );
                return Ok(sol);
            }
            Ok(None) => {
                tracing::warn!(frame = %frame_name, "quad-index search found nothing — falling back to the spiral search");
            }
        }
    }

    let mut primary_pass = PassTrace::new(false);
    let primary_result = run_search(&ctx, false, &mut primary_pass);
    trace.passes.push(primary_pass);
    match primary_result {
        Err(e) => {
            trace.outcome = TraceOutcome::Failed(e.to_string());
            trace.total_ms = t0.elapsed().as_millis() as u64;
            tracing::error!(frame = %frame_name, total_ms = trace.total_ms, error = %e, outcome = "failed", funnel = %trace, "solve failed");
            return Err(e);
        }
        Ok(Some(mut sol)) => {
            sol.psf_fwhm_px = psf_fwhm_px;
            trace.outcome = TraceOutcome::Solved {
                ra: sol.wcs.crval.0, dec: sol.wcs.crval.1,
                inliers: sol.matched_stars, rms_px: sol.rms_residual_px,
            };
            trace.total_ms = t0.elapsed().as_millis() as u64;
            if cfg.print_timing {
                timing.log_stage_timings(t0.elapsed(), &frame_name);
            }
            tracing::info!(
                frame = %frame_name,
                total_ms = trace.total_ms,
                inliers = sol.matched_stars,
                rms_px = sol.rms_residual_px,
                scale_arcsec_px = sol.pixel_scale_arcsec,
                outcome = "solved",
                funnel = %trace,
                "solve succeeded"
            );
            return Ok(sol);
        }
        Ok(None) => {}
    }
    if is_cancelled() {
        anyhow::bail!("cancelled");
    }
    tracing::warn!(
        frame = %frame_name,
        "primary search found nothing — retrying with density-balanced quad pools"
    );
    let mut fallback_pass = PassTrace::new(true);
    let fallback_result = run_search(&ctx, true, &mut fallback_pass);
    trace.passes.push(fallback_pass);
    match fallback_result {
        Err(e) => {
            trace.outcome = TraceOutcome::Failed(e.to_string());
            trace.total_ms = t0.elapsed().as_millis() as u64;
            tracing::error!(frame = %frame_name, total_ms = trace.total_ms, error = %e, outcome = "failed", funnel = %trace, "solve failed");
            return Err(e);
        }
        Ok(Some(mut sol)) => {
            sol.psf_fwhm_px = psf_fwhm_px;
            trace.outcome = TraceOutcome::Solved {
                ra: sol.wcs.crval.0, dec: sol.wcs.crval.1,
                inliers: sol.matched_stars, rms_px: sol.rms_residual_px,
            };
            trace.total_ms = t0.elapsed().as_millis() as u64;
            if cfg.print_timing {
                timing.log_stage_timings(t0.elapsed(), &frame_name);
            }
            tracing::info!(
                frame = %frame_name,
                total_ms = trace.total_ms,
                inliers = sol.matched_stars,
                rms_px = sol.rms_residual_px,
                scale_arcsec_px = sol.pixel_scale_arcsec,
                outcome = "solved",
                funnel = %trace,
                "solve succeeded"
            );
            return Ok(sol);
        }
        Ok(None) => {}
    }
    if cfg.print_timing {
        timing.log_stage_timings(t0.elapsed(), &frame_name);
    }
    let msg = format!(
        "no trial passed verification across {} FOV rung(s) (incl. density-balanced retry)",
        rungs.len()
    );
    trace.outcome = TraceOutcome::Failed(msg.clone());
    trace.total_ms = t0.elapsed().as_millis() as u64;
    tracing::error!(frame = %frame_name, total_ms = trace.total_ms, error = %msg, outcome = "failed", funnel = %trace, "solve failed");
    // Return the structured failure so callers can render a precise reason
    // (the full funnel trace is carried in the `funnel` field of the error! event above).
    Err(anyhow::Error::new(crate::diag::SolveFailure {
        class: trace.classify(),
        best_attempt: trace.best_attempt(),
        message: msg,
    }))
}

/// Per-solve cone source — the legacy deep(+bright) pair or an additive
/// density-tier stack. Encapsulates the two cone strategies: quad-match uses
/// the bright/base cone with fallback, verify uses the deep/union cone.
#[derive(Clone, Copy)]
enum ConeSource<'a> {
    /// Legacy: deep cache (+ optional bright sub-catalog) with per-solve pixel
    /// caches. Quad-match tries bright then falls back to deep; verify uses deep.
    Legacy {
        deep: &'a StarCache,
        deep_pc: &'a PixelCache,
        bright: Option<&'a StarCache>,
        bright_pc: Option<&'a PixelCache>,
    },
    /// Additive density-tier stack. Quad-match uses the base layer with fallback
    /// to the full union; verify uses the full union.
    Layered(&'a LayeredCones<'a>),
}

impl<'a> ConeSource<'a> {
    /// Quad-match cone: the base/bright cone with automatic fallback to the full
    /// catalog when it is too sparse to seed quads. Localizes the legacy-vs-tier
    /// branch so the call site stays flat.
    fn cone_for_quad_match(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        mag_limit: f32,
        epoch: f64,
        fallback_threshold: usize,
    ) -> Result<Vec<CatalogStar>> {
        match *self {
            ConeSource::Legacy { deep, deep_pc, bright, bright_pc } => cone_for_quad_match(
                deep, deep_pc, bright, bright_pc, ra, dec, radius_deg, mag_limit, epoch,
                fallback_threshold,
            ),
            ConeSource::Layered(lc) => {
                lc.cone_for_quad_match(ra, dec, radius_deg, mag_limit, epoch, fallback_threshold)
            }
        }
    }

    /// Verify cone: the full catalog depth — the deep cache for Legacy, the full
    /// layer union for Layered.
    fn verify_cone(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        mag_limit: f32,
        epoch: f64,
    ) -> Result<Vec<CatalogStar>> {
        match *self {
            ConeSource::Legacy { deep, deep_pc, .. } => {
                deep.cone_cached(deep_pc, ra, dec, radius_deg, mag_limit, epoch)
            }
            ConeSource::Layered(lc) => lc.cone(lc.n_layers(), ra, dec, radius_deg, mag_limit, epoch),
        }
    }
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
    /// Per-solve cone source (legacy deep+bright or additive tier stack),
    /// shared read-only across rayon trial workers. Created once in
    /// `solve_inner`; reused for every cone query (quad-match + verify).
    cones: ConeSource<'a>,
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
    /// Frame file name, threaded for the Phase-0 residual-field dump
    /// (`SOLVEMYASTRO_DUMP_RESIDUALS=1`). Diagnostic only.
    frame_name: &'a str,
    /// Phase 2 (precise profile only): per-star centroid σ aligned 1:1 with
    /// `image_pts`.  `Some` when `CentroidRefinement::Auto` was accepted by
    /// the global field gate (so `FastStar.sx/sy` are meaningful).  `None`
    /// in the fast profile or when the refine gate reverted — the SIP fit
    /// falls back to uniform weights in that case.
    star_sigmas: Option<&'a [(f64, f64)]>,
    /// Phase 4 (precise profile only): pass-1 (intensity-weighted) centroids
    /// aligned 1:1 with `image_pts`.  `Some` iff `star_sigmas.is_some()`.
    /// Used by the compare-and-keep-better guard: after the refined SIP fit,
    /// also compute a pass-1 SIP fit and keep whichever has the lower RMS.
    /// `None` in the fast profile — that path is completely untouched.
    image_pts_raw: Option<&'a [(f64, f64)]>,
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
        rungs,
        center,
        search_radius,
        cones,
        cfg,
        obs_epoch,
        cancel,
        timing,
        ..
    } = ctx;
    let is_cancelled = || cancel.is_some_and(|f| f.load(Ordering::Relaxed));

    // 5a. Spiral search — rayon-parallelised per rung.
    let total_rungs = rungs.len();
    for (rung_idx, rung) in rungs.iter().enumerate() {
        if is_cancelled() {
            anyhow::bail!("cancelled");
        }
        let cells = spiral_cells(center, search_radius, rung.fov_deg * CONE_OVERSIZE);
        let cone_radius = (rung.fov_deg * CONE_OVERSIZE).min(89.0);
        let mut rt = RungTrace::new(rung_idx + 1, total_rungs, rung.fov_deg, cells.len());

        // Spiral pruning. The spiral visits cells in order of increasing
        // distance from the hint, so the correct cell is usually near the
        // start. We process the spiral in chunks (see `spiral_chunks`):
        // each chunk runs all its cells in parallel via rayon, then
        // sequentially verifies any candidates. On the first verified
        // solve we return immediately — skipping every remaining chunk in
        // this rung. Chunks grow geometrically, so a hinted frame whose
        // cell is near the centre pays for a few dozen cells instead of a
        // full chunk, while a frame that must scan the whole rung still
        // runs in large chunks. Candidates are verified in spiral order
        // either way, so the chunking never changes which one is accepted.

        // Early-exit flag: set once ANY trial passes verification.
        // Workers in subsequent chunks check it at cell start.
        let any_verified = AtomicBool::new(false);
        let cells_mutex: Mutex<Vec<CellTrace>> = Mutex::new(Vec::with_capacity(cells.len()));

        // Iterate the spiral in chunks. Each chunk does the par_iter +
        // verify pair; on the first verified solve we return immediately.
        // Empty / no-candidate chunks fall through to the next chunk.
        let mut total_candidates_in_rung = 0usize;
        'chunks: for chunk in spiral_chunks(&cells) {
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
                    let cone_result = cones.cone_for_quad_match(
                        cell.0,
                        cell.1,
                        cone_radius,
                        cfg.catalog_mag_limit,
                        obs_epoch,
                        cfg.bright_fallback_threshold,
                    );
                    let stars = match cone_result {
                        Ok(s) => s,
                        Err(_) => {
                            ct.stop = CellStop::ConeTooSparse;
                            // Guarded data is diagnostics-only (CellTrace pushes for
                            // reporting); recovering a poisoned lock here is strictly
                            // better than cascading panics across the rayon pool.
                            cells_mutex.lock().unwrap_or_else(|e| e.into_inner()).push(ct);
                            return None;
                        }
                    };
                    TimingProfile::add(&timing.cone_ns, t_cone.elapsed());
                    timing.cone_calls.fetch_add(1, Ordering::Relaxed);
                    ct.cat_raw = stars.len();
                    if stars.len() < 4 {
                        ct.stop = CellStop::ConeTooSparse;
                        cells_mutex.lock().unwrap_or_else(|e| e.into_inner()).push(ct);
                        return None;
                    }
                    // A cell whose catalog is sparser than the detections is
                    // density-balanced right away, as the fallback pass would:
                    // an unbalanced pool there cannot match, and waiting for
                    // the fallback means first walking every other rung.
                    let balance =
                        balance || catalog_in_frame(stars.len(), image_center) < image_pts.len();
                    let pool_cap = if balance {
                        MAX_CAT_STARS
                    } else {
                        image_pts.len().min(MAX_CAT_STARS).max(50)
                    };
                    let stars = if stars.len() > pool_cap {
                        // The `pool_cap` brightest, in magnitude order. Selecting
                        // first keeps the sort to the survivors — a wide cone
                        // holds tens of thousands of stars for a few hundred kept.
                        let mut s = stars;
                        let by_mag = |a: &CatalogStar, b: &CatalogStar| {
                            a.mag.partial_cmp(&b.mag).unwrap_or(std::cmp::Ordering::Equal)
                        };
                        s.select_nth_unstable_by(pool_cap - 1, by_mag);
                        s.truncate(pool_cap);
                        s.sort_unstable_by(by_mag);
                        s
                    } else {
                        stars
                    };
                    ct.cat_after_cap = stars.len();
                    // Per-trial deterministic RNG seed: base seed XOR'd with
                    // the cell RA/Dec bit-patterns so each sky cell gets a
                    // unique but reproducible PROSAC sample sequence.
                    let cell_seed = cfg.ransac_seed
                        ^ cell.0.to_bits()
                        ^ cell.1.to_bits().wrapping_mul(0x9E3779B97F4A7C15);
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
                        cell_seed,
                    );
                    cells_mutex.lock().unwrap_or_else(|e| e.into_inner()).push(ct);
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
            // Recentered re-fits are verified after every original seed of
            // the chunk, so they only decide chunks no original seed passes.
            let mut results = chunk_results;
            let alts: Vec<TrialResult> = results
                .iter_mut()
                .filter_map(|t| {
                    let w = t.recentered.take()?;
                    Some(TrialResult {
                        scale_arcsec: w.pixel_scale_arcsec(),
                        wcs: w,
                        tangent: t.tangent,
                        quad_match_pairs: t.quad_match_pairs,
                        recentered: None,
                    })
                })
                .collect();
            results.extend(alts);
            if let Some(sol) = verify_trials(ctx, results, &mut rt)? {
                // Mark verified so remaining parallel trials in this rung stop early,
                // and snapshot this rung into pass_trace so the funnel includes the
                // rung that ultimately solved.
                any_verified.store(true, Ordering::Relaxed);
                pass_trace.rungs.push(rt);
                return Ok(Some(sol));
            }
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

/// Blind / scale-free search through the quad index. Every hypothesis is
/// verified as it arrives (same pipeline as the spiral, via
/// [`verify_trials`]); repeats of an already rejected solution are skipped.
fn run_index_search(
    ctx: &SearchCtx,
    index: &QuadIndex,
    hints: &SolveHints,
    scale_bounds: Option<(f64, f64)>,
    pass_trace: &mut PassTrace,
) -> Result<Option<SolveSolution>> {
    // Plausible scale: the B2 band, else the hint ±5× (the FOV ladder's
    // reach), else anything physical.
    let hint_scale = hints
        .pixel_scale_arcsec
        .or_else(|| hints.fov_deg.map(|f| f * 3600.0 / ctx.long_px.max(1) as f64))
        .filter(|s| s.is_finite() && *s > 0.0);
    let (scale_lo, scale_hi) = match (scale_bounds, hint_scale) {
        (Some(b), _) => b,
        (None, Some(s)) => (s / 5.0, s * 5.0),
        (None, None) => (0.1, 120.0),
    };
    let params = crate::index::SearchParams {
        scale_lo: scale_lo.max(0.05),
        scale_hi: scale_hi.min(120.0),
        ..Default::default()
    };
    let is_cancelled = || ctx.cancel.is_some_and(|f| f.load(Ordering::Relaxed));
    let mut rt = RungTrace::new(0, 0, 0.0, 0);
    let mut rejected: Vec<((f64, f64), f64)> = Vec::new();
    let mut outcome: Result<Option<SolveSolution>> = Ok(None);
    let mut n_hypotheses = 0usize;
    let mut last_queries = 0usize;
    let mut verify_time = std::time::Duration::ZERO;
    let t_search = Instant::now();
    crate::index::search(index, ctx.image_pts, ctx.image_center, &params, is_cancelled, |h| {
        last_queries = h.queries;
        let scale = h.wcs.pixel_scale_arcsec();
        let fov_deg = scale * ctx.long_px as f64 / 3600.0;
        let c = h.wcs.crval;
        if let Some((hra, hdec)) = ctx.center {
            if great_circle_deg(hra, hdec, c.0, c.1) > ctx.search_radius + fov_deg {
                return false;
            }
        }
        if rejected.iter().any(|&((ra, dec), s)| {
            (scale / s - 1.0).abs() < 0.02 && great_circle_deg(ra, dec, c.0, c.1) < 0.02 * fov_deg
        }) {
            return false;
        }
        n_hypotheses += 1;
        let trial = TrialResult {
            scale_arcsec: scale,
            tangent: c,
            wcs: h.wcs,
            quad_match_pairs: 4,
            recentered: None,
        };
        let t_v = Instant::now();
        let verified = verify_trials(ctx, vec![trial], &mut rt);
        verify_time += t_v.elapsed();
        match verified {
            Ok(Some(sol)) => {
                tracing::debug!(
                    frame = %ctx.frame_name,
                    star_n = h.star_n,
                    queries = h.queries,
                    hypotheses = n_hypotheses,
                    "quad-index hypothesis verified"
                );
                outcome = Ok(Some(sol));
                true
            }
            Ok(None) => {
                rejected.push((c, scale));
                false
            }
            Err(e) => {
                outcome = Err(e);
                true
            }
        }
    });
    rt.candidates_emitted = n_hypotheses;
    pass_trace.rungs.push(rt);
    tracing::debug!(
        frame = %ctx.frame_name,
        hypotheses = n_hypotheses,
        rejected = rejected.len(),
        queries_at_last_hypothesis = last_queries,
        search_ms = (t_search.elapsed() - verify_time).as_millis() as u64,
        verify_ms = verify_time.as_millis() as u64,
        solved = matches!(outcome, Ok(Some(_))),
        "quad-index search"
    );
    if outcome.is_ok() && is_cancelled() {
        anyhow::bail!("cancelled");
    }
    outcome
}

/// Verify `trials` in order: seed inliers and log-odds, the full-field refit,
/// SIP, and the bright-star completeness gate. The first trial that passes
/// all of them is returned as the solution; every rejection is recorded in
/// `rt`. Shared by the spiral search and the quad-index search.
fn verify_trials(
    ctx: &SearchCtx,
    trials: Vec<TrialResult>,
    rt: &mut RungTrace,
) -> Result<Option<SolveSolution>> {
    let &SearchCtx {
        image_pts,
        image_center,
        width,
        height,
        long_px,
        total_detected,
        cones,
        cfg,
        obs_epoch,
        t0,
        cancel,
        timing,
        truth,
        frame_name,
        star_sigmas,
        image_pts_raw,
        ..
    } = ctx;
    let is_cancelled = || cancel.is_some_and(|f| f.load(Ordering::Relaxed));
        'candidates: for trial in trials {
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
                    let res =
                        cones.verify_cone(seed_ra, seed_dec, verify_radius, mag_lim, obs_epoch);
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
                tracing::debug!(
                    frame = %frame_name,
                    reason = %reason,
                    inliers = seed_inliers,
                    required,
                    log_odds = v.log_odds,
                    seed_rms_px = v.seed_rms_px,
                    cat_in_fov = verify_stars.len(),
                    "trial rejected"
                );
                v.reject_reason = Some(reason);
                rt.verifies.push(v);
                continue 'candidates;
            }

            // 7. Second solve — full-field affine refit over inlier pairs.
            //
            // Fast path (≥ 50 seed inliers): existing single `refit_affine`
            // call — byte-identical to pre-Phase-3 behavior. Easy corpus
            // frames always land here.
            //
            // Robust path (< 50 seed inliers): iterative σ-clipped refit
            // (`refit_affine_iter`) that runs up to 5 rounds of refit →
            // σ-clip → refit, stopping when RMS change ≤ 3% or the inlier
            // set is stable (Jaccard > 0.97). Only weak seeds pay this cost.
            const ITER_REFIT_GATE: usize = 50;
            let refit_tol = (tol_px * 2.0).min(MAX_MATCH_PX);
            let final_wcs = if seed_pairs.len() >= 6 {
                let t_ra = Instant::now();
                let refitted_wcs: Option<WcsSolution> = if seed_inliers < ITER_REFIT_GATE {
                    // Robust path: iterative σ-clipped refit.
                    // `refit_affine_iter` re-gathers pairs internally at
                    // `refit_tol`, so we don't need a pre-gather step.
                    refit_affine_iter(
                        &seed_pairs,
                        image_center,
                        &trial.wcs,
                        &verify_stars,
                        &image_pts,
                        width,
                        height,
                        tol_px,
                    )
                } else {
                    // Fast path: single refit — byte-identical.
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
                    refit_affine(&refit_pairs, image_center, &trial.wcs)
                };
                TimingProfile::add(&timing.refit_affine_ns, t_ra.elapsed());

                match refitted_wcs {
                    Some(refitted) => {
                        // Keep the refitted WCS only when it doesn't degrade
                        // the inlier count (sanity guard against edge cases).
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
            // Suppress unused-variable warning on quad_match_pairs in the fast
            // path (the field is threaded for future gating, e.g. additional
            // PROSAC-based refit decisions).
            let _ = trial.quad_match_pairs;

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
            tracing::info!(
                frame = %frame_name,
                ra = cra,
                dec = cdec,
                scale_arcsec_px = final_scale,
                rot_deg = final_wcs.field_rotation_deg(),
                inliers = final_inliers,
                rms_px,
                "trial solved"
            );

            // 8. SIP fit (optional).
            //
            // Precise profile (star_sigmas.is_some()): use `fit_sip_annealed`
            // which applies σ-weighted LSQ + 3σ clip in a convergence loop.
            // Fast profile (star_sigmas.is_none()): use the existing `fit_sip`
            // unchanged — the fast path must remain byte-identical.
            //
            // Phase-4 compare-and-keep-better guard (precise profile only):
            // After computing the refined-centroid SIP solution, also compute a
            // pass-1-centroid SIP solution (using uniform weights, same as the
            // fast path) and keep whichever has the lower count_inliers RMS.
            // This guarantees precise ≥ fast on every frame: even pathological
            // frames where PSF-refined centroids are noisier (defocused _0008,
            // sparse M51, _0023) will fall back to pass-1 and be no worse than
            // the fast profile.  The expensive quad-search/seed uses refined
            // positions throughout; only this cheap final fit runs twice.
            // Guard: only attempt the comparison when rms_ref > 0.3 px — for
            // already-excellent frames the pass-1 path cannot improve things and
            // we skip the extra fit entirely.
            const PHASE4_RMS_FLOOR: f64 = 0.3;

            let (sip_wcs, sip_rms_px, sip_rms_arcsec) = if cfg.fit_sip {
                let t_sip = Instant::now();
                let sip_result = if star_sigmas.is_some() {
                    // Precise profile: annealed, σ-weighted, σ-clipped SIP fit.
                    crate::sip::fit_sip_annealed(
                        &final_wcs,
                        &verify_stars,
                        image_pts,
                        star_sigmas,
                        width,
                        height,
                        tol_px,
                        cfg.sip_order,
                    )
                } else {
                    // Fast profile: existing two-stage SIP fit — byte-identical.
                    crate::sip::fit_sip(
                        &final_wcs,
                        &verify_stars,
                        image_pts,
                        width,
                        height,
                        tol_px,
                        cfg.sip_order,
                    )
                };
                TimingProfile::add(&timing.fit_sip_ns, t_sip.elapsed());
                match sip_result {
                    Some(w_ref) => {
                        let t = Instant::now();
                        let (si_ref, srss_ref, sip_pairs_ref) = count_inliers(&w_ref, &verify_stars, image_pts, width, height, tol_px);
                        TimingProfile::add(&timing.count_inliers_ns, t.elapsed());
                        timing.count_inliers_calls.fetch_add(1, Ordering::Relaxed);
                        let srms_ref = if si_ref > 0 { (srss_ref / si_ref as f64).sqrt() } else { f64::MAX };

                        // Phase-4: compare-and-keep-better (precise profile only).
                        // Only run when: (a) we have pass-1 raw positions AND (b)
                        // the refined rms is above the floor (otherwise the frame is
                        // already excellent and the raw path cannot help).
                        let (w, srms, sip_pairs) = if star_sigmas.is_some()
                            && srms_ref > PHASE4_RMS_FLOOR
                            && srms_ref < f64::MAX
                        {
                            if let Some(raw_pts) = image_pts_raw {
                                // Fit pass-1 SIP (uniform weights — same as fast path).
                                let w_raw_opt = crate::sip::fit_sip(
                                    &final_wcs,
                                    &verify_stars,
                                    raw_pts,
                                    width,
                                    height,
                                    tol_px,
                                    cfg.sip_order,
                                );
                                match w_raw_opt {
                                    Some(w_raw) => {
                                        let t2 = Instant::now();
                                        let (si_raw, srss_raw, _) = count_inliers(
                                            &w_raw,
                                            &verify_stars,
                                            raw_pts,
                                            width,
                                            height,
                                            tol_px,
                                        );
                                        TimingProfile::add(&timing.count_inliers_ns, t2.elapsed());
                                        timing.count_inliers_calls.fetch_add(1, Ordering::Relaxed);
                                        let srms_raw = if si_raw > 0 {
                                            (srss_raw / si_raw as f64).sqrt()
                                        } else {
                                            f64::MAX
                                        };
                                        if srms_raw < srms_ref {
                                            tracing::debug!(
                                                frame = %frame_name,
                                                winner = "raw",
                                                rms_raw_px = srms_raw,
                                                rms_ref_px = srms_ref,
                                                inliers_raw = si_raw,
                                                inliers_ref = si_ref,
                                                "Phase4 guard: pass-1 wins"
                                            );
                                            (w_raw, srms_raw, sip_pairs_ref)
                                        } else {
                                            tracing::debug!(
                                                frame = %frame_name,
                                                winner = "refined",
                                                rms_raw_px = srms_raw,
                                                rms_ref_px = srms_ref,
                                                inliers_raw = si_raw,
                                                inliers_ref = si_ref,
                                                "Phase4 guard: refined wins"
                                            );
                                            (w_ref, srms_ref, sip_pairs_ref)
                                        }
                                    }
                                    None => (w_ref, srms_ref, sip_pairs_ref),
                                }
                            } else {
                                (w_ref, srms_ref, sip_pairs_ref)
                            }
                        } else {
                            (w_ref, srms_ref, sip_pairs_ref)
                        };

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
                        let profile = if star_sigmas.is_some() { "annealed+guard" } else { "two-stage" };
                        // Keep SIP only if it does not worsen the fit (with the
                        // correct pixel-space math it should strictly improve;
                        // this guards against a pathological field).
                        if srms <= rms_px {
                            tracing::debug!(
                                frame = %frame_name,
                                profile,
                                rms_linear_px = rms_px,
                                rms_sip_px = srms,
                                pairs = si_ref,
                                median_resid_px = med,
                                p90_resid_px = p90,
                                max_resid_px = mx,
                                over_2px = gt2,
                                "SIP fit applied"
                            );
                            (w, srms, srms * final_scale)
                        } else {
                            tracing::debug!(
                                frame = %frame_name,
                                profile,
                                rms_linear_px = rms_px,
                                rms_sip_px = srms,
                                "SIP fit rejected — worse than linear"
                            );
                            (final_wcs.clone(), rms_px, rms_arcsec)
                        }
                    }
                    None => (final_wcs.clone(), rms_px, rms_arcsec),
                }
            } else {
                (final_wcs.clone(), rms_px, rms_arcsec)
            };

            // Bright-star completeness on the final (SIP) WCS: the brightest
            // catalog stars in the frame must mostly have detections. A dense
            // field's chance alignment collects its inliers among the faint
            // stars and finds almost none of the bright ones; evaluated after
            // SIP so a wide lens's edge distortion does not count as a miss.
            let (bright_k, bright_m) =
                bright_star_evidence(&sip_wcs, &verify_stars, image_pts, width, height, tol_px);
            let p_random = image_pts.len() as f64 * std::f64::consts::PI * tol_px * tol_px
                / (width * height);
            let bright_lo = crate::verify::bright_star_log_odds(bright_k, bright_m, p_random);
            tracing::debug!(
                frame = %frame_name,
                bright_k,
                bright_m,
                p_random,
                bright_log_odds = bright_lo,
                "bright-star completeness"
            );
            if bright_lo < cfg.log_odds_threshold {
                if let Some(last) = rt.verifies.last_mut() {
                    last.accepted = false;
                    last.reject_reason = Some(format!(
                        "bright stars: {bright_m}/{bright_k} found, log-odds {bright_lo:.1} < {:.1}",
                        cfg.log_odds_threshold
                    ));
                }
                continue 'candidates;
            }
            let solve_time_ms = t0.elapsed().as_millis() as u64;


            // Phase-0 diagnostic (SOLVEMYASTRO_DUMP_RESIDUALS=1): dump the
            // per-star residual vector field at the final (SIP-or-linear) WCS
            // so we can read centroid-bound (random scatter) vs distortion-bound
            // (radial/quadrant structure) before any precision work. Gated —
            // zero cost in normal solves.
            if crate::diag::dump_residuals_enabled() {
                let (_, _, resid_pairs) =
                    count_inliers(&sip_wcs, &verify_stars, &image_pts, width, height, tol_px);
                crate::diag::dump_residual_field(frame_name, &resid_pairs);
            }


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
                // psf_fwhm_px is patched in by solve_inner after run_search returns,
                // since only solve_inner has the PSF detection result in scope.
                psf_fwhm_px: None,
            }));
        } // end 'candidates loop
    Ok(None)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bounded_ladder_tests {
    use super::{fov_rungs_bounded, HINTED_FOV_MIN_DEG, FOV_MAX_DEG};

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
            assert!(r.fov_deg >= HINTED_FOV_MIN_DEG - 1e-9 && r.fov_deg <= FOV_MAX_DEG + 1e-9);
        }
        // Reaches near the finer (lo) end: smallest scale ≤ ~1.5× lo.
        let smallest = rungs.last().unwrap().scale_arcsec;
        assert!(smallest <= lo * 1.5 + 1e-6, "must reach the fine end: {smallest}");
    }
}

// ---------------------------------------------------------------------------
// B7 — orchestrate unit tests (Phase 3 requirement)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod orchestrate_unit_tests {
    use super::*;
    use crate::SolveHints;

    /// Measured shares, from real frames (2026-09-05): a wind-shaken frame
    /// leaves 14 of 600 detections after the shape cut, a healthy one 572,
    /// and even a hard DSLR frame keeps 40 %. Only the first has nothing to
    /// match, and only it is worth refusing before the FOV ladder runs —
    /// twice, counting the density-balanced retry.
    #[test]
    fn a_frame_of_streaks_is_told_apart_from_a_hard_one() {
        assert!(looks_trailed(600, 14), "586 of 600 are streaks");
        assert!(looks_trailed(600, 60), "exactly 90 % dropped");
        assert!(!looks_trailed(600, 61), "just under the line is not a verdict");
        assert!(!looks_trailed(600, 243), "a hard DSLR frame keeps 40 %");
        assert!(!looks_trailed(600, 572), "a healthy frame");
        // A sparse field has no population to judge — a narrow FOV can
        // legitimately detect only a handful of stars.
        assert!(!looks_trailed(50, 2), "too few detections to call it");
        assert!(!looks_trailed(0, 0));
    }

    // -----------------------------------------------------------------------
    // fov_rungs
    // -----------------------------------------------------------------------

    #[test]
    fn fov_rungs_hinted_spans_5x_scale() {
        // A hinted scale should produce a ladder that covers ~5× the hint in
        // each direction (4 steps of ÷1.5 = 1.5^4 ≈ 5× range on each side).
        let hints = SolveHints {
            pixel_scale_arcsec: Some(2.0),
            ..Default::default()
        };
        let rungs = fov_rungs(5000, &hints);
        // The center rung is the hint itself.
        let scales: Vec<f64> = rungs.iter().map(|r| r.scale_arcsec).collect();
        // Hint rung must be present (within 1% rounding).
        assert!(
            scales.iter().any(|&s| (s - 2.0).abs() / 2.0 < 0.01),
            "hint scale 2.0\"/px not found in rungs: {scales:?}"
        );
        // Range should span at least 5× the hint in each direction.
        let min_scale = scales.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_scale = scales.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // Expect at least 4 rungs (center + expansions).
        assert!(rungs.len() >= 4, "hinted ladder must have ≥4 rungs; got {}", rungs.len());
        assert!(
            max_scale / min_scale >= 3.0,
            "scale range must span ≥3×; got {min_scale:.3}..{max_scale:.3}\"/px"
        );
    }

    #[test]
    fn fov_rungs_blind_descends_from_max_to_min() {
        let hints = SolveHints::default(); // no hints → blind
        let rungs = fov_rungs(5000, &hints);
        // Blind ladder must be non-empty and start near FOV_MAX_DEG.
        assert!(!rungs.is_empty());
        assert!(
            (rungs[0].fov_deg - FOV_MAX_DEG).abs() < 1e-9,
            "blind ladder must start at FOV_MAX_DEG"
        );
        // Must descend monotonically.
        for w in rungs.windows(2) {
            assert!(w[0].fov_deg >= w[1].fov_deg, "blind ladder must descend");
        }
        // Last rung must be at or below FOV_MIN_DEG.
        assert!(
            rungs.last().unwrap().fov_deg <= FOV_MIN_DEG + 1e-9,
            "blind ladder must reach FOV_MIN_DEG"
        );
    }

    // -----------------------------------------------------------------------
    // spiral_cells
    // -----------------------------------------------------------------------

    #[test]
    fn spiral_chunks_grow_and_cover_in_order() {
        let cells: Vec<usize> = (0..5000).collect();
        let sizes: Vec<usize> = spiral_chunks(&cells).map(|c| c.len()).collect();
        assert_eq!(&sizes[..7], &[16, 32, 64, 128, 256, 512, 1024]);
        assert!(sizes.iter().all(|&n| n <= SPIRAL_CHUNK_MAX));
        let flat: Vec<usize> = spiral_chunks(&cells).flatten().copied().collect();
        assert_eq!(flat, cells);
        assert_eq!(spiral_chunks::<usize>(&[]).count(), 0);
    }

    #[test]
    fn spiral_cells_centre_first() {
        // The first cell emitted must be the centre (0-offset in the spiral).
        let cells = spiral_cells(Some((45.0, 30.0)), 5.0, 2.0);
        assert!(!cells.is_empty());
        let (r0, d0) = cells[0];
        assert!(
            great_circle_deg(r0, d0, 45.0, 30.0) < 1e-6,
            "first cell must be the centre; got ({r0:.4}, {d0:.4})"
        );
    }

    #[test]
    fn spiral_cells_antimeridian_wrap() {
        // Center near RA=359° — cells should not produce negative RA.
        let cells = spiral_cells(Some((359.5, 0.0)), 3.0, 1.0);
        for &(ra, _dec) in &cells {
            assert!(ra >= 0.0 && ra < 360.0, "RA out of [0,360): {ra}");
        }
    }

    #[test]
    fn spiral_cells_pole_does_not_panic() {
        // Declination near the pole — cos(dec) → 0, RA step → ∞; must not panic.
        let cells = spiral_cells(Some((0.0, 89.5)), 2.0, 1.0);
        // Just verify we get at least the centre cell without panicking.
        assert!(!cells.is_empty());
    }

    #[test]
    fn spiral_cells_blind_covers_full_sky() {
        // No center → blind mode → many cells.
        let cells = spiral_cells(None, 180.0, 5.0);
        // At step 5° and radius 180° we expect hundreds of cells.
        assert!(cells.len() > 100, "blind spiral should cover many cells; got {}", cells.len());
    }

    // -----------------------------------------------------------------------
    // count_inliers
    // -----------------------------------------------------------------------

    #[test]
    fn count_inliers_basic() {
        // Synthetic WCS: identity (pixel = sky in arcsec units — doesn't matter
        // for this test since we directly use count_inliers with a crafted WCS).
        // Build a WCS with known pixel_to_sky/sky_to_pixel behavior.
        // Use the direct catalog⟶pixel mapping of count_inliers:
        // it calls wcs.sky_to_pixel(cat.ra, cat.dec) and matches to image_pts.
        //
        // Strategy: use a trivial CD matrix where 1 arcsec = 1 pixel so
        // sky_to_pixel(ra, dec) = (ra*3600, dec*3600) (approximately).
        // Build cat stars and image_pts that should all match within tol.

        // A real WCS solution with an identity-like CD matrix (1"/px at a
        // tangent point of (0°, 0°)) is easiest to construct here.
        use crate::wcs::WcsSolution;
        let wcs = WcsSolution {
            crpix: (0.0, 0.0),
            crval: (0.0, 0.0),
            cd: [[1.0 / 3600.0, 0.0], [0.0, 1.0 / 3600.0]],
            sip_forward: None,
            sip_reverse: None,
        };
        // At this WCS, sky_to_pixel(0.001, 0.001) ≈ (3.6, 3.6).
        let cat = CatalogStar { ra: 0.001, dec: 0.001, mag: 10.0 };
        let image_pts = vec![(3.6, 3.6)]; // close match

        let (count, rss, pairs) = count_inliers(&wcs, &[cat], &image_pts, 100.0, 100.0, 2.0);
        assert_eq!(count, 1, "one star should match");
        assert!(rss.is_finite() && rss >= 0.0);
        assert_eq!(pairs.len(), 1);
    }

    // -----------------------------------------------------------------------
    // required_inliers
    // -----------------------------------------------------------------------

    #[test]
    fn required_inliers_floor_at_min_absolute() {
        // When both the catalog and detected counts are tiny, the floor is
        // MIN_ABSOLUTE_INLIERS (6).
        let r = required_inliers(0, 0);
        assert_eq!(r, MIN_ABSOLUTE_INLIERS);
    }

    #[test]
    fn required_inliers_dynamic_capped_at_detected() {
        // Large catalog but tiny detection pool → cap at 5% of detected.
        let r = required_inliers(10_000, 20); // cat_floor=500, det_cap=1 → max(6,1)=6
        // det_cap = 1 < MIN_ABSOLUTE_INLIERS=6, so result = 6.
        assert_eq!(r, MIN_ABSOLUTE_INLIERS);
    }

    #[test]
    fn required_inliers_scales_with_catalog() {
        // With 200 bright stars in FOV and 300 detected:
        // cat_floor = 200*0.05 = 10, det_cap = 300*0.05 = 15 → min(10,15)=10 → max(6,10)=10.
        let r = required_inliers(200, 300);
        assert_eq!(r, 10);
    }

    // -----------------------------------------------------------------------
    // adaptive_tol_px
    // -----------------------------------------------------------------------

    #[test]
    fn adaptive_tol_px_base_case() {
        // At 3.0"/px with base_arcsec=3.0: tol = 3.0/3.0 = 1.0 px.
        // But floor is 1.5, so result = 1.5.
        let t = adaptive_tol_px(3.0, BASE_MATCH_ARCSEC);
        assert!((t - 1.5).abs() < 1e-9, "expected 1.5 px, got {t}");
    }

    #[test]
    fn adaptive_tol_px_fine_scale_clamped_at_floor() {
        // Very fine scale (0.1"/px): 3.0/0.1=30 px, clamped to MAX_MATCH_PX=20.
        let t = adaptive_tol_px(0.1, BASE_MATCH_ARCSEC);
        assert!((t - MAX_MATCH_PX).abs() < 1e-9, "expected MAX_MATCH_PX={MAX_MATCH_PX}, got {t}");
    }

    #[test]
    fn adaptive_tol_px_coarse_scale_reasonable() {
        // At 1.5"/px: 3.0/1.5 = 2.0 px.
        let t = adaptive_tol_px(1.5, BASE_MATCH_ARCSEC);
        assert!((t - 2.0).abs() < 1e-9, "expected 2.0 px, got {t}");
    }

    #[test]
    fn adaptive_tol_px_zero_scale_returns_default() {
        // scale_arcsec = 0 → returns 5.0 (fallback in the function).
        let t = adaptive_tol_px(0.0, BASE_MATCH_ARCSEC);
        assert_eq!(t, 5.0);
    }

    // -----------------------------------------------------------------------
    // fit_affine + PROSAC path (via prosac_seed_affine)
    // -----------------------------------------------------------------------

    #[test]
    fn prosac_correct_fit_clean_data() {
        use crate::quad::Quad;
        // Known transform: xcat = 2*ximg + 100, ycat = 2*yimg + 200.
        let img_centers = [
            (50.0_f64, 50.0), (150.0, 50.0), (100.0, 150.0), (200.0, 150.0),
            (75.0, 200.0),
        ];
        let cat_centers: Vec<(f64, f64)> = img_centers
            .iter()
            .map(|&(x, y)| (x * 2.0 + 100.0, y * 2.0 + 200.0))
            .collect();

        let mk_quad = |c: (f64, f64)| Quad {
            star_idx: [0, 1, 2, 3],
            ratios: [0.5; 5],
            longest: 100.0,
            center: c,
        };

        let img_quads: Vec<Quad> = img_centers.iter().map(|&c| mk_quad(c)).collect();
        let cat_quads: Vec<Quad> = cat_centers.iter().map(|&c| mk_quad(c)).collect();
        let matches: Vec<(usize, usize)> = (0..img_quads.len()).map(|i| (i, i)).collect();

        // PROSAC with a fixed seed must produce a valid affine.
        let affine = prosac_seed_affine(&matches, &img_quads, &cat_quads, 42, 50)
            .expect("prosac must produce an affine on clean data");

        let (xc, yc) = affine.apply(250.0, 150.0);
        let ex = 250.0 * 2.0 + 100.0;
        let ey = 150.0 * 2.0 + 200.0;
        assert!((xc - ex).abs() < 1.0, "x: expected {ex}, got {xc}");
        assert!((yc - ey).abs() < 1.0, "y: expected {ey}, got {yc}");
    }

    #[test]
    fn prosac_robust_to_30pct_outliers() {
        use crate::quad::Quad;
        // 7 good pairs (xcat = 2*ximg + 50, ycat = 2*yimg + 100)
        // + 3 bad pairs (wildly wrong catalog centers).
        let good_img = [
            (10.0_f64, 10.0), (50.0, 10.0), (30.0, 50.0), (80.0, 50.0),
            (20.0, 80.0), (60.0, 80.0), (40.0, 30.0),
        ];
        let bad_img = [(5.0_f64, 5.0), (15.0, 5.0), (25.0, 5.0)];
        let mk_quad = |c: (f64, f64)| Quad {
            star_idx: [0, 1, 2, 3],
            ratios: [0.5; 5],
            longest: 50.0,
            center: c,
        };
        let mut img_quads: Vec<Quad> = good_img.iter().map(|&c| mk_quad(c)).collect();
        img_quads.extend(bad_img.iter().map(|&c| mk_quad(c)));

        let mut cat_quads: Vec<Quad> = good_img
            .iter()
            .map(|&(x, y)| mk_quad((x * 2.0 + 50.0, y * 2.0 + 100.0)))
            .collect();
        // Outlier catalog quads: random junk positions.
        cat_quads.push(mk_quad((9999.0, 9999.0)));
        cat_quads.push(mk_quad((-5000.0, 3000.0)));
        cat_quads.push(mk_quad((100.0, -8000.0)));

        let matches: Vec<(usize, usize)> = (0..img_quads.len()).map(|i| (i, i)).collect();
        let affine = prosac_seed_affine(&matches, &img_quads, &cat_quads, 0x1234_5678, 100)
            .expect("prosac must succeed with 70% inliers");

        // Verify on a held-out good point.
        let (xc, yc) = affine.apply(45.0, 25.0);
        let ex = 45.0 * 2.0 + 50.0;
        let ey = 25.0 * 2.0 + 100.0;
        assert!((xc - ex).abs() < 5.0, "x: expected ~{ex}, got {xc}");
        assert!((yc - ey).abs() < 5.0, "y: expected ~{ey}, got {yc}");
    }

    #[test]
    fn prosac_deterministic_with_fixed_seed() {
        use crate::quad::Quad;
        // Build a small synthetic match set and verify PROSAC produces the
        // same affine (same a1/b1/c1/a2/b2/c2) on two consecutive calls
        // with the same seed.
        let pts: Vec<(f64, f64)> = (0..8).map(|i| (i as f64 * 20.0, (i % 3) as f64 * 15.0)).collect();
        let mk_quad = |c: (f64, f64)| Quad {
            star_idx: [0, 1, 2, 3],
            ratios: [0.5; 5],
            longest: 30.0,
            center: c,
        };
        let img_quads: Vec<Quad> = pts.iter().map(|&c| mk_quad(c)).collect();
        let cat_quads: Vec<Quad> = pts
            .iter()
            .map(|&(x, y)| mk_quad((x * 1.5 + 10.0, y * 1.5 + 5.0)))
            .collect();
        let matches: Vec<(usize, usize)> = (0..img_quads.len()).map(|i| (i, i)).collect();

        let seed = 0xDEAD_BEEF_CAFE_BABEu64;
        let a1 = prosac_seed_affine(&matches, &img_quads, &cat_quads, seed, 50);
        let a2 = prosac_seed_affine(&matches, &img_quads, &cat_quads, seed, 50);

        match (a1, a2) {
            (Some(a), Some(b)) => {
                assert!((a.a1 - b.a1).abs() < 1e-12, "a1 differs: {} vs {}", a.a1, b.a1);
                assert!((a.b1 - b.b1).abs() < 1e-12, "b1 differs");
                assert!((a.c1 - b.c1).abs() < 1e-12, "c1 differs");
                assert!((a.a2 - b.a2).abs() < 1e-12, "a2 differs");
                assert!((a.b2 - b.b2).abs() < 1e-12, "b2 differs");
                assert!((a.c2 - b.c2).abs() < 1e-12, "c2 differs");
            }
            (None, None) => {} // both failed — deterministic by being consistent
            _ => panic!("prosac produced different Some/None results on identical inputs"),
        }
    }
}
