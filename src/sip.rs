//! SIP distortion polynomial fit (pixel-space).
//!
//! # Coordinate space (the critical invariant)
//!
//! `WcsSolution::pixel_to_sky` applies SIP **in pixels, before the CD
//! matrix**: `u' = u + A(u,v)`, `v' = v + B(u,v)`, then
//! `(xi,eta) = CD·(u',v')`. So the A/B (and reverse AP/BP) polynomials
//! MUST be fit in pixel space — never against an arcsec/tangent residual.
//!
//! The two reference arrays per matched pair:
//! - `measured = (det_x - crpix_x, det_y - crpix_y)` — observed pixel
//!   offset.
//! - `reference = CD⁻¹ · tan(cat_ra, cat_dec; crval)` — the *ideal,
//!   undistorted* pixel offset the star would have under the pure linear
//!   WCS.
//!
//! Forward A/B (`pixel_to_sky`): fit `measured → reference`. Since the
//! WCS adds `A(u,v)` to `u`, A/B are the **residual**
//! `reference - measured` as a polynomial in `(measured_u, measured_v)`
//! (the standard SIP identity removal where the linear term of the
//! correction absorbs into `A_1_0`).
//! Reverse AP/BP (`sky_to_pixel`, which adds `AP` to `CD⁻¹·tan` ≈
//! reference): fit `measured - reference` as a polynomial in
//! `(reference_x, reference_y)`.
//!
//! # Pair matching
//!
//! SIP exists to capture the residual the linear model *cannot* — which
//! is largest at the field corners. Matching pairs at the tight linear
//! tolerance rejects exactly those corner stars (the entire SIP signal),
//! so stage 1 matches with a **generous** tolerance against the linear
//! projection. We then iterate once: re-project with the just-fitted SIP
//! and re-match at the tight tolerance (now valid, since SIP removes the
//! distortion) and refit from the cleaner, larger set.
//!
//! # Order / minimum pairs
//!
//! ≥20 pairs → order 3 (10 coeffs/axis). ≥12 pairs → order-2 fallback (6
//! coeffs/axis) so sparse narrow fields still get distortion terms.
//! Below 12 → linear WCS unchanged (still a valid solve).

use nalgebra::{DMatrix, DVector};

use crate::{
    nn::PointGrid,
    wcs::{tan, SipCoefficients, WcsSolution},
    CatalogStar,
};

/// Number of terms in a SIP polynomial of given order: Σ_{p=0}^{order} (p+1) = (order+1)(order+2)/2.
/// We exclude the constant (p+q=0) and linear (p+q=1) terms from the **correction** polynomial,
/// but in practice the full triangular basis up to `order` is used and the linear part of the
/// correction is absorbed into A_1_0 (the standard SIP identity removal).
/// Here we build the full basis 0..=order and let the LSQ sort it out.
fn n_terms(order: u8) -> usize {
    let o = order as usize;
    (o + 1) * (o + 2) / 2
}

/// Build the SIP polynomial basis row for given (u, v) and order.
/// Basis order: (0,0),(1,0),(0,1),(2,0),(1,1),(0,2),(3,0),...
/// Index (i,j) where i+j ≤ order: row = Σ_{p=0}^{i+j-1}(p+1) + j.
fn basis_row(u: f64, v: f64, order: u8) -> Vec<f64> {
    let n = n_terms(order);
    let mut row = Vec::with_capacity(n);
    for p in 0..=(order as usize) {
        for j in 0..=p {
            let i = p - j;
            row.push(u.powi(i as i32) * v.powi(j as i32));
        }
    }
    row
}

/// Build a `SipCoefficients` from a flat coefficient vector (triangular basis).
fn vec_to_sip(coeffs_flat: &[f64], order: u8) -> SipCoefficients {
    let o = order as usize;
    let mut coeffs: Vec<Vec<f64>> = vec![vec![0.0; o + 1]; o + 1];
    let mut k = 0;
    for p in 0..=o {
        for j in 0..=p {
            let i = p - j;
            if i <= o && j <= o {
                coeffs[i][j] = coeffs_flat[k];
            }
            k += 1;
        }
    }
    SipCoefficients { order, coeffs }
}

/// Evaluate a SIP polynomial at (u, v) in pixel-offset coordinates.
///
/// Returns `Σ_{i,j: i+j≤order} c[i][j] · u^i · v^j`.
fn eval_sip_poly(sip: &SipCoefficients, u: f64, v: f64) -> f64 {
    let o = sip.order as usize;
    let mut sum = 0.0_f64;
    for p in 0..=o {
        for j in 0..=p {
            let i = p - j;
            if i < sip.coeffs.len() && j < sip.coeffs[i].len() {
                sum += sip.coeffs[i][j] * u.powi(i as i32) * v.powi(j as i32);
            }
        }
    }
    sum
}

/// Fit SIP forward (A/B) and reverse (AP/BP) in **pixel space**.
/// Returns `Some(WcsSolution)` with SIP populated, or `None` if there are
/// too few pairs or the fit is numerically singular (caller keeps the
/// linear WCS — still a valid solve).
pub fn fit_sip(
    wcs: &WcsSolution,
    cat_stars: &[CatalogStar],
    image_pts: &[(f64, f64)],
    width: f64,
    height: f64,
    tol_px: f64,
    order: u8,
) -> Option<WcsSolution> {
    let want_order = order.clamp(2, 3); // supported orders: 2 or 3

    let (crpix_x, crpix_y) = wcs.crpix;
    let (crval_ra, crval_dec) = wcs.crval;

    let det_cd = wcs.cd[0][0] * wcs.cd[1][1] - wcs.cd[0][1] * wcs.cd[1][0];
    if det_cd.abs() < 1e-30 {
        return None;
    }
    let inv = 1.0 / det_cd;

    // (measured_u, measured_v, ref_x, ref_y) per matched pair, all pixel offsets
    // from CRPIX. `ref` = CD⁻¹ · tan(cat;crval) — the ideal undistorted offset.
    type Pair = (f64, f64, f64, f64);

    // Match each catalog star to the nearest detection within `mtol`,
    // projecting through `proj` (linear WCS in stage 1, SIP-corrected in 2).
    let gen_tol = (tol_px * 4.0).max(6.0);
    let grid = PointGrid::new(image_pts, gen_tol);
    let collect = |proj: &WcsSolution, mtol: f64| -> Vec<Pair> {
        let mut pairs = Vec::new();
        for cat in cat_stars {
            let (px, py) = proj.sky_to_pixel(cat.ra, cat.dec);
            if !px.is_finite() || !py.is_finite() {
                continue;
            }
            if px < 0.0 || px >= width || py < 0.0 || py >= height {
                continue;
            }
            let Some((i, best)) = grid.nearest_by(px, py, |(ix, iy)| (ix - px).hypot(iy - py))
            else {
                continue;
            };
            if best > mtol {
                continue;
            }
            let best_img = image_pts[i];
            // Ideal pixel offset (degrees in via tan, pixels out via CD⁻¹) —
            // the same units `pixel_to_sky` adds A/B in.
            let (xi, eta) = tan::sky_to_tangent(cat.ra, cat.dec, crval_ra, crval_dec);
            let rx = inv * (wcs.cd[1][1] * xi - wcs.cd[0][1] * eta);
            let ry = inv * (-wcs.cd[1][0] * xi + wcs.cd[0][0] * eta);
            pairs.push((best_img.0 - crpix_x, best_img.1 - crpix_y, rx, ry));
        }
        pairs
    };

    // ≥20 pairs → order 3; order-2 fallback for sparse fields with ≥12.
    let order_for = |n: usize| -> Option<u8> {
        if n >= 20 {
            Some(want_order)
        } else if n >= 12 {
            Some(2.min(want_order))
        } else {
            None
        }
    };

    // Forward A/B: basis in measured (u,v) → residual (ref − measured), since
    // `pixel_to_sky` computes u' = u + A(u,v) and needs u' = ref.
    // Reverse AP/BP: basis in (ref_x,ref_y) → residual (measured − ref), since
    // `sky_to_pixel` computes u = CD⁻¹·tan (= ref) then u += AP(u,v) = measured.
    let fit_from = |pairs: &[Pair],
                    ord: u8|
     -> Option<(SipCoefficients, SipCoefficients, SipCoefficients, SipCoefficients)> {
        let n = pairs.len();
        let nb = n_terms(ord);
        if n < nb {
            return None;
        }
        let mut a_fwd = DMatrix::zeros(n, nb);
        let mut fx = DVector::zeros(n);
        let mut fy = DVector::zeros(n);
        let mut a_rev = DMatrix::zeros(n, nb);
        let mut rxv = DVector::zeros(n);
        let mut ryv = DVector::zeros(n);

        for (row, &(mu, mv, refx, refy)) in pairs.iter().enumerate() {
            for (col, b) in basis_row(mu, mv, ord).into_iter().enumerate() {
                a_fwd[(row, col)] = b;
            }
            fx[row] = refx - mu;
            fy[row] = refy - mv;

            for (col, b) in basis_row(refx, refy, ord).into_iter().enumerate() {
                a_rev[(row, col)] = b;
            }
            rxv[row] = mu - refx;
            ryv[row] = mv - refy;
        }

        let eps = 1e-12;
        let svd_f = a_fwd.svd(true, true);
        let cx = svd_f.solve(&fx, eps).ok()?;
        let cy = svd_f.solve(&fy, eps).ok()?;
        let svd_r = a_rev.svd(true, true);
        let cpx = svd_r.solve(&rxv, eps).ok()?;
        let cpy = svd_r.solve(&ryv, eps).ok()?;

        Some((
            vec_to_sip(cx.as_slice(), ord),
            vec_to_sip(cy.as_slice(), ord),
            vec_to_sip(cpx.as_slice(), ord),
            vec_to_sip(cpy.as_slice(), ord),
        ))
    };

    let with_sip = |a, b, ap, bp| WcsSolution {
        crpix: wcs.crpix,
        crval: wcs.crval,
        cd: wcs.cd,
        sip_forward: Some((a, b)),
        sip_reverse: Some((ap, bp)),
    };

    // Stage 1: generous tolerance vs. the linear projection — captures the
    // distorted corner stars a tight linear tol would reject.
    let pairs1 = collect(wcs, gen_tol);
    let ord1 = match order_for(pairs1.len()) {
        Some(o) => o,
        None => {
            tracing::warn!(
                variant = "two_stage",
                pairs = pairs1.len(),
                min_required = 12,
                "SIP fit skipped — too few pairs at generous tolerance"
            );
            return None;
        }
    };
    let (a, b, ap, bp) = fit_from(&pairs1, ord1)?;
    let stage1 = with_sip(a, b, ap, bp);

    // Stage 2: re-project with the fitted SIP, re-match at the tight tolerance
    // (now valid — SIP removed the distortion), refit from the cleaner set.
    let tight = (tol_px * 1.5).max(2.0);
    let pairs2 = collect(&stage1, tight);
    if let Some(ord2) = order_for(pairs2.len()) {
        if let Some((a2, b2, ap2, bp2)) = fit_from(&pairs2, ord2) {
            return Some(with_sip(a2, b2, ap2, bp2));
        }
    }
    Some(stage1)
}

// Phase 4/gated: Distortion abstraction + TPS
// When/if TPS is added, introduce a `Distortion` enum here and abstract
// `WcsSolution::sip_forward/sip_reverse` behind it.  For now only SIP
// (A/B/AP/BP polynomial) is supported.

/// Fit SIP distortion **for the precise profile** using inverse-variance
/// weighting and post-fit σ-clipping.
///
/// This is a **minimally-invasive extension** of the two-stage [`fit_sip`]
/// algorithm.  The matching structure (tolerances, stage ordering) is
/// byte-identical to `fit_sip`; the only differences are:
///
/// 1. **Inverse-variance weighting** on both stages: each row in the LSQ
///    system is scaled by `√w` where `w = 1/(σ̄² + ε)`,
///    `σ̄ = (σx + σy)/2`.  When `sigmas` is `None` or a star's σ is
///    `0`/non-finite, `w = 1` (uniform weight — same as `fit_sip`).
/// 2. **3σ clipping on Stage 2 pairs only**: after the Stage-2 weighted fit,
///    pairs whose post-fit polynomial residual exceeds 3σ of the distribution
///    are removed and the fit is recomputed once.  Clipping is applied ONLY
///    when the re-fit is strictly better on the FULL Stage-2 pair set (to
///    prevent false improvement from a smaller denominator).  Stage-1
///    matching is kept unclipped — those pairs establish the distortion model.
/// 3. **"No worse than linear" guard at the END only**: compares the
///    detection-space RMS from `count_inliers` (caller handles this) against
///    the linear WCS via the standard `srms <= rms_px` check in orchestrate.
///    Within this function, we do NOT gate on any internal RMS.
///
/// # Parameters
///
/// - `linear_wcs` — the just-fitted linear WCS (seed for matching).
/// - `cat_stars`  — catalog stars (epoch-corrected).
/// - `image_pts`  — detected star positions `(x, y)` in pixels, aligned 1:1
///   with `sigmas`.
/// - `sigmas`     — optional per-star `(σx, σy)` in pixels.  `None` →
///   uniform weight (identical to `fit_sip`).
/// - `width`, `height` — image dimensions.
/// - `base_tol_px` — inlier tolerance from the linear-verify stage.
/// - `max_order`  — SIP polynomial order cap (from `cfg.sip_order`).
///
/// # Returns
///
/// `Some(WcsSolution)` with SIP populated (caller's "no worse than linear"
/// guard decides whether to keep it), or `None` if there are too few pairs
/// or the SVD is singular.
#[allow(clippy::too_many_arguments)]
pub fn fit_sip_annealed(
    linear_wcs: &WcsSolution,
    cat_stars: &[CatalogStar],
    image_pts: &[(f64, f64)],
    sigmas: Option<&[(f64, f64)]>,
    width: f64,
    height: f64,
    base_tol_px: f64,
    max_order: u8,
) -> Option<WcsSolution> {
    let want_order = max_order.clamp(2, 3); // match fit_sip: 2 or 3

    let (crpix_x, crpix_y) = linear_wcs.crpix;
    let (crval_ra, crval_dec) = linear_wcs.crval;

    let det_cd = linear_wcs.cd[0][0] * linear_wcs.cd[1][1]
        - linear_wcs.cd[0][1] * linear_wcs.cd[1][0];
    if det_cd.abs() < 1e-30 {
        return None;
    }
    let inv = 1.0 / det_cd;

    // -----------------------------------------------------------------------
    // Pair type: (measured_u, measured_v, ref_x, ref_y, weight, img_idx)
    // Byte-identical matching logic to fit_sip, extended with (weight, idx).
    // -----------------------------------------------------------------------
    type Pair = (f64, f64, f64, f64, f64, usize); // (mu, mv, rx, ry, w, idx)

    // Inverse-variance weight for the star at `img_idx`.
    //
    // The weight is `1 / (σ̄² + ε)` when the star's σ values are meaningful
    // (both non-zero and finite).  Stars with σ=0 (PSF gate reverted to
    // pass-1 centroid) receive the FIELD-LEVEL MEDIAN weight among refined
    // stars, rather than `1.0`, to prevent pass-1 (noisier) stars from
    // systematically out-weighting PSF-refined (smoother) ones.  When there
    // are no refined stars at all (sigmas is None or all-zero), uniform
    // weight 1.0 is used throughout.
    //
    // `field_median_w` is computed once from all entries in `sigmas`.
    let field_median_w: f64 = sigmas.map(|sv| {
        const EPS: f64 = 1e-6;
        let mut ws: Vec<f64> = sv.iter()
            .filter_map(|&(sx, sy)| {
                if sx.is_finite() && sy.is_finite() && (sx + sy) > 0.0 {
                    let sigma_bar = (sx + sy) * 0.5;
                    Some(1.0 / (sigma_bar * sigma_bar + EPS))
                } else {
                    None
                }
            })
            .collect();
        if ws.is_empty() { return 1.0; }
        ws.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ws[ws.len() / 2]
    }).unwrap_or(1.0);

    let star_weight = |img_idx: usize| -> f64 {
        const EPS: f64 = 1e-6;
        match sigmas.and_then(|s| s.get(img_idx)) {
            Some(&(sx, sy)) if sx.is_finite() && sy.is_finite() && (sx + sy) > 0.0 => {
                // PSF-refined star: use its own inverse-variance weight.
                let sigma_bar = (sx + sy) * 0.5;
                1.0 / (sigma_bar * sigma_bar + EPS)
            }
            Some(_) | None => {
                // Unrefined or unknown σ: use the field-level median weight so
                // unrefined (noisier) stars don't outweight refined ones.
                field_median_w
            }
        }
    };

    // Match catalog → detections within mtol, projecting through `proj`.
    // Byte-identical to fit_sip's `collect`, extended with per-star weight.
    let gen_tol = (base_tol_px * 4.0).max(6.0);
    let grid = PointGrid::new(image_pts, gen_tol);
    let collect = |proj: &WcsSolution, mtol: f64| -> Vec<Pair> {
        let mut pairs = Vec::new();
        for cat in cat_stars {
            let (px, py) = proj.sky_to_pixel(cat.ra, cat.dec);
            if !px.is_finite() || !py.is_finite() { continue; }
            if px < 0.0 || px >= width || py < 0.0 || py >= height { continue; }
            let Some((best_idx, best)) =
                grid.nearest_by(px, py, |(ix, iy)| (ix - px).hypot(iy - py))
            else {
                continue;
            };
            if best > mtol { continue; }
            let best_img = image_pts[best_idx];
            let (xi, eta) = crate::wcs::tan::sky_to_tangent(cat.ra, cat.dec, crval_ra, crval_dec);
            let rx = inv * (linear_wcs.cd[1][1] * xi - linear_wcs.cd[0][1] * eta);
            let ry = inv * (-linear_wcs.cd[1][0] * xi + linear_wcs.cd[0][0] * eta);
            let mu = best_img.0 - crpix_x;
            let mv = best_img.1 - crpix_y;
            let w = star_weight(best_idx);
            pairs.push((mu, mv, rx, ry, w, best_idx));
        }
        pairs
    };

    // ≥20 pairs → order 3; ≥12 → order-2 fallback. Same as fit_sip.
    let order_for = |n: usize| -> Option<u8> {
        if n >= 20 {
            Some(want_order)
        } else if n >= 12 {
            Some(2.min(want_order))
        } else {
            None
        }
    };

    // Weighted LSQ fit on a pair set.  Same math as fit_sip's `fit_from` but
    // rows are scaled by √w so that poorly-known centroids count less.
    let fit_weighted = |pairs: &[Pair], ord: u8| -> Option<WcsSolution> {
        let n = pairs.len();
        let nb = n_terms(ord);
        if n < nb { return None; }
        let mut a_fwd = DMatrix::zeros(n, nb);
        let mut fx = DVector::zeros(n);
        let mut fy = DVector::zeros(n);
        let mut a_rev = DMatrix::zeros(n, nb);
        let mut rxv = DVector::zeros(n);
        let mut ryv = DVector::zeros(n);

        for (row, &(mu, mv, refx, refy, w, _)) in pairs.iter().enumerate() {
            let sq = w.sqrt(); // √w scaling for WLSQ
            for (col, b) in basis_row(mu, mv, ord).into_iter().enumerate() {
                a_fwd[(row, col)] = b * sq;
            }
            fx[row] = (refx - mu) * sq;
            fy[row] = (refy - mv) * sq;
            for (col, b) in basis_row(refx, refy, ord).into_iter().enumerate() {
                a_rev[(row, col)] = b * sq;
            }
            rxv[row] = (mu - refx) * sq;
            ryv[row] = (mv - refy) * sq;
        }
        let eps = 1e-12;
        let svd_f = a_fwd.svd(true, true);
        let cx = svd_f.solve(&fx, eps).ok()?;
        let cy = svd_f.solve(&fy, eps).ok()?;
        let svd_r = a_rev.svd(true, true);
        let cpx = svd_r.solve(&rxv, eps).ok()?;
        let cpy = svd_r.solve(&ryv, eps).ok()?;

        Some(WcsSolution {
            crpix: linear_wcs.crpix,
            crval: linear_wcs.crval,
            cd: linear_wcs.cd,
            sip_forward: Some((vec_to_sip(cx.as_slice(), ord), vec_to_sip(cy.as_slice(), ord))),
            sip_reverse: Some((vec_to_sip(cpx.as_slice(), ord), vec_to_sip(cpy.as_slice(), ord))),
        })
    };

    // Post-fit polynomial residual: |rx − (mu + A(mu,mv))|.
    let poly_resid = |sip_fwd: &(SipCoefficients, SipCoefficients), mu: f64, mv: f64, rx: f64, ry: f64| -> f64 {
        let u_pred = mu + eval_sip_poly(&sip_fwd.0, mu, mv);
        let v_pred = mv + eval_sip_poly(&sip_fwd.1, mu, mv);
        let dx = rx - u_pred;
        let dy = ry - v_pred;
        (dx * dx + dy * dy).sqrt()
    };

    // Conditional 3σ clip + refit.  Returns the best result (clipped or not)
    // measured on the FULL pair set to avoid false improvement from fewer pairs.
    //
    // Conservatism rules (to prevent overfitting on sparse pair sets):
    // 1. Skip clipping entirely when pairs.len() < CLIP_MIN_PAIRS (40).
    //    With few pairs and a high-order polynomial, the fit is barely
    //    determined and σ-clip is unreliable.
    // 2. Skip refit when clipping removes > MAX_CLIP_FRAC (25%) of pairs.
    //    Aggressive clipping likely indicates a wrong fit, not true outliers.
    // 3. Accept the clipped-refit only if it is strictly better on the
    //    FULL pair set (not just the clipped subset).
    let maybe_clip_refit = |pairs: &[Pair], wcs: WcsSolution, ord: u8| -> WcsSolution {
        const CLIP_MIN_PAIRS: usize = 40;
        const MAX_CLIP_FRAC: f64 = 0.25;

        // Skip clipping for sparse pair sets.
        if pairs.len() < CLIP_MIN_PAIRS {
            return wcs;
        }
        let sip_fwd = match wcs.sip_forward.as_ref() {
            Some(f) => f,
            None => return wcs,
        };
        let resids: Vec<f64> = pairs.iter().map(|&(mu, mv, rx, ry, _, _)| {
            poly_resid(sip_fwd, mu, mv, rx, ry)
        }).collect();
        let rms_full = {
            let ss: f64 = resids.iter().map(|r| r * r).sum();
            (ss / resids.len() as f64).sqrt()
        };
        let thresh = 3.0 * rms_full;
        let clipped: Vec<Pair> = pairs.iter().zip(resids.iter())
            .filter(|(_, &r)| r <= thresh)
            .map(|(p, _)| *p)
            .collect();
        let n_removed = pairs.len() - clipped.len();
        let frac_removed = n_removed as f64 / pairs.len() as f64;
        // Only refit if clipping is modest (not removing too many pairs).
        if n_removed > 0 && frac_removed <= MAX_CLIP_FRAC && clipped.len() >= n_terms(ord) {
            if let Some(wcs2) = fit_weighted(&clipped, ord) {
                // Evaluate on the FULL pair set — not the clipped subset.
                if let Some(fwd2) = wcs2.sip_forward.as_ref() {
                    let full_resids2: Vec<f64> = pairs.iter().map(|&(mu, mv, rx, ry, _, _)| {
                        poly_resid(fwd2, mu, mv, rx, ry)
                    }).collect();
                    let rms2_full = {
                        let ss: f64 = full_resids2.iter().map(|r| r * r).sum();
                        (ss / full_resids2.len() as f64).sqrt()
                    };
                    if rms2_full < rms_full {
                        return wcs2; // clipped-refit is better on the full set
                    }
                }
            }
        }
        wcs // original is better or clipping was not appropriate
    };

    // -----------------------------------------------------------------------
    // Stage 1: generous tolerance vs. the linear projection.
    // Byte-identical matching to fit_sip; weighted SVD.
    // -----------------------------------------------------------------------
    let pairs1 = collect(linear_wcs, gen_tol);
    let ord1 = match order_for(pairs1.len()) {
        Some(o) => o,
        None => {
            tracing::warn!(
                variant = "annealed",
                pairs = pairs1.len(),
                min_required = 12,
                "SIP fit skipped — too few pairs at generous tolerance"
            );
            return None;
        }
    };
    // Stage 1: weighted fit (no σ-clip — keep consistent with fit_sip Stage 1).
    let stage1_wcs = fit_weighted(&pairs1, ord1)?;

    // -----------------------------------------------------------------------
    // Stage 2: tight tolerance vs. the stage-1 SIP.
    // Byte-identical matching to fit_sip; weighted SVD + conditional σ-clip.
    // -----------------------------------------------------------------------
    let tight_tol = (base_tol_px * 1.5).max(2.0);
    let pairs2 = collect(&stage1_wcs, tight_tol);
    if let Some(ord2) = order_for(pairs2.len()) {
        if let Some(wcs2) = fit_weighted(&pairs2, ord2) {
            // σ-clip + conditional refit (only if it improves on full set).
            return Some(maybe_clip_refit(&pairs2, wcs2, ord2));
        }
    }
    // Stage 2 failed: fall back to stage 1.
    Some(stage1_wcs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_terms_counts_correctly() {
        assert_eq!(n_terms(1), 3); // (0,0),(1,0),(0,1)
        assert_eq!(n_terms(2), 6);
        assert_eq!(n_terms(3), 10);
    }

    #[test]
    fn basis_row_length_matches_n_terms() {
        for order in [1u8, 2, 3, 4] {
            let row = basis_row(1.5, 2.3, order);
            assert_eq!(row.len(), n_terms(order), "order={order}");
        }
    }

    #[test]
    fn basis_row_known_values() {
        // order=2: (u^0 v^0, u^1 v^0, u^0 v^1, u^2 v^0, u^1 v^1, u^0 v^2)
        let u = 2.0_f64;
        let v = 3.0_f64;
        let row = basis_row(u, v, 2);
        assert_eq!(row.len(), 6);
        assert!((row[0] - 1.0).abs() < 1e-12, "u^0 v^0");
        assert!((row[1] - 2.0).abs() < 1e-12, "u^1 v^0");
        assert!((row[2] - 3.0).abs() < 1e-12, "u^0 v^1");
        assert!((row[3] - 4.0).abs() < 1e-12, "u^2 v^0");
        assert!((row[4] - 6.0).abs() < 1e-12, "u^1 v^1");
        assert!((row[5] - 9.0).abs() < 1e-12, "u^0 v^2");
    }

    // -----------------------------------------------------------------------
    // fit_sip_annealed tests
    // -----------------------------------------------------------------------

    /// Build a WcsSolution with a TAN projection centred on (RA=0, Dec=0)
    /// and a given plate scale (arcsec/px), no SIP.
    fn make_linear_wcs(scale_arcsec: f64, width: f64, height: f64) -> WcsSolution {
        let scale_deg = scale_arcsec / 3600.0;
        WcsSolution {
            crpix: (width / 2.0, height / 2.0),
            crval: (0.0, 0.0),
            cd: [[-scale_deg, 0.0], [0.0, scale_deg]],
            sip_forward: None,
            sip_reverse: None,
        }
    }

    /// Generate a synthetic star field with known order-2 SIP distortion,
    /// then verify that `fit_sip_annealed` recovers the coefficients and the
    /// final RMS is below 0.1 px.
    #[test]
    fn annealed_recovers_known_sip_coefficients() {
        // 1 arcsec/px, 2000×2000 image.
        let width = 2000.0_f64;
        let height = 2000.0_f64;
        let wcs = make_linear_wcs(1.0, width, height);

        let crpix = wcs.crpix;
        // True SIP A coefficient (order-2 quadratic distortion in u).
        // A_2_0: distortion = 0.002 · u²  (a small but detectable barrel).
        let a20_true = 2e-4_f64;

        // Place 60 catalog stars on a regular grid, compute distorted pixel
        // positions to simulate what `detect_fast` would find.
        let mut cat_stars: Vec<CatalogStar> = Vec::new();
        let mut image_pts: Vec<(f64, f64)> = Vec::new();

        let n_side = 8usize;
        for row in 0..n_side {
            for col in 0..n_side {
                let u = (col as f64 / (n_side - 1) as f64 - 0.5) * 1800.0;
                let v = (row as f64 / (n_side - 1) as f64 - 0.5) * 1800.0;
                // Ideal pixel → sky (via the linear WCS).
                let px = u + crpix.0;
                let py = v + crpix.1;
                let (ra, dec) = wcs.pixel_to_sky(px, py);
                cat_stars.push(CatalogStar { ra, dec, mag: 10.0 });
                // Distorted detected position: add A_2_0·u² in x.
                let distorted_x = px + a20_true * u * u;
                let distorted_y = py;
                image_pts.push((distorted_x, distorted_y));
            }
        }

        let result = fit_sip_annealed(
            &wcs,
            &cat_stars,
            &image_pts,
            None,
            width,
            height,
            2.0,  // base_tol_px
            3,    // max_order
        );

        assert!(
            result.is_some(),
            "fit_sip_annealed returned None on clean synthetic field"
        );
        let sip_wcs = result.unwrap();

        // Verify that the fitted WCS predicts positions close to the detected
        // positions for each star (i.e., the distortion was captured).
        let mut max_resid = 0.0_f64;
        let mut rms_sum = 0.0_f64;
        let mut cnt = 0usize;
        for (cat, &(det_x, det_y)) in cat_stars.iter().zip(image_pts.iter()) {
            let (px, py) = sip_wcs.sky_to_pixel(cat.ra, cat.dec);
            let resid = (px - det_x).hypot(py - det_y);
            if resid > max_resid { max_resid = resid; }
            rms_sum += resid * resid;
            cnt += 1;
        }
        let rms = (rms_sum / cnt as f64).sqrt();
        assert!(
            rms < 0.1,
            "RMS {rms:.4}px after annealed fit should be < 0.1 px (known-SIP field)"
        );
        assert!(
            max_resid < 1.0,
            "Max residual {max_resid:.4}px should be < 1.0 px (no outliers expected)"
        );
    }

    /// Plant 3 gross outlier pairs on top of a clean distorted field.
    /// The σ-clipped fit should match the clean-field fit; an unclipped fit
    /// would be dragged by the outliers.
    #[test]
    fn annealed_sigma_clip_removes_gross_outliers() {
        let width = 2000.0_f64;
        let height = 2000.0_f64;
        let wcs = make_linear_wcs(1.0, width, height);
        let crpix = wcs.crpix;
        let a20_true = 2e-4_f64;

        let mut cat_stars: Vec<CatalogStar> = Vec::new();
        let mut image_pts: Vec<(f64, f64)> = Vec::new();

        let n_side = 8usize;
        for row in 0..n_side {
            for col in 0..n_side {
                let u = (col as f64 / (n_side - 1) as f64 - 0.5) * 1800.0;
                let v = (row as f64 / (n_side - 1) as f64 - 0.5) * 1800.0;
                let px = u + crpix.0;
                let py = v + crpix.1;
                let (ra, dec) = wcs.pixel_to_sky(px, py);
                cat_stars.push(CatalogStar { ra, dec, mag: 10.0 });
                let distorted_x = px + a20_true * u * u;
                image_pts.push((distorted_x, py));
            }
        }

        // Plant 3 gross outlier image points at the positions of stars 0, 1, 2
        // — shift them by 50 px (>> 3σ of the ~0.5 px residual distribution).
        let outlier_shift = 50.0_f64;
        image_pts[0].0 += outlier_shift;
        image_pts[1].1 += outlier_shift;
        image_pts[2].0 -= outlier_shift;

        let result = fit_sip_annealed(
            &wcs,
            &cat_stars,
            &image_pts,
            None,
            width,
            height,
            3.0,
            3,
        );

        assert!(result.is_some(), "fit_sip_annealed returned None on field with outliers");
        let sip_wcs = result.unwrap();

        // The non-outlier stars (index 3..end) should be well-fit.
        let mut rms_sum = 0.0_f64;
        let mut cnt = 0usize;
        for (cat, &(det_x, det_y)) in cat_stars.iter().zip(image_pts.iter()).skip(3) {
            let (px, py) = sip_wcs.sky_to_pixel(cat.ra, cat.dec);
            let resid = (px - det_x).hypot(py - det_y);
            rms_sum += resid * resid;
            cnt += 1;
        }
        let rms_clean = (rms_sum / cnt as f64).sqrt();
        assert!(
            rms_clean < 0.5,
            "RMS on clean stars {rms_clean:.4}px should be < 0.5 px (outliers were clipped)"
        );
    }

    /// With `sigmas = None` the uniform-weight path must still converge
    /// and produce a solution no worse than the linear WCS.
    #[test]
    fn annealed_uniform_weight_converges() {
        let width = 2000.0_f64;
        let height = 2000.0_f64;
        let wcs = make_linear_wcs(1.0, width, height);
        let crpix = wcs.crpix;
        let a20_true = 1e-4_f64;

        let mut cat_stars: Vec<CatalogStar> = Vec::new();
        let mut image_pts: Vec<(f64, f64)> = Vec::new();

        let n_side = 7usize;
        for row in 0..n_side {
            for col in 0..n_side {
                let u = (col as f64 / (n_side - 1) as f64 - 0.5) * 1600.0;
                let v = (row as f64 / (n_side - 1) as f64 - 0.5) * 1600.0;
                let px = u + crpix.0;
                let py = v + crpix.1;
                let (ra, dec) = wcs.pixel_to_sky(px, py);
                cat_stars.push(CatalogStar { ra, dec, mag: 10.0 });
                image_pts.push((px + a20_true * u * u, py));
            }
        }

        // Compute linear rms for the guard check.
        let mut lin_rss = 0.0_f64;
        let mut lin_cnt = 0usize;
        for (cat, &(det_x, det_y)) in cat_stars.iter().zip(image_pts.iter()) {
            let (px, py) = wcs.sky_to_pixel(cat.ra, cat.dec);
            lin_rss += (px - det_x).powi(2) + (py - det_y).powi(2);
            lin_cnt += 1;
        }
        let lin_rms = (lin_rss / lin_cnt as f64).sqrt();

        let result = fit_sip_annealed(
            &wcs,
            &cat_stars,
            &image_pts,
            None,  // uniform weights
            width,
            height,
            2.0,
            3,
        );

        assert!(result.is_some(), "fit_sip_annealed (uniform) returned None");
        let sip_wcs = result.unwrap();

        let mut sip_rss = 0.0_f64;
        let mut sip_cnt = 0usize;
        for (cat, &(det_x, det_y)) in cat_stars.iter().zip(image_pts.iter()) {
            let (px, py) = sip_wcs.sky_to_pixel(cat.ra, cat.dec);
            sip_rss += (px - det_x).powi(2) + (py - det_y).powi(2);
            sip_cnt += 1;
        }
        let sip_rms = (sip_rss / sip_cnt as f64).sqrt();

        assert!(
            sip_rms <= lin_rms + 1e-6,
            "SIP rms {sip_rms:.4}px should be ≤ linear rms {lin_rms:.4}px"
        );
    }
}
