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
            let mut best = f64::MAX;
            let mut best_img = (0.0_f64, 0.0_f64);
            for &(ix, iy) in image_pts {
                let d = (ix - px).hypot(iy - py);
                if d < best {
                    best = d;
                    best_img = (ix, iy);
                }
            }
            if best > mtol {
                continue;
            }
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
    let gen_tol = (tol_px * 4.0).max(6.0);
    let pairs1 = collect(wcs, gen_tol);
    let ord1 = match order_for(pairs1.len()) {
        Some(o) => o,
        None => {
            eprintln!(
                "[solvemyastro] SIP skipped: only {} pairs at generous tol (need ≥12)",
                pairs1.len()
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
}
