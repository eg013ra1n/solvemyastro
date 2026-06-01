//! Frame-to-frame registration for stacking-preparation.
//!
//! Aligns a "sub" frame to a "reference" frame by matching their detected
//! star positions directly — no catalog, no external prior. Relative
//! frame-to-frame matching cancels common catalog errors and is therefore
//! more precise than anchoring each sub independently to the catalog.
//!
//! # Algorithm
//!
//! 1. Build quads from both detection lists (`build_quads` + `group_size_for`).
//! 2. Match quads (`match_quads`) and fit an initial affine from quad centres
//!    (`fit_affine`). Both sides are "image" here; the reference plays the role
//!    of "catalog" — `affine0` maps sub pixel → reference pixel.
//! 3. Star-level refit: for every sub detection, project it through `affine0`
//!    and find the nearest reference detection. Inliers within the pixel
//!    tolerance get an SVD least-squares affine refit directly on star pairs
//!    (more robust than quad centres). This is `transform`.
//! 4. Compose `refined_wcs = reference_wcs ∘ transform`. The reference WCS is
//!    the sky anchor; composing it with the pixel-to-pixel transform produces a
//!    WCS for the sub frame such that `sub_wcs.pixel_to_sky(p)` equals
//!    `reference_wcs.pixel_to_sky(transform(p))` for any sub pixel `p`.
//!
//! # Phase 2 hook
//!
//! `Registration::residual_pairs` carries `(transform(sub), ref)` per inlier.
//! The post-affine residual field encodes differential distortion between the
//! two frames and will be used to fit a relative SIP delta in Phase 2. Nothing
//! is needed here beyond populating that field.

use anyhow::{bail, Context};
use nalgebra::{DMatrix, DVector};

use crate::{
    quad::{build_quads, fit_affine, group_size_for, match_quads, Affine},
    wcs::WcsSolution,
    SolveConfig,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Result of a sub→reference frame registration.
#[derive(Clone, Debug)]
pub struct Registration {
    /// Sub-frame pixel → reference-frame pixel transform.
    ///
    /// Obtained by a star-level SVD refit on inlier pairs after an initial
    /// quad-based bootstrap. Maps `(sub_x, sub_y)` → `(ref_x, ref_y)`.
    pub transform: Affine,

    /// The sub frame's absolute WCS, derived as `reference_wcs ∘ transform`.
    ///
    /// Guarantees `refined_wcs.pixel_to_sky(p) ≈ reference_wcs.pixel_to_sky(transform(p))`
    /// for any sub pixel `p`. SIP coefficients are `None`; Phase 2 will fit a
    /// relative distortion delta and populate them.
    pub refined_wcs: WcsSolution,

    /// Inlier star pairs in **reference pixel space**: each element is
    /// `(predicted, actual)` where `predicted = transform(sub_detection)` and
    /// `actual = matched_ref_detection`.
    ///
    /// // Phase 2: fit a relative SIP delta over this residual field to capture
    /// // differential distortion between the sub and the reference frame.
    pub residual_pairs: Vec<((f64, f64), (f64, f64))>,

    /// Number of inlier star pairs used for the final affine refit.
    pub matched: usize,

    /// RMS residual `|transform(sub) − ref|` over inliers, in pixels.
    pub rms_px: f64,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Squared Euclidean distance (avoids sqrt for nearest-neighbour searches).
#[inline]
fn sq_dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    dx * dx + dy * dy
}

/// Fit a 6-parameter affine directly from `(from, to)` star-pair correspondences
/// using SVD least-squares, mirroring the approach in `quad::fit_affine` but
/// operating on raw `(f64, f64)` pairs rather than `Quad` structs.
///
/// Requires at least 3 pairs. Returns `None` when SVD fails or the solution is
/// degenerate (near-zero determinant on the linear part).
fn fit_affine_from_pairs(pairs: &[((f64, f64), (f64, f64))]) -> Option<Affine> {
    let n = pairs.len();
    if n < 3 {
        return None;
    }

    let mut a_mat = DMatrix::zeros(n, 3);
    let mut bx = DVector::zeros(n);
    let mut by = DVector::zeros(n);

    for (row, &((fx, fy), (tx, ty))) in pairs.iter().enumerate() {
        a_mat[(row, 0)] = fx;
        a_mat[(row, 1)] = fy;
        a_mat[(row, 2)] = 1.0;
        bx[row] = tx;
        by[row] = ty;
    }

    let svd = a_mat.svd(true, true);
    let eps = 1e-12;
    let px = svd.solve(&bx, eps).ok()?;
    let py = svd.solve(&by, eps).ok()?;

    Some(Affine { a1: px[0], b1: px[1], c1: px[2], a2: py[0], b2: py[1], c2: py[2] })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Align a sub frame's star detections to a reference frame, returning the
/// sub→reference pixel transform and the sub frame's derived WCS.
///
/// # Arguments
///
/// * `reference_wcs` — absolute WCS of the reference frame (already solved).
/// * `ref_detections` — star centroids in the reference frame, pixel coords.
/// * `sub_detections` — star centroids in the sub frame, pixel coords.
/// * `cfg` — solver configuration; [`SolveConfig::quad_tolerance`] is reused
///   for the quad-ratio match tolerance.
///
/// # Errors
///
/// Returns `Err` when:
/// - Either detection list has fewer than 4 stars or yields fewer than 3 quads.
/// - Quad matching + initial affine fit produce fewer than 3 matches.
/// - The star-level inlier refit is degenerate (fewer than 3 inliers or
///   near-zero affine determinant after refit).
pub fn register(
    reference_wcs: &WcsSolution,
    ref_detections: &[(f64, f64)],
    sub_detections: &[(f64, f64)],
    cfg: &SolveConfig,
) -> anyhow::Result<Registration> {
    // ------------------------------------------------------------------
    // Step 1 — Build quads from both detection lists.
    // ------------------------------------------------------------------
    if ref_detections.len() < 4 {
        bail!(
            "register: reference has {} detections — need at least 4",
            ref_detections.len()
        );
    }
    if sub_detections.len() < 4 {
        bail!(
            "register: sub has {} detections — need at least 4",
            sub_detections.len()
        );
    }

    let ref_gs = group_size_for(ref_detections.len());
    let sub_gs = group_size_for(sub_detections.len());

    let ref_quads = build_quads(ref_detections, ref_detections.len(), ref_gs);
    let sub_quads = build_quads(sub_detections, sub_detections.len(), sub_gs);

    if ref_quads.len() < 3 {
        bail!(
            "register: reference produced only {} quads — need at least 3 \
             (detection list may be too sparse or degenerate)",
            ref_quads.len()
        );
    }
    if sub_quads.len() < 3 {
        bail!(
            "register: sub produced only {} quads — need at least 3 \
             (detection list may be too sparse or degenerate)",
            sub_quads.len()
        );
    }

    // ------------------------------------------------------------------
    // Step 2 — Quad match + initial affine (quad centres, sub → ref).
    // ------------------------------------------------------------------
    // Convention: `match_quads(image, catalog, tol)` returns (image_idx, catalog_idx).
    // Here: sub plays "image", reference plays "catalog".
    let matches = match_quads(&sub_quads, &ref_quads, cfg.quad_tolerance);
    if matches.len() < 3 {
        bail!(
            "register: quad matching produced only {} matches — need at least 3",
            matches.len()
        );
    }

    // fit_affine(matches, image_quads, catalog_quads) → image→catalog transform.
    let affine0 = fit_affine(&matches, &sub_quads, &ref_quads).context(
        "register: initial quad-centre affine fit failed (xy-scale sanity or SVD failure)",
    )?;

    // ------------------------------------------------------------------
    // Step 3 — Star-level inlier refit.
    // ------------------------------------------------------------------
    // Fixed inlier tolerance: 4 px. Sufficient for typical dithered subs
    // where the quad-based bootstrap brings the error below a few pixels.
    // Phase 2 may parameterise this via SolveConfig if needed.
    const INLIER_TOL_PX: f64 = 4.0;
    let inlier_tol_sq = INLIER_TOL_PX * INLIER_TOL_PX;

    // For each sub detection, project through affine0, find the nearest
    // reference detection, and keep the pair if within tolerance.
    let mut inlier_pairs: Vec<((f64, f64), (f64, f64))> = Vec::new();
    for &(sx, sy) in sub_detections {
        let predicted = affine0.apply(sx, sy);
        // Linear scan — detection lists are typically 50–2000 stars, so this
        // is fast without a spatial index. A KD-tree can be added in Phase 2
        // if this becomes a bottleneck for very crowded fields.
        let nearest = ref_detections
            .iter()
            .copied()
            .min_by(|a, b| {
                sq_dist(predicted, *a)
                    .partial_cmp(&sq_dist(predicted, *b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if let Some(ref_pt) = nearest {
            if sq_dist(predicted, ref_pt) <= inlier_tol_sq {
                inlier_pairs.push(((sx, sy), ref_pt));
            }
        }
    }

    if inlier_pairs.len() < 3 {
        bail!(
            "register: star-level refit has only {} inliers within {} px — \
             registration failed (frames may be too different or detections too sparse)",
            inlier_pairs.len(),
            INLIER_TOL_PX
        );
    }

    // SVD refit on inlier (sub_pixel → ref_pixel) pairs — more robust than
    // quad centres because it uses actual star positions.
    let transform = fit_affine_from_pairs(&inlier_pairs)
        .context("register: star-level SVD refit failed (degenerate geometry)")?;

    // Recompute inliers and RMS under the refined transform.
    let residual_pairs: Vec<((f64, f64), (f64, f64))> = inlier_pairs
        .iter()
        .filter_map(|&((sx, sy), ref_pt)| {
            let projected = transform.apply(sx, sy);
            if sq_dist(projected, ref_pt) <= inlier_tol_sq {
                Some((projected, ref_pt))
            } else {
                None
            }
        })
        .collect();

    if residual_pairs.is_empty() {
        bail!("register: no inliers remain after refined transform — registration failed");
    }

    let matched = residual_pairs.len();
    let rms_px = {
        let sum_sq: f64 = residual_pairs
            .iter()
            .map(|&((px, py), (rx, ry))| (px - rx).powi(2) + (py - ry).powi(2))
            .sum();
        (sum_sq / matched as f64).sqrt()
    };

    // ------------------------------------------------------------------
    // Step 4 — Compose refined_wcs = reference_wcs ∘ transform.
    //
    // The affine maps sub pixel (x,y) → reference pixel (x',y'):
    //   x' = a1·x + b1·y + c1
    //   y' = a2·x + b2·y + c2
    //
    // In matrix notation, with M = [[a1, b1], [a2, b2]] and t = (c1, c2):
    //   ref_pixel = M · sub_pixel + t
    //
    // The reference WCS evaluates:
    //   sky = reference_wcs.pixel_to_sky(ref_pixel)
    //       = reference_wcs.pixel_to_sky(M · sub_pixel + t)
    //
    // We want refined_wcs such that:
    //   refined_wcs.pixel_to_sky(sub_pixel) = reference_wcs.pixel_to_sky(transform(sub_pixel))
    //
    // Expanding pixel_to_sky for a linear (no-SIP) WCS:
    //   sky = TAN( CD · (ref_pixel − CRPIX_ref) )
    //       = TAN( CD · (M · sub_pixel + t − CRPIX_ref) )
    //       = TAN( (CD · M) · sub_pixel + CD · (t − CRPIX_ref) )
    //       = TAN( CD' · (sub_pixel − CRPIX') )
    //
    // where:
    //   CD'    = CD · M           (2×2 matmul)
    //   CRPIX' = M⁻¹ · (CRPIX_ref − t)
    //
    // CRVAL stays identical — the tangent point is the reference frame's sky.
    // SIP is dropped (Phase 2 will fit a relative distortion delta).
    // ------------------------------------------------------------------
    let Affine { a1, b1, c1, a2, b2, c2 } = transform;
    let t = (c1, c2);

    // CD' = CD · M
    let cd_ref = reference_wcs.cd;
    // M = [[a1, b1], [a2, b2]]
    let cd_prime = [
        [
            cd_ref[0][0] * a1 + cd_ref[0][1] * a2,
            cd_ref[0][0] * b1 + cd_ref[0][1] * b2,
        ],
        [
            cd_ref[1][0] * a1 + cd_ref[1][1] * a2,
            cd_ref[1][0] * b1 + cd_ref[1][1] * b2,
        ],
    ];

    // CRPIX' = M⁻¹ · (CRPIX_ref − t)
    // Invert 2×2 M: det = a1·b2 − b1·a2
    let det_m = a1 * b2 - b1 * a2;
    if det_m.abs() < 1e-12 {
        bail!(
            "register: affine transform matrix is singular (|det| = {det_m:.2e}) — \
             registration failed (degenerate star configuration)"
        );
    }
    let inv_det = 1.0 / det_m;
    // M⁻¹ = (1/det) · [[b2, -b1], [-a2, a1]]
    let rhs_x = reference_wcs.crpix.0 - t.0;
    let rhs_y = reference_wcs.crpix.1 - t.1;
    let crpix_x = inv_det * (b2 * rhs_x - b1 * rhs_y);
    let crpix_y = inv_det * (-a2 * rhs_x + a1 * rhs_y);

    let refined_wcs = WcsSolution {
        crpix: (crpix_x, crpix_y),
        crval: reference_wcs.crval,
        cd: cd_prime,
        sip_forward: None,
        sip_reverse: None,
    };

    Ok(Registration { transform, refined_wcs, residual_pairs, matched, rms_px })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wcs::WcsSolution;

    // ------------------------------------------------------------------
    // Shared helpers
    // ------------------------------------------------------------------

    /// Build a WCS for testing: 1"/px scale, no rotation, centred at (ra0, dec0),
    /// with crpix at the image centre.
    fn test_wcs(ra0: f64, dec0: f64) -> WcsSolution {
        let s = 1.0 / 3600.0; // 1 arcsec/px in degrees/px
        WcsSolution {
            crpix: (2000.0, 1500.0),
            crval: (ra0, dec0),
            cd: [[-s, 0.0], [0.0, s]], // RA increases to the left (east)
            sip_forward: None,
            sip_reverse: None,
        }
    }

    /// Generate an irregular pseudo-random star field using a deterministic LCG.
    ///
    /// Regular grids are pathological for quad matching: every 4-subset has the
    /// same ratio fingerprint, swamping the matcher with false positives and
    /// causing the affine fit to fail. An irregular distribution gives each quad
    /// a unique fingerprint, matching the character of real detected-star lists.
    fn irregular_stars(n: usize, seed: u64, width: f64, height: f64) -> Vec<(f64, f64)> {
        let mut state = seed;
        let mut next = || -> f64 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        (0..n).map(|_| (next() * width, next() * height)).collect()
    }

    /// Invert an affine: given `known` mapping sub→ref, return the sub coords
    /// that map to each ref coord.
    fn invert_affine(known: &Affine, ref_pts: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let det = known.a1 * known.b2 - known.b1 * known.a2;
        let inv = 1.0 / det;
        ref_pts
            .iter()
            .map(|&(rx, ry)| {
                let dx = rx - known.c1;
                let dy = ry - known.c2;
                let sx = inv * (known.b2 * dx - known.b1 * dy);
                let sy = inv * (-known.a2 * dx + known.a1 * dy);
                (sx, sy)
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Test 1 — Synthetic field: known affine recovery
    // ------------------------------------------------------------------

    /// Build an irregular reference star field; apply a known rotation+scale+
    /// translation to make the sub coordinates; verify `register` recovers the
    /// transform so that `transform(sub) ≈ ref` to < 0.05 px.
    #[test]
    fn recovers_known_affine() {
        let ref_wcs = test_wcs(83.0, -5.0);

        // Irregular reference field — 40 stars in a 4096×3072 image plane.
        let ref_pts = irregular_stars(40, 0x1234_5678_9ABC_DEF0, 4096.0, 3072.0);

        // Known sub→ref transform: 2° rotation + 0.3% scale + small translation.
        let theta = 2.0_f64.to_radians();
        let s = 1.003_f64;
        let (sin_t, cos_t) = (theta.sin(), theta.cos());
        let known = Affine {
            a1: s * cos_t,
            b1: -s * sin_t,
            c1: 5.0,
            a2: s * sin_t,
            b2: s * cos_t,
            c2: -3.0,
        };

        // Sub detections: invert `known` applied to ref — so `known(sub) = ref`.
        let sub_pts = invert_affine(&known, &ref_pts);

        let cfg = SolveConfig::default();
        let reg = register(&ref_wcs, &ref_pts, &sub_pts, &cfg)
            .expect("register should succeed on synthetic irregular field");

        // Most stars should be inliers.
        assert!(
            reg.matched >= ref_pts.len() / 2,
            "expected at least half the stars as inliers, got {} of {}",
            reg.matched,
            ref_pts.len()
        );

        // The recovered transform should map sub_pts back to ref_pts < 0.05 px.
        for &(sx, sy) in &sub_pts {
            let (rx_pred, ry_pred) = reg.transform.apply(sx, sy);
            let nearest = ref_pts
                .iter()
                .map(|&r| (sq_dist((rx_pred, ry_pred), r), r))
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, r)| r)
                .unwrap();
            let err = sq_dist((rx_pred, ry_pred), nearest).sqrt();
            assert!(
                err < 0.05,
                "transform error {err:.4} px at ({sx:.1},{sy:.1}) — expected < 0.05 px"
            );
        }

        assert!(
            reg.rms_px < 0.05,
            "RMS {:.4} px exceeds 0.05 px threshold",
            reg.rms_px
        );
    }

    // ------------------------------------------------------------------
    // Test 2 — refined_wcs round-trip
    // ------------------------------------------------------------------

    /// For several sub pixels, verify that `refined_wcs.pixel_to_sky(p)`
    /// agrees with `reference_wcs.pixel_to_sky(transform.apply(p))` to < 1e-6°.
    #[test]
    fn refined_wcs_round_trip() {
        let ref_wcs = test_wcs(150.0, 40.0);

        // 35-star irregular field; pure rotation (1.5°), no translation.
        let ref_pts = irregular_stars(35, 0xDEAD_CAFE_BEEF_1234, 4096.0, 3072.0);
        let theta = 1.5_f64.to_radians();
        let (sin_t, cos_t) = (theta.sin(), theta.cos());
        let known = Affine {
            a1: cos_t,
            b1: -sin_t,
            c1: 0.0,
            a2: sin_t,
            b2: cos_t,
            c2: 0.0,
        };
        let sub_pts = invert_affine(&known, &ref_pts);

        let cfg = SolveConfig::default();
        let reg = register(&ref_wcs, &ref_pts, &sub_pts, &cfg)
            .expect("register should succeed for WCS round-trip test");

        // Test several sub pixels (both inside and outside the star distribution).
        let test_pixels: &[(f64, f64)] = &[
            (500.0, 400.0),
            (1000.0, 800.0),
            (1500.0, 1200.0),
            (300.0, 900.0),
            (2100.0, 600.0),
            (3800.0, 2900.0),
        ];

        for &(sx, sy) in test_pixels {
            let (ra1, dec1) = reg.refined_wcs.pixel_to_sky(sx, sy);
            let (ref_x, ref_y) = reg.transform.apply(sx, sy);
            let (ra2, dec2) = ref_wcs.pixel_to_sky(ref_x, ref_y);

            let d_ra = (ra1 - ra2).abs();
            let d_dec = (dec1 - dec2).abs();
            assert!(
                d_ra < 1e-6 && d_dec < 1e-6,
                "WCS mismatch at ({sx},{sy}): refined=({ra1:.8},{dec1:.8}), \
                 composed=({ra2:.8},{dec2:.8}), dRA={d_ra:.2e}, dDec={d_dec:.2e}"
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 3 — Robustness: 30% spurious detections
    // ------------------------------------------------------------------

    /// Adding ~30% spurious (random) detections to the sub list should still
    /// recover the correct transform.
    #[test]
    fn robust_against_spurious_detections() {
        let ref_wcs = test_wcs(200.0, -20.0);

        // 40-star irregular reference field.
        let ref_pts = irregular_stars(40, 0xCAFE_BABE_0000_0001, 4096.0, 3072.0);

        // Pure translation sub→ref.
        let known = Affine { a1: 1.0, b1: 0.0, c1: 8.0, a2: 0.0, b2: 1.0, c2: -5.0 };
        let n_real = ref_pts.len();
        let mut sub_pts = invert_affine(&known, &ref_pts);

        // Add ~30% spurious detections via a different LCG seed.
        let n_spurious = n_real * 30 / 100;
        let spurious = irregular_stars(n_spurious, 0xF00D_CAFE_1234_5678, 4096.0, 3072.0);
        sub_pts.extend_from_slice(&spurious);

        let cfg = SolveConfig::default();
        let reg = register(&ref_wcs, &ref_pts, &sub_pts, &cfg)
            .expect("register should succeed even with ~30% spurious detections");

        // At least half the real stars should be matched.
        assert!(
            reg.matched >= n_real / 2,
            "expected ≥ {} inliers, got {}",
            n_real / 2,
            reg.matched
        );
        assert!(
            reg.rms_px < 0.5,
            "RMS {:.4} px — too large, likely matched to spurious stars",
            reg.rms_px
        );
    }

    // ------------------------------------------------------------------
    // Test 4 — Degenerate cases: too few detections / no overlap
    // ------------------------------------------------------------------

    #[test]
    fn errors_on_too_few_ref_detections() {
        let ref_wcs = test_wcs(0.0, 0.0);
        let few = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]; // 3 < 4
        let sub = irregular_stars(20, 0x1111, 1000.0, 1000.0);
        let cfg = SolveConfig::default();
        assert!(
            register(&ref_wcs, &few, &sub, &cfg).is_err(),
            "should error when reference has < 4 detections"
        );
    }

    #[test]
    fn errors_on_too_few_sub_detections() {
        let ref_wcs = test_wcs(0.0, 0.0);
        let ref_pts = irregular_stars(20, 0x2222, 1000.0, 1000.0);
        let few = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]; // 3 < 4
        let cfg = SolveConfig::default();
        assert!(
            register(&ref_wcs, &ref_pts, &few, &cfg).is_err(),
            "should error when sub has < 4 detections"
        );
    }

    #[test]
    fn errors_when_no_quad_overlap() {
        // Two completely unrelated irregular star distributions separated by 10000 px.
        // Their quad fingerprints may superficially match but the resulting affine
        // will produce zero inliers in the star-level refit (no real star within
        // 4 px of any projected sub detection).
        let ref_wcs = test_wcs(0.0, 0.0);
        // Reference: irregular field in [0, 100]² — very small image region.
        let ref_pts: Vec<(f64, f64)> = irregular_stars(12, 0xAAAA, 100.0, 100.0);
        // Sub: irregular field in [50000, 50100]² — completely disjoint in pixel space.
        let sub_pts: Vec<(f64, f64)> = irregular_stars(12, 0xBBBB, 100.0, 100.0)
            .into_iter()
            .map(|(x, y)| (x + 50000.0, y + 50000.0))
            .collect();
        let cfg = SolveConfig::default();
        // register must return Err: either quad matching fails, affine fit fails,
        // or the star-level refit yields < 3 inliers.
        assert!(
            register(&ref_wcs, &ref_pts, &sub_pts, &cfg).is_err(),
            "should error when frames have no geometric overlap"
        );
    }
}
