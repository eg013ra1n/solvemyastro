//! Convert a quad-centre affine (image px → tangent-plane arcsec) to a linear
//! [`WcsSolution`].
//!
//! # Background
//!
//! After `fit_affine` succeeds, we have a 6-parameter affine:
//!
//! ```text
//! ξ_as = a1·x + b1·y + c1
//! η_as = a2·x + b2·y + c2
//! ```
//!
//! where `(x, y)` are image pixels and `(ξ_as, η_as)` are tangent-plane
//! arcsec about the trial's tangent point `(ra0, dec0)`.
//!
//! A standard WCS [`WcsSolution`] maps pixels to sky via:
//!
//! ```text
//! ξ_deg = CD1_1·(x − CRPIX1) + CD1_2·(y − CRPIX2)
//! η_deg = CD2_1·(x − CRPIX1) + CD2_2·(y − CRPIX2)
//! (ra, dec) = tangent_to_sky(ξ_deg, η_deg, CRVAL1, CRVAL2)
//! ```
//!
//! We choose `CRPIX = image_center` and derive `CRVAL` by asking: what sky
//! position does the image centre map to under the affine?  That is
//! `tangent_to_sky(a1·cx + b1·cy + c1, a2·cx + b2·cy + c2)` converted from
//! arcsec to degrees, relative to `(ra0, dec0)`.
//!
//! The CD matrix is then just `a1/3600, b1/3600` on row 0 and
//! `a2/3600, b2/3600` on row 1 — arcsec → degree conversion only.
//!
//! This mirrors `refine.rs::seed_similarity_from_affine` +
//! `service::similarity_to_wcs` in the Athenaeum reference (clean
//! reimplementation, not a dependency).

use crate::{
    quad::Affine,
    wcs::{tan, WcsSolution},
};

/// Arcsec per degree (conversion factor).
const ARCSEC_PER_DEG: f64 = 3600.0;

/// Build a linear (SIP = None) [`WcsSolution`] from a quad-centre affine.
///
/// # Parameters
///
/// - `affine` — maps image pixels → tangent-plane **arcsec** about `tangent`.
/// - `image_center` — the pixel coordinate chosen as `CRPIX` (0-based).
///   Typically `((width-1)/2, (height-1)/2)`.
/// - `tangent` — `(ra0, dec0)` degrees; the gnomonic tangent point the affine
///   was solved against.
///
/// # Returns
///
/// A [`WcsSolution`] where:
/// - `crpix = image_center`
/// - `crval = sky(image_center)` derived from the affine
/// - `cd` = the affine's linear coefficients divided by 3600 (arcsec→deg)
/// - `sip_forward = sip_reverse = None` (linear solution)
pub fn affine_to_wcs(
    affine: &Affine,
    image_center: (f64, f64),
    tangent: (f64, f64),
) -> WcsSolution {
    let (cx, cy) = image_center;
    let (ra0, dec0) = tangent;

    // Evaluate affine at image centre → tangent-plane arcsec.
    let (xi_as, eta_as) = affine.apply(cx, cy);

    // Convert arcsec → degrees for the tangent-to-sky call.
    let xi_deg = xi_as / ARCSEC_PER_DEG;
    let eta_deg = eta_as / ARCSEC_PER_DEG;

    // Sky position at image centre.
    let (crval_ra, crval_dec) = tan::tangent_to_sky(xi_deg, eta_deg, ra0, dec0);

    // CD matrix: arcsec/px → deg/px.
    let cd = [
        [affine.a1 / ARCSEC_PER_DEG, affine.b1 / ARCSEC_PER_DEG],
        [affine.a2 / ARCSEC_PER_DEG, affine.b2 / ARCSEC_PER_DEG],
    ];

    WcsSolution {
        crpix: image_center,
        crval: (crval_ra, crval_dec),
        cd,
        sip_forward: None,
        sip_reverse: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        project::project_catalog,
        quad::{build_quads, fit_affine, group_size_for, match_quads},
        CatalogStar,
    };

    /// Helper: build a synthetic WCS with known parameters, project stars
    /// gnomonically, build quads, match, fit affine, then recover via
    /// `affine_to_wcs`.
    fn synthetic_wcs_params() -> (f64, f64, f64, f64) {
        // scale 1.5"/px, rotation 17°, centre (180, 40)
        (1.5, 17.0, 180.0, 40.0)
    }

    fn build_synthetic_image_pixels(
        stars: &[CatalogStar],
        ra0: f64,
        dec0: f64,
        scale_arcsec_per_px: f64,
        rot_deg: f64,
        image_center: (f64, f64),
    ) -> Vec<(f64, f64)> {
        let (sin_r, cos_r) = rot_deg.to_radians().sin_cos();
        let (cx, cy) = image_center;
        stars
            .iter()
            .map(|s| {
                let (xi_deg, eta_deg) = tan::sky_to_tangent(s.ra, s.dec, ra0, dec0);
                let xa = xi_deg * 3600.0 / scale_arcsec_per_px;
                let ya = eta_deg * 3600.0 / scale_arcsec_per_px;
                (cos_r * xa - sin_r * ya + cx, sin_r * xa + cos_r * ya + cy)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Unit: affine_to_wcs for a known similarity
    // -----------------------------------------------------------------------

    #[test]
    fn affine_to_wcs_recovers_tangent_point() {
        // Affine that maps image centre to (0,0) in tangent arcsec → the
        // image centre's sky position should be the tangent point.
        let (ra0, dec0) = (180.0_f64, 40.0_f64);
        let scale = 1.5_f64;
        let (s, c) = 17.0_f64.to_radians().sin_cos();
        let (cx, cy) = (2048.0_f64, 2048.0_f64);
        // Affine that maps image coords relative to (cx, cy) to arcsec:
        // ξ_as = scale*c*(x-cx) - scale*s*(y-cy)
        // η_as = scale*s*(x-cx) + scale*c*(y-cy)
        let a1 = scale * c;
        let b1 = -scale * s;
        let a2 = scale * s;
        let b2 = scale * c;
        let affine = Affine {
            a1,
            b1,
            c1: -(a1 * cx + b1 * cy), // ensures affine(cx,cy) = (0,0)
            a2,
            b2,
            c2: -(a2 * cx + b2 * cy),
        };

        let wcs = affine_to_wcs(&affine, (cx, cy), (ra0, dec0));

        // CRPIX should be image centre.
        assert_eq!(wcs.crpix, (cx, cy));

        // CRVAL should be the tangent point (image centre maps to (0,0) arcsec).
        assert!(
            (wcs.crval.0 - ra0).abs() < 1e-8,
            "CRVAL RA must equal tangent-point RA: {} vs {}",
            wcs.crval.0, ra0
        );
        assert!(
            (wcs.crval.1 - dec0).abs() < 1e-8,
            "CRVAL Dec must equal tangent-point Dec: {} vs {}",
            wcs.crval.1, dec0
        );

        // Pixel scale: CD columns give scale in deg/px; geometric mean → arcsec/px.
        let recovered_scale = wcs.pixel_scale_arcsec();
        assert!(
            (recovered_scale - scale).abs() < 1e-6,
            "recovered scale {recovered_scale} must equal {scale}"
        );
    }

    // -----------------------------------------------------------------------
    // Synthetic end-to-end fit
    // -----------------------------------------------------------------------

    /// **Synthetic end-to-end correctness gate.**
    ///
    /// 12 synthetic sky stars, known WCS (1.5"/px, 17°, centre (180, 40)):
    ///
    /// 1. Project to image pixels via gnomonic similarity.
    /// 2. Project to catalog arcsec via `project_catalog`.
    /// 3. Build quads, `match_quads`, `fit_affine`, `affine_to_wcs`.
    /// 4. Assert:
    ///    - Recovered centre (CRVAL) within 1e-4° of (180, 40).
    ///    - Recovered pixel scale within 0.1% of 1.5"/px.
    ///    - `sky_to_pixel(known_star)` within 0.5 px of its true pixel.
    #[test]
    fn synthetic_end_to_end_fit() {
        let (scale, rot_deg, ra0, dec0) = synthetic_wcs_params();
        let image_center = (2048.0_f64, 2048.0_f64);

        // 12 synthetic sky stars.
        let offsets_deg: &[(f64, f64)] = &[
            (-1.5, -1.0), (1.4, -0.7), (-0.6, 1.35), (1.15, 1.1),
            (0.2, -1.45), (-1.45, 0.25), (0.75, 0.1), (-0.35, 0.8),
            (0.55, -0.5), (-1.1, -0.5), (0.9, 1.4), (-0.4, -1.3),
        ];
        let cos_dec = dec0.to_radians().cos();
        let stars: Vec<CatalogStar> = offsets_deg
            .iter()
            .enumerate()
            .map(|(i, &(dra, ddec))| CatalogStar {
                ra: ra0 + dra / cos_dec,
                dec: dec0 + ddec,
                mag: 6.0 + i as f32 * 0.2,
            })
            .collect();

        // Build image pixels (pure gnomonic similarity).
        let image_pixels = build_synthetic_image_pixels(&stars, ra0, dec0, scale, rot_deg, image_center);

        // Build catalog projected arcsec.
        let cat_proj = project_catalog(&stars, ra0, dec0);

        // Build quads on both sides.
        let group_size = group_size_for(image_pixels.len());
        let img_quads = build_quads(&image_pixels, image_pixels.len(), group_size);
        let cat_quads = build_quads(&cat_proj, cat_proj.len(), group_size);

        assert!(!img_quads.is_empty(), "image quads must not be empty");
        assert!(!cat_quads.is_empty(), "catalog quads must not be empty");

        // Match quads.
        let matches = match_quads(&img_quads, &cat_quads, 0.007);
        assert!(
            matches.len() >= 3,
            "must have >= 3 quad matches; got {}",
            matches.len()
        );

        // Fit affine.
        let affine = fit_affine(&matches, &img_quads, &cat_quads)
            .expect("fit_affine must succeed on a perfect-consistent field");

        // Convert to WCS.
        let wcs = affine_to_wcs(&affine, image_center, (ra0, dec0));

        // Gate 1: Recovered centre within 1e-4°.
        let (rec_ra, rec_dec) = wcs.crval;
        assert!(
            (rec_ra - ra0).abs() < 1e-4,
            "RA centre: expected {ra0}, got {rec_ra} (Δ={})",
            (rec_ra - ra0).abs()
        );
        assert!(
            (rec_dec - dec0).abs() < 1e-4,
            "Dec centre: expected {dec0}, got {rec_dec} (Δ={})",
            (rec_dec - dec0).abs()
        );

        // Gate 2: Recovered pixel scale within 0.1% of 1.5"/px.
        let rec_scale = wcs.pixel_scale_arcsec();
        let scale_err_frac = (rec_scale - scale).abs() / scale;
        assert!(
            scale_err_frac < 0.001,
            "pixel scale: expected {scale}\"/px, got {rec_scale}\"/px (frac error={scale_err_frac})"
        );

        // Gate 3: sky_to_pixel round-trip for all input stars.
        for (i, (star, &(px_true, py_true))) in
            stars.iter().zip(image_pixels.iter()).enumerate()
        {
            let (px_rec, py_rec) = wcs.sky_to_pixel(star.ra, star.dec);
            let dist = ((px_rec - px_true).powi(2) + (py_rec - py_true).powi(2)).sqrt();
            assert!(
                dist < 0.5,
                "star {i}: sky_to_pixel residual {dist:.4} px (> 0.5 px threshold); \
                 true=({px_true:.2},{py_true:.2}) rec=({px_rec:.2},{py_rec:.2})"
            );
        }
    }

    // -----------------------------------------------------------------------
    // CD matrix elements match the affine coefficients
    // -----------------------------------------------------------------------

    #[test]
    fn cd_matrix_equals_affine_coefficients_over_3600() {
        let affine = Affine {
            a1: 1.5,   // arcsec/px
            b1: -0.3,
            c1: 100.0,
            a2: 0.3,
            b2: 1.5,
            c2: -50.0,
        };
        let wcs = affine_to_wcs(&affine, (1000.0, 800.0), (90.0, 20.0));
        assert!((wcs.cd[0][0] - affine.a1 / 3600.0).abs() < 1e-15, "CD[0][0]");
        assert!((wcs.cd[0][1] - affine.b1 / 3600.0).abs() < 1e-15, "CD[0][1]");
        assert!((wcs.cd[1][0] - affine.a2 / 3600.0).abs() < 1e-15, "CD[1][0]");
        assert!((wcs.cd[1][1] - affine.b2 / 3600.0).abs() < 1e-15, "CD[1][1]");
    }

    #[test]
    fn sip_terms_are_none() {
        let affine = Affine { a1: 1.5, b1: 0.0, c1: 0.0, a2: 0.0, b2: 1.5, c2: 0.0 };
        let wcs = affine_to_wcs(&affine, (512.0, 512.0), (45.0, 10.0));
        assert!(wcs.sip_forward.is_none(), "SIP must be None for linear solution");
        assert!(wcs.sip_reverse.is_none(), "SIP must be None for linear solution");
    }
}
