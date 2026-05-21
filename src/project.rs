//! Gnomonic (TAN) catalog projection for per-trial quad building.
//!
//! # Key invariant
//!
//! A camera is itself a gnomonic projector: it maps sky positions onto a flat
//! focal plane centred at the boresight. Therefore, for any trial whose
//! tangent point equals the true field centre, the image-pixel positions are a
//! **pure similarity** (rotation + uniform scale + translation) of the
//! gnomonically-projected catalog positions. A pure similarity preserves all
//! 6 pairwise-distance ratios exactly, so `build_quads` on image pixels and
//! `build_quads` on projected catalog arcsec yield quads from the same family
//! — the `match_quads` ratio test is then exactly satisfiable.
//!
//! This is the root-cause fix for a class of bugs in legacy catalog
//! indexes that built quads with great-circle distances (a *different*
//! metric from image Euclidean), which diverges O(θ²) for wide fields and
//! pushes the correct quad outside the hash bin.
//!
//! Implementation: `wcs::tan::sky_to_tangent` → convert to arcsec, then
//! feed the projected positions into `quad::build_quads` with the same
//! `group_size` used for the image side.

use crate::{math::great_circle_deg, wcs::tan, CatalogStar};

/// Arcsec per degree.
const ARCSEC_PER_DEG: f64 = 3600.0;

/// Maximum angular separation from the tangent point beyond which a catalog
/// star is dropped before projection (degrees). Gnomonic projection diverges
/// for points near the opposite hemisphere; 89° is a safe guard since valid
/// trial cones are never wider than a few degrees.
const MAX_SEP_DEG: f64 = 89.0;

/// Project catalog stars onto the tangent plane centred at `(ra0, dec0)`.
///
/// Each star is mapped via gnomonic (TAN) projection:
/// `(ξ_deg, η_deg) = sky_to_tangent(ra, dec, ra0, dec0)`,
/// then converted to arcsec: `(ξ_deg · 3600, η_deg · 3600)`.
///
/// Stars more than [`MAX_SEP_DEG`] from the tangent point are dropped to
/// keep the projection numerically stable; valid catalog cones are far
/// smaller, so this is a purely defensive guard and never panics.
///
/// The returned coordinates are in **arcsec** on the tangent plane so that
/// [`crate::quad::build_quads`] operates in the same units regardless of
/// whether it is called on image pixels or projected catalog stars.  (Ratios
/// are unit-invariant; the absolute scale matters for the affine-fit
/// translation that [`crate::wcs_fit::affine_to_wcs`] converts back to CD.)
pub fn project_catalog(stars: &[CatalogStar], ra0: f64, dec0: f64) -> Vec<(f64, f64)> {
    stars
        .iter()
        .filter(|s| great_circle_deg(s.ra, s.dec, ra0, dec0) <= MAX_SEP_DEG)
        .map(|s| {
            let (xi_deg, eta_deg) = tan::sky_to_tangent(s.ra, s.dec, ra0, dec0);
            (xi_deg * ARCSEC_PER_DEG, eta_deg * ARCSEC_PER_DEG)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_star(ra: f64, dec: f64) -> CatalogStar {
        CatalogStar { ra, dec, mag: 8.0 }
    }

    // -----------------------------------------------------------------------
    // Basic projection
    // -----------------------------------------------------------------------

    #[test]
    fn tangent_point_projects_to_origin() {
        let (ra0, dec0) = (180.0, 40.0);
        let stars = vec![make_star(ra0, dec0)];
        let proj = project_catalog(&stars, ra0, dec0);
        assert_eq!(proj.len(), 1);
        assert!(proj[0].0.abs() < 1e-9 && proj[0].1.abs() < 1e-9,
            "tangent point must project to (0,0), got {:?}", proj[0]);
    }

    #[test]
    fn near_tangent_point_is_arcsec_linear() {
        // 1 arcmin = 1/60 deg offset in dec; should project to ~60 arcsec in η.
        let (ra0, dec0) = (90.0, 30.0);
        let delta_arcmin = 1.0_f64;
        let stars = vec![make_star(ra0, dec0 + delta_arcmin / 60.0)];
        let proj = project_catalog(&stars, ra0, dec0);
        assert_eq!(proj.len(), 1);
        let eta = proj[0].1; // η_arcsec
        // Should be ~60 arcsec; allow 0.1% error (gnomonic ≈ linear for small angles).
        assert!((eta - 60.0).abs() < 0.1, "1 arcmin offset → ~60 arcsec; got {eta}");
    }

    // -----------------------------------------------------------------------
    // Far-hemisphere guard
    // -----------------------------------------------------------------------

    #[test]
    fn star_beyond_89deg_is_dropped() {
        let (ra0, dec0) = (0.0, 0.0);
        // A star at dec=90 is 90° away → dropped.
        let stars = vec![make_star(0.0, 90.0), make_star(0.0, 0.0)];
        let proj = project_catalog(&stars, ra0, dec0);
        assert_eq!(proj.len(), 1, "the 90°-away star must be dropped");
    }

    #[test]
    fn star_just_inside_89deg_is_kept() {
        let (ra0, dec0) = (0.0, 0.0);
        // 88° away — inside the 89° guard.
        let stars = vec![make_star(0.0, 88.0)];
        let proj = project_catalog(&stars, ra0, dec0);
        assert_eq!(proj.len(), 1, "star within 89° must be kept");
    }

    // -----------------------------------------------------------------------
    // Gnomonic-consistency invariant
    // -----------------------------------------------------------------------

    /// **THE decisive correctness test.**
    ///
    /// For any 4 stars whose sky positions are mapped to image pixels via a
    /// *pure similarity* of their gnomonic tangent-plane coordinates about the
    /// same boresight, the 5 quad ratios built from the image-pixel set must
    /// equal the ratios built from the projected-catalog set to within 1e-6.
    ///
    /// This proves that `build_quads` on image pixels and `build_quads` on
    /// `project_catalog` output are the same family — the invariant the solver
    /// depends on.
    #[test]
    fn gnomonic_consistency_invariant() {
        use crate::quad::build_quads;

        let (ra0, dec0) = (180.0_f64, 40.0_f64);

        // 12 synthetic sky stars distributed around the tangent point.
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

        // Camera: gnomonic about the same boresight, 1.5"/px, 17° rotation,
        // image centre at (2048, 2048).
        let scale_arcsec_per_px = 1.5_f64;
        let rot_deg = 17.0_f64;
        let (sin_r, cos_r) = rot_deg.to_radians().sin_cos();
        let (cx, cy) = (2048.0_f64, 2048.0_f64);

        let image_pixels: Vec<(f64, f64)> = stars
            .iter()
            .map(|s| {
                let (xi_deg, eta_deg) = tan::sky_to_tangent(s.ra, s.dec, ra0, dec0);
                // Convert tangent deg → arcsec → pixels.
                let xa = xi_deg * 3600.0 / scale_arcsec_per_px;
                let ya = eta_deg * 3600.0 / scale_arcsec_per_px;
                // Apply rotation + translate to image centre.
                (cos_r * xa - sin_r * ya + cx, sin_r * xa + cos_r * ya + cy)
            })
            .collect();

        // Projected catalog (arcsec).
        let cat_proj = project_catalog(&stars, ra0, dec0);

        // Build quads from both sides with the same group_size.
        let img_quads = build_quads(&image_pixels, image_pixels.len(), 4);
        let cat_quads = build_quads(&cat_proj, cat_proj.len(), 4);

        assert!(!img_quads.is_empty(), "image quads must not be empty");
        assert!(!cat_quads.is_empty(), "catalog quads must not be empty");

        // For the first image quad, find the matching catalog quad by star indices.
        // Since both quad lists are built from the same stars (in the same ordering),
        // matching star_idx sets identify the corresponding quads.
        let mut found_match = false;
        let mut max_dev = 0.0_f64;

        for iq in &img_quads {
            // Find a catalog quad with the same sorted star indices.
            if let Some(cq) = cat_quads.iter().find(|cq| cq.star_idx == iq.star_idx) {
                // Compute max |Δratio| across the 5 positions.
                let dev = iq
                    .ratios
                    .iter()
                    .zip(cq.ratios.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                if dev > max_dev {
                    max_dev = dev;
                }
                found_match = true;
            }
        }

        assert!(found_match, "at least one quad pair with the same star set must exist");
        assert!(
            max_dev < 1e-6,
            "gnomonic-consistency invariant: max ratio deviation must be < 1e-6 (pure similarity), got {max_dev}"
        );
    }

    #[test]
    fn empty_input_returns_empty() {
        let proj = project_catalog(&[], 180.0, 40.0);
        assert!(proj.is_empty());
    }

    #[test]
    fn multiple_stars_all_projected() {
        let stars: Vec<CatalogStar> = (0..5)
            .map(|i| make_star(180.0 + i as f64 * 0.1, 40.0 + i as f64 * 0.05))
            .collect();
        let proj = project_catalog(&stars, 180.0, 40.0);
        assert_eq!(proj.len(), stars.len(), "all near stars must be projected");
    }
}
