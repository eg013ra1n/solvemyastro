//! The acceptance gate: whether a solve's numbers are a real solution or a
//! noise alignment, and the in-field bright-star count the gate's
//! dense-field ratio rule reads.
//!
//! The gate never looks at pixels or the solver's internals: a caller builds
//! [`GateMetrics`] from a solve result and asks [`Gate::accepts`]. Calibrated
//! from a real-library audit: `inlier_ratio` is the primary discriminator
//! (real solves >= ~0.08, false positives <= ~0.001); rms and absolute inlier
//! count do not separate, so those are loose backstops.

use crate::Caches;

/// Fields with more bright catalogue stars than this are "dense": the
/// inlier-ratio rule applies there only. Sparse fields have too few stars to
/// trust a ratio, and noise alignments occur in dense regions.
pub const DENSE_FIELD_MIN_EXPECTED: usize = 100;

/// The gate's thresholds. `Default` is Athenaeum's own defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    /// Base verification tolerance, arcseconds; [`gate_tol_px`] turns it into
    /// the per-frame pixel tolerance the RMS ceiling is relative to.
    pub base_tolerance_arcsec: f64,
    /// The RMS ceiling, as a multiple of the pixel tolerance.
    pub rms_max_px_mult: f64,
    /// Minimum inlier ratio in a dense field.
    pub min_inlier_ratio: f64,
    /// Absolute inlier floor (weak backstop).
    pub inlier_floor: usize,
    /// Lowest physically plausible recovered scale, arcsec/px.
    pub scale_sanity_min: f64,
    /// Highest physically plausible recovered scale, arcsec/px.
    pub scale_sanity_max: f64,
    /// The recovered scale may differ from the header's by at most this
    /// factor either way (coarse backstop: a wrong FOCALLEN is what the blind
    /// fallback is for).
    pub scale_header_tol: f64,
}

impl Default for Gate {
    fn default() -> Self {
        Self {
            base_tolerance_arcsec: 8.0,
            rms_max_px_mult: 2.5,
            min_inlier_ratio: 0.04,
            inlier_floor: 6,
            scale_sanity_min: 0.05,
            scale_sanity_max: 60.0,
            scale_header_tol: 8.0,
        }
    }
}

/// What the gate reads from one solve.
#[derive(Clone, Debug, PartialEq)]
pub struct GateMetrics {
    /// Matched stars.
    pub inliers: usize,
    /// Bright catalogue stars expected in the field ([`expected_bright_in_fov`]).
    pub expected_in_fov: usize,
    /// Residual RMS, pixels.
    pub rms_px: f64,
    /// The pixel tolerance, [`gate_tol_px`] of the solve's scale.
    pub tol_px: f64,
    /// Matched stars over detected stars.
    pub inlier_ratio: f64,
    /// The solved scale, arcsec/px.
    pub recovered_scale_arcsec: f64,
    /// The scale the frame's own optics imply, when it declares them.
    pub header_scale_arcsec: Option<f64>,
}

impl Gate {
    /// Whether `m` is an acceptable solution. The checks run in a fixed
    /// order; the first failure refuses.
    pub fn accepts(&self, m: &GateMetrics) -> bool {
        // Geometric fit must be tight (scale-relative ceiling; loose backstop).
        if !m.rms_px.is_finite() || m.rms_px > self.rms_max_px_mult * m.tol_px {
            return false;
        }
        // Absolute inlier floor (weak backstop).
        if m.inliers < self.inlier_floor {
            return false;
        }
        // Dense-field confidence ratio: the primary false-positive gate.
        if m.expected_in_fov > DENSE_FIELD_MIN_EXPECTED && m.inlier_ratio < self.min_inlier_ratio {
            return false;
        }
        // Recovered scale must be physically plausible.
        if !(self.scale_sanity_min..=self.scale_sanity_max).contains(&m.recovered_scale_arcsec) {
            return false;
        }
        // ...and not wildly off the header scale when the header had one.
        if let Some(hs) = m.header_scale_arcsec {
            if hs > 0.0 {
                let r = m.recovered_scale_arcsec / hs;
                if r < 1.0 / self.scale_header_tol || r > self.scale_header_tol {
                    return false;
                }
            }
        }
        true
    }
}

/// Convert an arcsecond-scale base tolerance to a per-frame pixel tolerance,
/// clamped to [4, 20] px (10 when the scale is ~0). Tight fields get smaller
/// pixel tolerances (fewer false matches); wide fields get larger ones
/// (slightly defocused stars still count).
pub fn gate_tol_px(pixel_scale_arcsec: f64, base_arcsec: f64) -> f64 {
    if pixel_scale_arcsec.abs() < 1e-6 {
        return 10.0;
    }
    (base_arcsec / pixel_scale_arcsec).clamp(4.0, 20.0)
}

/// Bright catalogue stars (at the verification magnitude limit) in the solved
/// field: a cone around `crval` on the deep cache (`Legacy`) or the base layer
/// (`Layered`, where every bright star lives in the brightest tier).
///
/// The image size is not known here, so the radius is reconstructed from the
/// scale and the detection count: `min(3, scale * sqrt(n) / 3600 / 2 * 1.41)`
/// degrees when both are positive, else 1.5. A cone error counts 0 and is
/// logged.
///
/// A change to the radius formula or to the magnitude limit changes the
/// accept/refuse verdicts an embedding app has cached: the app must bump its
/// solve-recipe version (Athenaeum's `config::SOLVE_RECIPE_VERSION`).
pub fn expected_bright_in_fov(
    caches: &Caches<'_>,
    crval: (f64, f64),
    pixel_scale_arcsec: f64,
    total_detected: usize,
    epoch: f64,
) -> usize {
    let radius_deg = if pixel_scale_arcsec > 0.0 && total_detected > 0 {
        3.0_f64.min(pixel_scale_arcsec * (total_detected as f64).sqrt() / 3600.0 / 2.0 * 1.41)
    } else {
        1.5
    };
    let count = |c: &crate::StarCache| match c.cone(
        crval.0,
        crval.1,
        radius_deg,
        crate::orchestrate::VERIFY_MAG_LIMIT,
        epoch,
    ) {
        Ok(v) => v.len(),
        Err(e) => {
            tracing::warn!(error = %e, radius_deg, "bright-star cone failed; counting 0");
            0
        }
    };
    match *caches {
        Caches::Legacy { deep, .. } => count(deep),
        Caches::Layered { layers } => layers.first().map(|&c| count(c)).unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{build_cache, StarCache, StarRecord};

    fn star(ra: f64, dec: f64, mag: f32) -> StarRecord {
        StarRecord { ra, dec, mag, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 }
    }

    /// The gate refuses a six-match noise alignment in a dense field even
    /// though the solver returned it. Real numbers, from a DSLR frame with no
    /// FOCALLEN/XPIXSZ in its header that "solved" at Dec -68 deg, a sky it
    /// was never pointed at: 6 inliers, ratio 0.037, in a field where the
    /// catalogue expects hundreds of stars (2026-09-05).
    #[test]
    fn gate_refuses_a_six_match_noise_alignment() {
        let gate = Gate::default();
        let noise = GateMetrics {
            inliers: 6,
            expected_in_fov: 480,
            rms_px: 1.159,
            tol_px: gate_tol_px(0.2178, gate.base_tolerance_arcsec),
            inlier_ratio: 0.0373,
            recovered_scale_arcsec: 0.2178,
            header_scale_arcsec: None,
        };
        assert!(
            !gate.accepts(&noise),
            "six matches at ratio 0.037 is a noise alignment, not a solve"
        );
        // The same field with a real solve's confidence still passes.
        let real = GateMetrics { inliers: 60, inlier_ratio: 0.125, ..noise };
        assert!(gate.accepts(&real));
    }

    #[test]
    fn gate_table() {
        // The table's own floor of 12 (the default is 6; checked below).
        let gate = Gate { inlier_floor: 12, ..Gate::default() };
        let base = GateMetrics {
            inliers: 40,
            expected_in_fov: 800,
            rms_px: 1.2,
            tol_px: 6.0,
            inlier_ratio: 0.30,
            recovered_scale_arcsec: 1.8,
            header_scale_arcsec: Some(1.9),
        };
        // Good solve passes.
        assert!(gate.accepts(&base));
        // Loose RMS rejected.
        assert!(!gate.accepts(&GateMetrics { rms_px: 20.0, ..base.clone() }));
        // Low ratio on a DENSE field rejected (the calibrated primary gate).
        assert!(!gate.accepts(&GateMetrics { inlier_ratio: 0.001, ..base.clone() }));
        // Sparse field (expected <= 100) NOT punished by the ratio rule.
        assert!(gate.accepts(&GateMetrics {
            expected_in_fov: 40,
            inlier_ratio: 0.001,
            inliers: 14,
            ..base.clone()
        }));
        // Too few inliers rejected.
        assert!(!gate.accepts(&GateMetrics { inliers: 8, ..base.clone() }));
        // Absurd recovered scale rejected.
        assert!(!gate.accepts(&GateMetrics { recovered_scale_arcsec: 0.001, ..base.clone() }));
        // Recovered scale wildly off header scale rejected (ratio ~10 > tol 8).
        assert!(!gate.accepts(&GateMetrics {
            recovered_scale_arcsec: 20.0,
            header_scale_arcsec: Some(1.9),
            ..base.clone()
        }));
        // Non-finite RMS is never a real solve (degenerate solves can produce NaN/inf).
        assert!(!gate.accepts(&GateMetrics { rms_px: f64::NAN, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { rms_px: f64::INFINITY, ..base.clone() }));
        // No header scale: the header-tolerance rule is skipped entirely.
        assert!(gate.accepts(&GateMetrics {
            header_scale_arcsec: None,
            recovered_scale_arcsec: 20.0,
            ..base.clone()
        }));
        // Scale-sanity MAX bound.
        assert!(!gate.accepts(&GateMetrics { recovered_scale_arcsec: 100.0, ..base.clone() }));
        // The default floor is 6: five refuses, six passes.
        let d = Gate::default();
        assert!(!d.accepts(&GateMetrics { inliers: 5, ..base.clone() }));
        assert!(d.accepts(&GateMetrics { inliers: 6, ..base.clone() }));
    }

    /// The strict edges of the gate. The inputs are exact in f64 (4.0 * 2.5 =
    /// 10.0, 8.0 / 1.0 = 8.0, 1.0 / 8.0 = 0.125), so no assert is float-fragile.
    #[test]
    fn gate_boundaries_are_pinned() {
        let gate = Gate::default();
        let base = GateMetrics {
            inliers: 40,
            expected_in_fov: 800,
            rms_px: 1.0,
            tol_px: 4.0,
            inlier_ratio: 0.30,
            recovered_scale_arcsec: 1.0,
            header_scale_arcsec: Some(1.0),
        };
        assert!(gate.accepts(&base));
        // rms exactly at the ceiling accepts; just above refuses.
        assert!(gate.accepts(&GateMetrics { rms_px: 10.0, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { rms_px: 10.5, ..base.clone() }));
        // 100 expected stars is exempt from the ratio rule; 101 is gated.
        let low = GateMetrics { inlier_ratio: 0.001, ..base.clone() };
        assert!(gate.accepts(&GateMetrics { expected_in_fov: 100, ..low.clone() }));
        assert!(!gate.accepts(&GateMetrics { expected_in_fov: 101, ..low }));
        // A header ratio exactly at the tolerance accepts, either way round;
        // beyond it refuses.
        assert!(gate.accepts(&GateMetrics { recovered_scale_arcsec: 8.0, ..base.clone() }));
        assert!(gate.accepts(&GateMetrics { recovered_scale_arcsec: 0.125, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { recovered_scale_arcsec: 8.5, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { recovered_scale_arcsec: 0.1, ..base }));
    }

    #[test]
    fn gate_defaults_are_the_apps_defaults() {
        let g = Gate::default();
        assert_eq!(g.base_tolerance_arcsec, 8.0);
        assert_eq!(g.rms_max_px_mult, 2.5);
        assert_eq!(g.min_inlier_ratio, 0.04);
        assert_eq!(g.inlier_floor, 6);
        assert_eq!(g.scale_sanity_min, 0.05);
        assert_eq!(g.scale_sanity_max, 60.0);
        assert_eq!(g.scale_header_tol, 8.0);
        assert_eq!(DENSE_FIELD_MIN_EXPECTED, 100);
    }

    #[test]
    fn gate_tol_px_clamps() {
        // Narrow field (0.5"/px): 8 / 0.5 = 16, within the band.
        assert!((gate_tol_px(0.5, 8.0) - 16.0).abs() < 1e-9);
        // Wide field (5"/px): 8 / 5 = 1.6, clamps UP to 4.
        assert!((gate_tol_px(5.0, 8.0) - 4.0).abs() < 1e-9);
        // Tiny scale: huge tolerance clamps to 20.
        assert!((gate_tol_px(0.1, 8.0) - 20.0).abs() < 1e-9);
        // Large scale: clamps to 4.
        assert!((gate_tol_px(100.0, 8.0) - 4.0).abs() < 1e-9);
        // Zero pixel scale (defensive): the safe 10 px default.
        assert!((gate_tol_px(0.0, 8.0) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn expected_bright_in_fov_uses_the_first_layer_and_the_radius_formula() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let epoch = 2016.0;
        let build = |name: &str, recs: Vec<StarRecord>| {
            let dir = tmp.path().join(name);
            build_cache(recs, &dir, epoch, |_| {}).expect("build_cache");
            StarCache::open(&dir).expect("open")
        };
        // Mag 10 at 0.30 and 0.40 deg, mag 13 at 0.10 deg from (10, 20).
        let first = build(
            "first",
            vec![star(10.0, 20.30, 10.0), star(10.0, 20.40, 10.0), star(10.0, 20.10, 13.0)],
        );
        // The second layer holds a mag-10 star inside the radius: a union of
        // layers would count 2, the first layer alone counts 1.
        let second = build("second", vec![star(10.0, 20.20, 10.0)]);
        let crval = (10.0, 20.0);

        // 3.6"/px, 250 000 detections: radius 3.6 * 500 / 3600 / 2 * 1.41 = 0.3525 deg.
        let legacy = Caches::deep_only(&first);
        assert_eq!(expected_bright_in_fov(&legacy, crval, 3.6, 250_000, epoch), 1);
        // No detections: the radius is 1.5 deg, both mag-10 stars count and
        // the mag-13 star never does.
        assert_eq!(expected_bright_in_fov(&legacy, crval, 3.6, 0, epoch), 2);
        // Layered: the first layer alone is counted.
        let stack = [&first, &second];
        let layered = Caches::layered(&stack);
        assert_eq!(expected_bright_in_fov(&layered, crval, 3.6, 250_000, epoch), 1);
        // The second layer alone would count 1 as well; only the first is read.
        let swapped = [&second, &first];
        assert_eq!(
            expected_bright_in_fov(&Caches::layered(&swapped), crval, 3.6, 250_000, epoch),
            1
        );
        // An empty stack counts 0.
        let none: [&StarCache; 0] = [];
        assert_eq!(expected_bright_in_fov(&Caches::layered(&none), crval, 3.6, 0, epoch), 0);
    }

    #[test]
    fn expected_bright_in_fov_caps_the_radius_and_falls_back_without_a_scale() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let epoch = 2016.0;
        let dir = tmp.path().join("wide");
        // Mag-10 stars at 1.4, 1.6, 2.5 and 3.2 deg north of (10, 20).
        let recs = vec![
            star(10.0, 21.4, 10.0),
            star(10.0, 21.6, 10.0),
            star(10.0, 22.5, 10.0),
            star(10.0, 23.2, 10.0),
        ];
        build_cache(recs, &dir, epoch, |_| {}).expect("build_cache");
        let cache = StarCache::open(&dir).expect("open");
        let caches = Caches::deep_only(&cache);
        let crval = (10.0, 20.0);
        // The formula gives 3.6 * 1e4 / 3600 / 2 * 1.41 = 7.05 deg; the cap
        // is 3 deg, so 1.4, 1.6 and 2.5 count and 3.2 does not.
        assert_eq!(expected_bright_in_fov(&caches, crval, 3.6, 100_000_000, epoch), 3);
        // No usable scale with detections: the radius is 1.5 deg, only 1.4 counts.
        assert_eq!(expected_bright_in_fov(&caches, crval, 0.0, 100, epoch), 1);
    }

    /// The gate counts bright stars to `VERIFY_MAG_LIMIT`; moving it moves
    /// every embedding app's gate verdicts.
    #[test]
    fn the_verify_magnitude_limit_is_pinned() {
        assert_eq!(
            crate::orchestrate::VERIFY_MAG_LIMIT,
            12.0,
            "VERIFY_MAG_LIMIT moves the gate's bright-star count and with it embedding apps' \
             gate verdicts: bump their solve-recipe version (Athenaeum's config::SOLVE_RECIPE_VERSION)"
        );
    }
}
