//! Global fallback-gate logic for PSF centroid refinement.
//!
//! After `detect_fast` with `centroid_refine=true`, the individual per-star
//! gates inside `rustafits` handle bad fits (non-physical parameters, centre
//! shift > 2 px). This module adds a **field-level** gate: if the field as a
//! whole is undersampled (median FWHM < 1.5 px) or the median SNR is too low
//! (< 10), PSF fitting is harmful rather than helpful and we revert to the
//! pass-1 (unrefined) positions.
//!
//! Phase 1 scope: refined centroids are passed unchanged to the quad-builder
//! and the existing fit + matching pipeline. The σ values (`FastStar.sx/sy`)
//! are produced by `rustafits` and exposed on the solution as `psf_fwhm_px`,
//! but σ-weighted least-squares is Phase 2 and is NOT implemented here.

/// Minimum acceptable median FWHM for PSF refinement.
///
/// Below 1.5 px the PSF is undersampled: a Moffat fit on a 3-pixel stamp
/// has fewer degrees of freedom than parameters and the centroid update is
/// dominated by the discrete pixel grid rather than the PSF profile.
const FWHM_MIN_PX: f32 = 1.5;

/// Minimum acceptable median aperture SNR for the whole field.
///
/// If the brightest detected stars have SNR < 10, the Moffat fit noise floor
/// exceeds the PSF signal and refinement makes centroids *worse* on average.
const SNR_MIN: f32 = 10.0;

/// Outcome of the global PSF-refinement gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefineDecision {
    /// Accept refinement: use PSF-fitted centroids.
    Accept,
    /// Reject refinement: revert to pass-1 centroids (field-level gate fired).
    Revert { reason: RefineRevertReason },
}

/// Why the global gate reverted to pass-1 centroids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefineRevertReason {
    /// Median FWHM < [`FWHM_MIN_PX`] — undersampled field.
    Undersampled,
    /// Median SNR < [`SNR_MIN`] — low-SNR field.
    LowSnr,
}

/// Evaluate whether the refined centroid set should be used or discarded.
///
/// `stars` is the slice of detected stars **after** rustafits PSF refinement;
/// the per-star gate inside `rustafits` has already handled individual bad
/// fits, so `stars[i].fwhm` is `0.0` for unrefined stars and > 0.0 for
/// successfully refined ones.
///
/// Returns [`RefineDecision::Accept`] if the refined centroids should be fed
/// to the quad-builder, or [`RefineDecision::Revert`] with a reason if the
/// field-level gate fires.
///
/// # Parameters
///
/// - `stars` — detected stars after rustafits PSF refinement.
///
/// # Returns
///
/// [`RefineDecision`] and the **median FWHM** (px) of successfully refined
/// stars (0.0 if no stars were refined or the gate fires).
pub(crate) fn evaluate_refinement(
    stars: &[astroimage::FastStar],
) -> (RefineDecision, f64) {
    // Collect FWHM values from stars that were successfully refined.
    let mut fwhms: Vec<f32> = stars
        .iter()
        .filter(|s| s.fwhm > 0.0)
        .map(|s| s.fwhm)
        .collect();

    // If no stars were refined at all, treat as low-SNR / no signal.
    if fwhms.is_empty() {
        return (
            RefineDecision::Revert {
                reason: RefineRevertReason::LowSnr,
            },
            0.0,
        );
    }

    fwhms.sort_unstable_by(|a, b| a.total_cmp(b));
    let median_fwhm = fwhms[fwhms.len() / 2];

    // Gate 1: undersampled PSF.
    if median_fwhm < FWHM_MIN_PX {
        return (
            RefineDecision::Revert {
                reason: RefineRevertReason::Undersampled,
            },
            median_fwhm as f64,
        );
    }

    // Gate 2: low-SNR field (use the full star list, not just refined ones).
    let mut snrs: Vec<f32> = stars.iter().map(|s| s.snr).collect();
    if snrs.is_empty() {
        return (
            RefineDecision::Revert {
                reason: RefineRevertReason::LowSnr,
            },
            median_fwhm as f64,
        );
    }
    snrs.sort_unstable_by(|a, b| a.total_cmp(b));
    let median_snr = snrs[snrs.len() / 2];

    if median_snr < SNR_MIN {
        return (
            RefineDecision::Revert {
                reason: RefineRevertReason::LowSnr,
            },
            median_fwhm as f64,
        );
    }

    (RefineDecision::Accept, median_fwhm as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_star(fwhm: f32, snr: f32) -> astroimage::FastStar {
        astroimage::FastStar {
            x: 0.0,
            y: 0.0,
            raw_x: 0.0,
            raw_y: 0.0,
            peak: 0.0,
            flux: 0.0,
            snr,
            sx: 0.0,
            sy: 0.0,
            fwhm,
        }
    }

    #[test]
    fn accepts_normal_field() {
        // 5 stars with FWHM≈3 px and SNR≈50 — should be accepted.
        let stars: Vec<astroimage::FastStar> = (0..5)
            .map(|_| make_star(3.0, 50.0))
            .collect();
        let (decision, median_fwhm) = evaluate_refinement(&stars);
        assert_eq!(decision, RefineDecision::Accept);
        assert!((median_fwhm - 3.0).abs() < 0.1, "median_fwhm={median_fwhm}");
    }

    #[test]
    fn rejects_undersampled_field() {
        // FWHM = 1.0 px — below the 1.5 px floor.
        let stars: Vec<astroimage::FastStar> = (0..5)
            .map(|_| make_star(1.0, 50.0))
            .collect();
        let (decision, _) = evaluate_refinement(&stars);
        assert_eq!(
            decision,
            RefineDecision::Revert { reason: RefineRevertReason::Undersampled }
        );
    }

    #[test]
    fn rejects_low_snr_field() {
        // FWHM is fine (3 px) but SNR = 5 — below the 10 floor.
        let stars: Vec<astroimage::FastStar> = (0..5)
            .map(|_| make_star(3.0, 5.0))
            .collect();
        let (decision, _) = evaluate_refinement(&stars);
        assert_eq!(
            decision,
            RefineDecision::Revert { reason: RefineRevertReason::LowSnr }
        );
    }

    #[test]
    fn rejects_empty_star_list() {
        let stars: Vec<astroimage::FastStar> = Vec::new();
        let (decision, median_fwhm) = evaluate_refinement(&stars);
        assert_eq!(
            decision,
            RefineDecision::Revert { reason: RefineRevertReason::LowSnr }
        );
        assert_eq!(median_fwhm, 0.0);
    }

    #[test]
    fn stars_with_fwhm_zero_not_counted_in_median() {
        // 3 stars with fwhm=0 (unrefined by per-star gate), 2 with fwhm=3.
        // The field gate should only count the 2 refined stars.
        // Their median FWHM is 3.0 px — above the floor → accept.
        let mut stars: Vec<astroimage::FastStar> = (0..3)
            .map(|_| make_star(0.0, 50.0)) // unrefined (per-star gate fired)
            .collect();
        stars.extend((0..2).map(|_| make_star(3.0, 50.0)));
        let (decision, median_fwhm) = evaluate_refinement(&stars);
        assert_eq!(decision, RefineDecision::Accept);
        assert!((median_fwhm - 3.0).abs() < 0.1, "median_fwhm={median_fwhm}");
    }

    #[test]
    fn all_unrefined_stars_give_low_snr_revert() {
        // All stars have fwhm=0 (none refined) → treated as no-refinement.
        let stars: Vec<astroimage::FastStar> = (0..5)
            .map(|_| make_star(0.0, 50.0))
            .collect();
        let (decision, _) = evaluate_refinement(&stars);
        assert_eq!(
            decision,
            RefineDecision::Revert { reason: RefineRevertReason::LowSnr }
        );
    }
}
