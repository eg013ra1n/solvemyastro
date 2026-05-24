//! Bayesian log-odds verification (astrometry.net style).
//!
//! Replaces the brittle 5%-of-FOV-catalog inlier floor with a principled
//! foreground-vs-background log-likelihood ratio. The Bayesian gate is
//! density-adaptive by construction:
//!
//! - More catalog stars in FOV → each match contributes less log-odds →
//!   demands more matches in crowded fields (anti-false-positive).
//! - Larger search area → background is diluted → wide fields verify with
//!   fewer matches.
//! - Tighter centroid residuals → matches contribute dramatically more →
//!   rewards centroid accuracy.
//!
//! Formula (per [astrometry.net's verify.c](https://github.com/dstndstn/astrometry.net/blob/main/solver/verify.c)):
//!
//! ```text
//! log_odds = Σ_{matched stars} (log_fg_i - log_bg)
//! log_fg_i = log((1 - D) / (2π · σ² · NR)) - d_i² / (2σ²)
//! log_bg   = log(1 / A)
//! ```
//!
//! - `D`  = distractor fraction (default 0.25)
//! - `σ`  = positional tolerance (px)
//! - `NR` = catalog reference stars in FOV
//! - `A`  = search area (px²)
//! - `d_i`= pixel distance of each inlier from its predicted position

/// Default acceptance threshold (astrometry.net's `DLOG_ODDS_MIN`).
/// `log(1e6) ≈ 13.815` — odds 1,000,000:1 in favour of the WCS being correct.
pub const DEFAULT_LOG_ODDS_MIN: f64 = 13.815_510_557_964_274; // log(1e6)

/// Default distractor fraction. 0.25 means we expect ~25% of catalog stars in
/// FOV to have no matching detection (below detection threshold, masked,
/// near edge, etc.). Matches astrometry.net's default.
pub const DEFAULT_DISTRACTOR_FRACTION: f64 = 0.25;

/// Safety floor: a candidate must have at least this many inliers regardless
/// of log-odds. Guards against numerical pathologies where 1-2 perfect
/// matches can mathematically clear the Bayesian gate.
pub const DEFAULT_MIN_ABSOLUTE_INLIERS: usize = 4;

/// Safety cap: seed-stage RMS must not exceed `max_rms_sigma_factor × σ`,
/// otherwise the matches are too scattered to plausibly be real. 3.0 means
/// "if RMS > 3σ, those aren't real inliers".
pub const DEFAULT_MAX_RMS_SIGMA_FACTOR: f64 = 3.0;

/// Compute the Bayesian log-odds for a candidate WCS, given its inlier set.
///
/// Returns `f64::NEG_INFINITY` for degenerate inputs (empty inliers, σ ≤ 0,
/// zero catalog stars, zero area). Otherwise the formula:
///
/// ```text
/// log_odds = Σ (log_fg - log_bg)
/// log_fg = log((1 - D) / (2π σ² NR)) - d² / (2σ²)
/// log_bg = log(1 / A)
/// ```
///
/// # Parameters
///
/// - `inlier_distances_px`: pixel residual of each inlier (length = match count).
/// - `sigma_px`: positional tolerance σ used for the match.
/// - `n_catalog_in_fov`: total catalog stars NR in the FOV.
/// - `area_px2`: search area A in pixels² (typically `width × height`).
/// - `distractor`: distractor fraction D (`0 ≤ D < 1`).
pub fn log_odds(
    inlier_distances_px: &[f64],
    sigma_px: f64,
    n_catalog_in_fov: usize,
    area_px2: f64,
    distractor: f64,
) -> f64 {
    if inlier_distances_px.is_empty()
        || sigma_px <= 0.0
        || n_catalog_in_fov == 0
        || area_px2 <= 0.0
        || !(0.0..1.0).contains(&distractor)
    {
        return f64::NEG_INFINITY;
    }
    let sig2 = sigma_px * sigma_px;
    let nr = n_catalog_in_fov as f64;
    // log of the peak (d=0) foreground likelihood.
    let log_fg_peak = ((1.0 - distractor) / (2.0 * std::f64::consts::PI * sig2 * nr)).ln();
    let log_bg = (1.0_f64 / area_px2).ln();
    let mut acc = 0.0_f64;
    for &d in inlier_distances_px {
        if !d.is_finite() {
            continue;
        }
        let log_fg = log_fg_peak - (d * d) / (2.0 * sig2);
        acc += log_fg - log_bg;
    }
    acc
}

/// Configuration for [`accept_bayesian`].
#[derive(Clone, Copy, Debug)]
pub struct BayesianGateConfig {
    /// Minimum acceptable log-odds.
    pub log_odds_threshold: f64,
    /// Minimum absolute inlier count (safety floor against numerical edge cases).
    pub min_absolute_inliers: usize,
    /// `seed_rms ≤ max_rms_sigma_factor × σ` required.
    pub max_rms_sigma_factor: f64,
    /// Distractor fraction `D`.
    pub distractor: f64,
}

impl Default for BayesianGateConfig {
    fn default() -> Self {
        Self {
            log_odds_threshold: DEFAULT_LOG_ODDS_MIN,
            min_absolute_inliers: DEFAULT_MIN_ABSOLUTE_INLIERS,
            max_rms_sigma_factor: DEFAULT_MAX_RMS_SIGMA_FACTOR,
            distractor: DEFAULT_DISTRACTOR_FRACTION,
        }
    }
}

/// The accept/reject decision returned by [`accept_bayesian`].
#[derive(Debug, Clone)]
pub enum AcceptanceDecision {
    Accepted,
    Rejected(String),
}

/// Combine the Bayesian log-odds gate with two safety constraints:
/// an absolute inlier floor and a maximum-RMS cap. Both safety constraints
/// are belt-and-suspenders — the Bayesian gate alone is the principled
/// criterion, but in real-world numerical edge cases the floors catch a
/// few pathological inputs cheaply.
pub fn accept_bayesian(
    log_odds_val: f64,
    n_inliers: usize,
    seed_rms_px: f64,
    sigma_px: f64,
    cfg: &BayesianGateConfig,
) -> AcceptanceDecision {
    if n_inliers < cfg.min_absolute_inliers {
        return AcceptanceDecision::Rejected(format!(
            "inliers {} < min_absolute {}",
            n_inliers, cfg.min_absolute_inliers
        ));
    }
    let max_rms = cfg.max_rms_sigma_factor * sigma_px;
    if seed_rms_px > max_rms {
        return AcceptanceDecision::Rejected(format!(
            "seed_rms {:.2} > {:.2} ({:.1}×σ scatter cap)",
            seed_rms_px, max_rms, cfg.max_rms_sigma_factor
        ));
    }
    if !log_odds_val.is_finite() || log_odds_val < cfg.log_odds_threshold {
        return AcceptanceDecision::Rejected(format!(
            "log_odds {:.2} < threshold {:.2}",
            log_odds_val, cfg.log_odds_threshold
        ));
    }
    AcceptanceDecision::Accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_inliers_returns_neg_inf() {
        assert_eq!(log_odds(&[], 1.0, 100, 1e6, 0.25), f64::NEG_INFINITY);
    }

    #[test]
    fn zero_sigma_returns_neg_inf() {
        assert_eq!(log_odds(&[0.0; 10], 0.0, 100, 1e6, 0.25), f64::NEG_INFINITY);
    }

    #[test]
    fn perfect_matches_clear_threshold_for_0023_case() {
        // 0023 scenario: 13 inliers, σ=1.5 px (adaptive_tol_px floor), seed_rms ≈ 1.04 px,
        // NR=2486 catalog stars, A ≈ 3009×3009 (Samyang frame).
        // We use 1.04 px for each distance to model seed_rms behaviour.
        let distances = vec![1.04_f64; 13];
        let lo = log_odds(&distances, 1.5, 2486, 3009.0 * 3009.0, 0.25);
        assert!(lo > DEFAULT_LOG_ODDS_MIN,
            "0023's 13 matches @ rms 1.04 must clear log(1e6)=13.8; got {lo:.2}");
    }

    #[test]
    fn single_match_does_not_clear_threshold() {
        // A 1-match WCS in a typical mid-density field must NOT pass.
        let lo = log_odds(&[0.5], 1.5, 1000, 6e6, 0.25);
        assert!(lo < DEFAULT_LOG_ODDS_MIN,
            "single match should not clear threshold; got {lo:.2}");
    }

    #[test]
    fn doubling_n_catalog_decreases_per_match_logodds_by_ln2() {
        let distances = vec![0.5_f64; 5];
        let lo_a = log_odds(&distances, 1.5, 1000, 1e7, 0.25);
        let lo_b = log_odds(&distances, 1.5, 2000, 1e7, 0.25);
        let per_match_diff = (lo_b - lo_a) / distances.len() as f64;
        assert!((per_match_diff - (-2.0_f64.ln())).abs() < 1e-9,
            "per-match drop should be exactly -ln(2); got {per_match_diff}");
    }

    #[test]
    fn doubling_area_increases_per_match_logodds_by_ln2() {
        let distances = vec![0.5_f64; 5];
        let lo_a = log_odds(&distances, 1.5, 1000, 1e7, 0.25);
        let lo_b = log_odds(&distances, 1.5, 1000, 2e7, 0.25);
        let per_match_diff = (lo_b - lo_a) / distances.len() as f64;
        assert!((per_match_diff - 2.0_f64.ln()).abs() < 1e-9,
            "per-match gain should be exactly +ln(2); got {per_match_diff}");
    }

    #[test]
    fn accept_bayesian_rejects_below_floor() {
        let cfg = BayesianGateConfig::default();
        let dec = accept_bayesian(50.0, 2, 0.5, 1.5, &cfg);
        assert!(matches!(dec, AcceptanceDecision::Rejected(_)),
            "should reject when inliers below floor");
    }

    #[test]
    fn accept_bayesian_rejects_scattered_rms() {
        let cfg = BayesianGateConfig::default();
        // log_odds high enough, inliers above floor, but seed_rms = 6 px > 3σ = 4.5
        let dec = accept_bayesian(50.0, 10, 6.0, 1.5, &cfg);
        assert!(matches!(dec, AcceptanceDecision::Rejected(_)),
            "should reject scattered matches even with high log_odds");
    }

    #[test]
    fn accept_bayesian_accepts_normal_solve() {
        let cfg = BayesianGateConfig::default();
        let dec = accept_bayesian(50.0, 20, 0.6, 1.5, &cfg);
        assert!(matches!(dec, AcceptanceDecision::Accepted),
            "should accept a normal solve");
    }

    // ===== FL-extreme stress tests =====
    //
    // The Bayesian gate is supposed to be density-adaptive across the full
    // focal-length range users shoot (50mm wide-field down to 6500mm
    // narrow-field). These tests parameterize the formula with realistic
    // (NR, A, σ) values for several FL extremes and confirm the gate's
    // accept/reject decisions are sensible.

    /// Realistic per-FL probe: returns `log_odds` for a CORRECT-WCS scenario
    /// (matches at typical `seed_rms` ≈ 0.7σ).
    fn lo_for_correct_wcs(
        n_matches: usize,
        sigma_px: f64,
        n_catalog_in_fov: usize,
        area_px2: f64,
    ) -> f64 {
        let d = sigma_px * 0.7;
        let distances: Vec<f64> = vec![d; n_matches];
        log_odds(&distances, sigma_px, n_catalog_in_fov, area_px2, 0.25)
    }

    #[test]
    fn ultra_wide_50mm_milky_way_accepts() {
        // 50mm lens on full-frame ≈ 30° FOV. Image 6000×4000 px → A = 2.4e7.
        // Dense Milky Way: ~500k catalog stars in 30° FOV at G<19.
        // σ ≈ 2 px. Typical: ~200 matches when WCS correct.
        let lo = lo_for_correct_wcs(200, 2.0, 500_000, 2.4e7);
        assert!(lo > DEFAULT_LOG_ODDS_MIN,
            "50mm Milky Way (200 matches, 500k cat, σ=2) must clear threshold; got {lo:.2}");
    }

    #[test]
    fn ultra_wide_50mm_high_galactic_latitude_still_accepts() {
        // Same lens away from MW plane: ~30k catalog stars in 30° FOV.
        let lo = lo_for_correct_wcs(150, 2.0, 30_000, 2.4e7);
        assert!(lo > DEFAULT_LOG_ODDS_MIN,
            "50mm high-lat (150 matches, 30k cat) must clear threshold; got {lo:.2}");
    }

    #[test]
    fn narrow_2000mm_typical_accepts() {
        // C8/RC8 ~ 2000mm. FOV ~0.5°, NR ≈ 200 catalog stars in FOV.
        // σ ≈ 1.5 px, ~50 matches typical.
        let lo = lo_for_correct_wcs(50, 1.5, 200, 1.2e7);
        assert!(lo > DEFAULT_LOG_ODDS_MIN,
            "2000mm typical (50 matches, 200 cat) must clear threshold; got {lo:.2}");
    }

    #[test]
    fn very_narrow_4000mm_sparse_accepts_with_few_matches() {
        // C14/EdgeHD 14 + reducer ≈ 4000mm. FOV ~0.2°, NR ≈ 15 catalog
        // stars in FOV. σ ≈ 1 px (good seeing), ~6 matches typical.
        let lo = lo_for_correct_wcs(6, 1.0, 15, 1.2e7);
        assert!(lo > DEFAULT_LOG_ODDS_MIN,
            "4000mm sparse (6 matches, 15 cat) must clear threshold; got {lo:.2}");
    }

    #[test]
    fn extremely_narrow_6500mm_marginal_passes_with_safety_floor() {
        // 6500mm Cassegrain (FOV ~0.07°) with ~5 catalog stars in FOV.
        // 4-of-5 perfect matches must pass both log_odds AND the
        // absolute-inlier safety floor (≥4).
        let lo = lo_for_correct_wcs(4, 1.0, 5, 1.2e7);
        let cfg = BayesianGateConfig::default();
        let dec = accept_bayesian(lo, 4, 0.7, 1.0, &cfg);
        assert!(matches!(dec, AcceptanceDecision::Accepted),
            "6500mm with 4 perfect matches must pass (log_odds={lo:.2}); decision={dec:?}");
    }

    #[test]
    fn extremely_narrow_6500mm_too_few_inliers_rejects() {
        // Same field, only 3 matches — below the absolute floor.
        let lo = lo_for_correct_wcs(3, 1.0, 5, 1.2e7);
        let cfg = BayesianGateConfig::default();
        let dec = accept_bayesian(lo, 3, 0.7, 1.0, &cfg);
        assert!(matches!(dec, AcceptanceDecision::Rejected(_)),
            "3 inliers must be rejected by the safety floor regardless of log_odds; got {dec:?}");
    }

    #[test]
    fn ultra_dense_field_resists_false_positive_with_few_matches() {
        // Adversarial: 50mm Milky Way with only 5 inliers — could be
        // a chance alignment. The gate MUST reject (high NR makes each
        // match contribute less; 5 matches alone shouldn't clear).
        let lo = lo_for_correct_wcs(5, 2.0, 500_000, 2.4e7);
        let cfg = BayesianGateConfig::default();
        let dec = accept_bayesian(lo, 5, 1.4, 2.0, &cfg);
        assert!(matches!(dec, AcceptanceDecision::Rejected(_)),
            "5 inliers in 500k-cat dense field MUST be rejected (false-positive guard); got log_odds={lo:.2} decision={dec:?}");
    }

    #[test]
    fn per_focal_length_match_count_sweep() {
        // Sweep: in every FL regime, more matches → higher log_odds (monotonic).
        // `typical_n_match_for_correct_wcs` is what the solver expects when
        // the WCS is right; this number scales with FL because:
        //   - Wide field (50mm MW): catalog has 500k stars, image caps at 600,
        //     ~500+ inliers when WCS is correct (most detections match a
        //     catalog star). The high NR also makes each match worth less
        //     log_odds, so we NEED hundreds of matches to clear — which we
        //     also HAVE.
        //   - Narrow field: very few catalog stars but a high fraction match.
        //
        // The "typical" column below reflects what realistically passes
        // through the corpus and the user's frames.
        let scenarios: &[(&str, usize, f64, f64, usize)] = &[
            // (label,    NR,        A,     σ,   typical_matches_for_correct_wcs)
            ("50mm MW",   500_000,   2.4e7, 2.0, 500),
            ("50mm hi-l", 30_000,    2.4e7, 2.0, 200),
            ("200mm",     50_000,    1.5e7, 1.5, 200),
            ("500mm",     5_000,     1.5e7, 1.5, 100),
            ("2000mm",    200,       1.2e7, 1.5, 30),
            ("4000mm",    15,        1.2e7, 1.0, 6),
        ];
        for (label, nr, area, sigma, typical_n) in scenarios {
            let lo_1  = lo_for_correct_wcs(1,  *sigma, *nr, *area);
            let lo_5  = lo_for_correct_wcs(5,  *sigma, *nr, *area);
            let lo_typ = lo_for_correct_wcs(*typical_n, *sigma, *nr, *area);
            assert!(lo_5 > lo_1, "{label}: 5 matches should exceed 1");
            assert!(lo_typ > lo_5, "{label}: typical {typical_n} matches should exceed 5");
            assert!(lo_typ > DEFAULT_LOG_ODDS_MIN,
                "{label}: typical {typical_n} matches must clear log(1e6)=13.8; got {lo_typ:.2}");
        }
    }

    #[test]
    fn gate_requires_more_matches_in_denser_fields() {
        // Empirical property: for the same `n_matches`, the gate gets
        // STRICTER as catalog density increases. This is the explicit
        // anti-false-positive design — dense fields need more evidence.
        let nr_sparse  = 100;
        let nr_dense   = 500_000;
        let lo_sparse  = lo_for_correct_wcs(20, 1.5, nr_sparse, 1.2e7);
        let lo_dense   = lo_for_correct_wcs(20, 1.5, nr_dense,  1.2e7);
        assert!(lo_sparse > lo_dense,
            "sparse should clear more easily than dense for same #matches: sparse={lo_sparse:.2}, dense={lo_dense:.2}");
        // And specifically: 20 matches passes in sparse, fails in dense.
        assert!(lo_sparse > DEFAULT_LOG_ODDS_MIN, "20 matches must pass in sparse field");
        assert!(lo_dense < DEFAULT_LOG_ODDS_MIN, "20 matches must FAIL in dense field (anti-FP guard)");
    }
}
