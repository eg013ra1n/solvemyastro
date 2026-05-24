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
}
