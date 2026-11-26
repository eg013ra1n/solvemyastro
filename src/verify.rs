//! Bayesian log-odds verification: a foreground-vs-background likelihood ratio, the
//! verification of a hypothesis as a Bayesian decision ([1] §2.4).
//!
//! [1] D. Lang, D. W. Hogg, K. Mierle, M. Blanton, S. Roweis, The Astronomical Journal
//! 139, 1782-1800 (2010), arXiv:0910.2233, §2.4.
//!
//! Two uses live here. [`log_odds`] scores the matched stars of a seed (the seed
//! pre-filter at ln 10^6). [`scan_log10_odds`] is the decision: it scores every image
//! star of the frame against the reference list, so a star without a counterpart costs
//! `ln d`, and its bar [`ACCEPT_LOG10_ODDS`] rises with the hypotheses tried
//! ([`required_log10_odds`]).
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
//! Formula (foreground-vs-background log-likelihood ratio over the matched stars):
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

/// Default threshold of the seed pre-filter: `log(1e6) ≈ 13.815`, odds 1,000,000:1
/// in favour of the WCS being correct ([1], §2.4).
pub const DEFAULT_LOG_ODDS_MIN: f64 = 13.815_510_557_964_274; // log(1e6)

/// Default distractor fraction `d` of [1], §2.4's mixture: the share of image
/// stars with no catalog counterpart (hot pixels, faint stars below the
/// catalog, artefacts). A catalog star with no detection costs nothing.
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

/// The decision's bar: posterior odds 10^3 over a prior of 10^-6 ([1], §2.4).
pub const ACCEPT_LOG10_ODDS: f64 = 9.0;

/// Beyond this many σ a test star has no counterpart and adds `ln d`.
pub const SCAN_CUTOFF_SIGMA: f64 = 3.0;

/// Hypotheses the prior's budget covers; beyond it the bar rises
/// ([`required_log10_odds`]).
pub const HYPOTHESIS_BUDGET: f64 = 1e6;

/// What the decision reads: every image star, the hypothesis's own stars among
/// them, and the reference list on the final WCS.
pub struct ScanInput<'a> {
    /// Every image star in SNR order, highest first (pixels).
    pub image_stars: &'a [(f64, f64)],
    /// Indices into `image_stars`: the stars the hypothesis was made from.
    pub own_stars: &'a [usize],
    /// R: the in-frame verification stars on the final WCS (pixels). Its length
    /// is `N_R`; the function never filters it.
    pub references: &'a [(f64, f64)],
    /// σ: the matching tolerance, never fitted.
    pub sigma_px: f64,
    /// `A = W × H` in px².
    pub area_px2: f64,
    /// d: the distractor fraction.
    pub distractor: f64,
}

/// The decision's score and the count it made.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanResult {
    /// `max over n ≥ 1 of L_n / ln 10`; `-inf` when degenerate.
    pub log10_odds: f64,
    /// n at the maximum, 1-based over the scanned stars; 0 when none.
    pub peak_rank: usize,
    /// `|T|`: `image_stars` minus the own stars.
    pub test_stars: usize,
    /// Scanned stars whose claimed reference lies within 1σ.
    pub scan_matched: usize,
    /// Own stars within 1σ of their claimed reference, plus `scan_matched`.
    pub matched_stars: usize,
}

/// The references bucketed in cells of `2 × cutoff` (one cell is as wide as a
/// star's search disc, so a disc meets at most 2 × 2 cells), sorted by cell and
/// then by index, so a cell's references come in ascending order.
struct RefGrid<'a> {
    refs: &'a [(f64, f64)],
    cell: f64,
    cells: Vec<((i64, i64), u32)>,
    claimed: Vec<bool>,
}

impl<'a> RefGrid<'a> {
    fn new(refs: &'a [(f64, f64)], cell: f64) -> Self {
        let mut cells = Vec::with_capacity(refs.len());
        for (i, &(x, y)) in refs.iter().enumerate() {
            // A reference without a finite place can explain no star.
            if let Some(key) = Self::key(x, y, cell) {
                cells.push((key, i as u32));
            }
        }
        cells.sort_unstable();
        Self { refs, cell, cells, claimed: vec![false; refs.len()] }
    }

    fn key(x: f64, y: f64, cell: f64) -> Option<(i64, i64)> {
        let (cx, cy) = ((x / cell).floor(), (y / cell).floor());
        (cx.is_finite() && cy.is_finite() && cx.abs() < 4e18 && cy.abs() < 4e18).then_some((cx as i64, cy as i64))
    }

    /// The nearest unclaimed reference within `cutoff` of `(x, y)` (a tie goes to
    /// the lower index), claimed; `(index, r²)`.
    fn claim_nearest(&mut self, x: f64, y: f64, cutoff: f64) -> Option<(usize, f64)> {
        let cut2 = cutoff * cutoff;
        // The cell window is padded a hair so rounding at a cell edge never drops
        // a reference that the exact distance test would keep.
        let reach = cutoff * (1.0 + 1e-9);
        let lo = Self::key(x - reach, y - reach, self.cell)?;
        let hi = Self::key(x + reach, y + reach, self.cell)?;
        let mut best: Option<(f64, usize)> = None;
        for cx in lo.0..=hi.0 {
            for cy in lo.1..=hi.1 {
                let start = self.cells.partition_point(|e| e.0 < (cx, cy));
                for &(k, i) in &self.cells[start..] {
                    if k != (cx, cy) {
                        break;
                    }
                    let i = i as usize;
                    if self.claimed[i] {
                        continue;
                    }
                    let (dx, dy) = (self.refs[i].0 - x, self.refs[i].1 - y);
                    let r2 = dx * dx + dy * dy;
                    if r2 <= cut2 && best.is_none_or(|(b2, bi)| (r2, i) < (b2, bi)) {
                        best = Some((r2, i));
                    }
                }
            }
        }
        let (r2, i) = best?;
        self.claimed[i] = true;
        Some((i, r2))
    }
}

/// The decision ([1], §2.4): the best prefix of the image stars, scored as a
/// foreground-versus-background likelihood ratio.
///
/// Per test star `ln(p_F/p_B) = ln(d + (1 - d) · A / (2π σ² N_R) · exp(-r²/2σ²))`
/// with `r` the distance to the nearest unclaimed reference; beyond `3σ` the
/// star adds `ln d`. A reference explains at most one star. The hypothesis's own
/// stars claim their references first and add no term; the rest are scanned in
/// SNR order and `log10_odds` is the highest running sum over `n ≥ 1`, so junk
/// at the faint end never lowers it. Degenerate input (`N_R = 0`, `σ ≤ 0`,
/// `A ≤ 0`, `d` outside `(0, 1)`, no scanned star) scores `-inf`.
pub fn scan_log10_odds(input: &ScanInput<'_>) -> ScanResult {
    let n_img = input.image_stars.len();
    let mut is_own = vec![false; n_img];
    for &i in input.own_stars {
        if let Some(f) = is_own.get_mut(i) {
            *f = true;
        }
    }
    let test_stars = is_own.iter().filter(|&&o| !o).count();
    let degenerate = ScanResult {
        log10_odds: f64::NEG_INFINITY,
        peak_rank: 0,
        test_stars,
        scan_matched: 0,
        matched_stars: 0,
    };
    let (sigma, d) = (input.sigma_px, input.distractor);
    if input.references.is_empty()
        || !(sigma > 0.0 && sigma.is_finite())
        || !(input.area_px2 > 0.0 && input.area_px2.is_finite())
        || !(d > 0.0 && d < 1.0)
        || test_stars == 0
    {
        return degenerate;
    }
    let cutoff = SCAN_CUTOFF_SIGMA * sigma;
    let c = (1.0 - d) * input.area_px2 / (2.0 * std::f64::consts::PI * sigma * sigma * input.references.len() as f64);
    let (one_sigma2, two_sigma2) = (sigma * sigma, 2.0 * sigma * sigma);
    let mut grid = RefGrid::new(input.references, 2.0 * cutoff);

    // Own stars, ascending, claim their references and add no term.
    let mut own_matched = 0usize;
    for (i, _) in is_own.iter().enumerate().filter(|(_, &o)| o) {
        let (x, y) = input.image_stars[i];
        if let Some((_, r2)) = grid.claim_nearest(x, y, cutoff) {
            if r2 <= one_sigma2 {
                own_matched += 1;
            }
        }
    }

    let ln_d = d.ln();
    let mut sum = 0.0f64;
    let mut best = f64::NEG_INFINITY;
    let mut peak_rank = 0usize;
    let mut scan_matched = 0usize;
    let mut n = 0usize;
    for (i, &(x, y)) in input.image_stars.iter().enumerate() {
        if is_own[i] {
            continue;
        }
        n += 1;
        sum += match grid.claim_nearest(x, y, cutoff) {
            Some((_, r2)) => {
                if r2 <= one_sigma2 {
                    scan_matched += 1;
                }
                (d + c * (-r2 / two_sigma2).exp()).ln()
            }
            None => ln_d,
        };
        let odds = sum / std::f64::consts::LN_10;
        if odds > best {
            best = odds;
            peak_rank = n;
        }
    }
    ScanResult {
        log10_odds: best,
        peak_rank,
        test_stars,
        scan_matched,
        matched_stars: own_matched + scan_matched,
    }
}

/// `base + max(0, log10(hypotheses / HYPOTHESIS_BUDGET))`: the bar of the
/// decision after `hypotheses` hypotheses were verified in one solve.
pub fn required_log10_odds(base: f64, hypotheses: usize) -> f64 {
    base + (hypotheses as f64 / HYPOTHESIS_BUDGET).log10().max(0.0)
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

    // ---- the decision: scan_log10_odds ----

    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256StarStar;

    const D: f64 = 0.25;

    fn c_of(sigma: f64, area: f64, n_r: usize) -> f64 {
        (1.0 - D) * area / (2.0 * std::f64::consts::PI * sigma * sigma * n_r as f64)
    }

    /// 50 references on a 100 px lattice (10 per row).
    fn lattice_refs() -> Vec<(f64, f64)> {
        (0..50).map(|i| (100.0 + 100.0 * (i % 10) as f64, 100.0 + 100.0 * (i / 10) as f64)).collect()
    }

    fn scan(stars: &[(f64, f64)], own: &[usize], refs: &[(f64, f64)], sigma: f64, area: f64) -> ScanResult {
        scan_log10_odds(&ScanInput {
            image_stars: stars,
            own_stars: own,
            references: refs,
            sigma_px: sigma,
            area_px2: area,
            distractor: D,
        })
    }

    fn uniform(rng: &mut Xoshiro256StarStar, n: usize, w: f64, h: f64) -> Vec<(f64, f64)> {
        (0..n).map(|_| (rng.random_range(0.0..w), rng.random_range(0.0..h))).collect()
    }

    #[test]
    fn degenerate_inputs_are_never_confident() {
        let refs = lattice_refs();
        let stars = refs[..10].to_vec();
        let never = |r: ScanResult| r.log10_odds == f64::NEG_INFINITY && r.peak_rank == 0;
        assert!(never(scan(&stars, &[], &[], 1.5, 6e6)), "N_R = 0");
        assert!(never(scan(&stars, &[], &refs, 0.0, 6e6)), "sigma 0");
        assert!(never(scan(&stars, &[], &refs, 1.5, 0.0)), "area 0");
        assert!(never(scan(&[], &[], &refs, 1.5, 6e6)), "no image star");
        assert!(never(scan(&stars, &(0..10).collect::<Vec<_>>(), &refs, 1.5, 6e6)), "every star own");
        assert!(never(scan(&stars, &[], &refs, f64::NAN, 6e6)), "sigma NaN");
        for d in [0.0, 1.0, -0.1, f64::NAN] {
            let r = scan_log10_odds(&ScanInput {
                image_stars: &stars,
                own_stars: &[],
                references: &refs,
                sigma_px: 1.5,
                area_px2: 6e6,
                distractor: d,
            });
            assert!(never(r), "distractor {d}");
        }
    }

    #[test]
    fn an_unmatched_image_star_costs_ln_d() {
        let refs = lattice_refs();
        let on: Vec<(f64, f64)> = refs[..10].to_vec();
        let base = scan(&on, &[], &refs, 1.5, 6e6);
        let mut with_far = vec![(50.0, 50.0)];
        with_far.extend_from_slice(&on);
        let far = scan(&with_far, &[], &refs, 1.5, 6e6);
        assert!((far.log10_odds - base.log10_odds - D.log10()).abs() < 1e-12);
        assert_eq!(far.peak_rank, base.peak_rank + 1);
        assert_eq!(base.peak_rank, 10);
        assert_eq!(base.scan_matched, 10);
    }

    #[test]
    fn lost_catalog_stars_leave_the_odds_unchanged() {
        let a = lattice_refs();
        // The first 10 stay; the other 40 move 37 px right and 23 px down, far
        // (more than 3 sigma) from every test star in row 0.
        let b: Vec<(f64, f64)> = a.iter().enumerate().map(|(i, &(x, y))| if i < 10 { (x, y) } else { (x + 37.0, y + 23.0) }).collect();
        let stars = a[..10].to_vec();
        assert_eq!(scan(&stars, &[], &a, 1.5, 6e6), scan(&stars, &[], &b, 1.5, 6e6));
    }

    #[test]
    fn n_r_is_the_length_of_the_list_passed() {
        let refs = lattice_refs();
        let mut padded = refs.clone();
        padded.extend((0..50).map(|i| (-1000.0 - i as f64 * 50.0, -1000.0)));
        let star = [refs[0]];
        let r50 = scan(&star, &[], &refs, 1.5, 6e6);
        let r100 = scan(&star, &[], &padded, 1.5, 6e6);
        assert!((r50.log10_odds - (D + c_of(1.5, 6e6, 50)).log10()).abs() < 1e-12);
        assert!((r100.log10_odds - (D + c_of(1.5, 6e6, 100)).log10()).abs() < 1e-12);
    }

    #[test]
    fn a_reference_explains_one_test_star() {
        let refs = lattice_refs();
        let r = scan(&[refs[0], refs[0]], &[], &refs, 1.5, 6e6);
        assert_eq!(r.scan_matched, 1);
        assert_eq!(r.peak_rank, 1);
        assert!((r.log10_odds - (D + c_of(1.5, 6e6, 50)).log10()).abs() < 1e-12);
    }

    #[test]
    fn the_cut_offs_are_inclusive_at_one_and_three_sigma() {
        let refs = lattice_refs();
        let (sigma, c) = (1.5, c_of(1.5, 6e6, 50));
        let at = |dx: f64| scan(&[(refs[0].0 + dx, refs[0].1)], &[], &refs, sigma, 6e6);
        let one = at(sigma);
        assert_eq!(one.scan_matched, 1, "exactly sigma counts");
        assert!((one.log10_odds - (D + c * (-0.5f64).exp()).log10()).abs() < 1e-12);
        let three = at(3.0 * sigma);
        assert_eq!(three.scan_matched, 0, "3 sigma claims but is not counted");
        assert!((three.log10_odds - (D + c * (-4.5f64).exp()).log10()).abs() < 1e-12, "exactly 3 sigma claims");
        let beyond = at(3.0 * sigma * (1.0 + 1e-9));
        assert_eq!(beyond.scan_matched, 0);
        assert!((beyond.log10_odds - D.log10()).abs() < 1e-12, "beyond 3 sigma adds ln d");
    }

    #[test]
    fn own_stars_claim_their_references_first() {
        let refs = lattice_refs();
        let r = scan(&[refs[0], refs[0]], &[0], &refs, 1.5, 6e6);
        assert_eq!(r.test_stars, 1);
        assert_eq!(r.scan_matched, 0);
        assert_eq!(r.matched_stars, 1);
        assert_eq!(r.peak_rank, 1);
        assert!((r.log10_odds - D.log10()).abs() < 1e-12, "the scanned star finds its reference taken");
    }

    #[test]
    fn an_own_star_claims_before_any_scanned_star_whatever_its_place() {
        // The own star is behind the scanned one in the SNR order and still claims first.
        let refs = lattice_refs();
        let r = scan(&[refs[0], refs[0]], &[1], &refs, 1.5, 6e6);
        assert_eq!(r.test_stars, 1);
        assert_eq!(r.scan_matched, 0);
        assert_eq!(r.matched_stars, 1);
        assert!((r.log10_odds - D.log10()).abs() < 1e-12, "the scanned star finds its reference taken");
    }

    #[test]
    fn own_stars_claim_in_ascending_index_order() {
        // Star 0 (2 sigma off the reference) claims it before star 2 (on it) does.
        let (sigma, refs) = (1.5, lattice_refs());
        let stars = [(refs[0].0 + 2.0 * sigma, refs[0].1), (5000.0, 3000.0), refs[0]];
        let r = scan(&stars, &[2, 0], &refs, sigma, 6e6);
        assert_eq!(r.matched_stars, 0, "star 0 claims at 2 sigma, star 2 finds nothing");
        assert_eq!(r.scan_matched, 0);
        assert_eq!(r.test_stars, 1);
    }

    /// The decision's scan the plain way: every reference against every star.
    fn plain_scan(stars: &[(f64, f64)], own: &[usize], refs: &[(f64, f64)], sigma: f64, area: f64) -> ScanResult {
        let cut2 = (3.0 * sigma) * (3.0 * sigma);
        let c = c_of(sigma, area, refs.len());
        let mut claimed = vec![false; refs.len()];
        let mut nearest = |p: (f64, f64)| -> Option<f64> {
            let mut best: Option<(f64, usize)> = None;
            for (i, r) in refs.iter().enumerate() {
                let r2 = (r.0 - p.0) * (r.0 - p.0) + (r.1 - p.1) * (r.1 - p.1);
                if !claimed[i] && r2 <= cut2 && best.is_none_or(|b| (r2, i) < b) {
                    best = Some((r2, i));
                }
            }
            let (r2, i) = best?;
            claimed[i] = true;
            Some(r2)
        };
        let mut own_sorted: Vec<usize> = own.to_vec();
        own_sorted.sort_unstable();
        own_sorted.dedup();
        let mut own_matched = 0;
        for &i in &own_sorted {
            if nearest(stars[i]).is_some_and(|r2| r2 <= sigma * sigma) {
                own_matched += 1;
            }
        }
        let (mut sum, mut best, mut peak, mut n, mut scan_matched) = (0.0f64, f64::NEG_INFINITY, 0, 0, 0);
        for (i, &p) in stars.iter().enumerate() {
            if own_sorted.contains(&i) {
                continue;
            }
            n += 1;
            sum += match nearest(p) {
                Some(r2) => {
                    scan_matched += usize::from(r2 <= sigma * sigma);
                    (D + c * (-r2 / (2.0 * sigma * sigma)).exp()).ln()
                }
                None => D.ln(),
            };
            if sum / std::f64::consts::LN_10 > best {
                best = sum / std::f64::consts::LN_10;
                peak = n;
            }
        }
        ScanResult {
            log10_odds: best,
            peak_rank: peak,
            test_stars: stars.len() - own_sorted.len(),
            scan_matched,
            matched_stars: own_matched + scan_matched,
        }
    }

    #[test]
    fn the_grid_agrees_with_a_plain_nearest_unclaimed_scan() {
        // Negative coordinates, points on multiples of 6 sigma (the cell edges), and stars
        // within a hair of a reference; every output equal, not merely close.
        let sigma = 1.5;
        let cell = 6.0 * sigma;
        for seed in 0..40u64 {
            let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
            let n_ref = 30 + 10 * (seed as usize % 6);
            let mut refs: Vec<(f64, f64)> = (0..n_ref)
                .map(|_| (rng.random_range(-60.0..60.0), rng.random_range(-60.0..60.0)))
                .collect();
            for k in 0..8 {
                // References and stars on the cell edges, negative ones included.
                refs.push((cell * (k as f64 - 4.0), cell * ((k * 3 % 8) as f64 - 4.0)));
            }
            let mut stars: Vec<(f64, f64)> = Vec::new();
            for r in &refs {
                stars.push((r.0 + rng.random_range(-4.0..4.0) * sigma, r.1 + rng.random_range(-4.0..4.0) * sigma));
            }
            for k in 0..30 {
                stars.push((cell * (k % 11) as f64 - 5.0 * cell, cell * (k % 7) as f64 - 3.0 * cell));
                stars.push((rng.random_range(-70.0..70.0), rng.random_range(-70.0..70.0)));
            }
            let n = stars.len();
            for k in (0..n).rev().step_by(5) {
                let j = rng.random_range(0..=k);
                stars.swap(k, j);
            }
            let own: Vec<usize> = (0..6).map(|_| rng.random_range(0..n)).collect();
            let want = plain_scan(&stars, &own, &refs, sigma, 14400.0);
            let got = scan(&stars, &own, &refs, sigma, 14400.0);
            assert_eq!(got, want, "seed {seed}");
        }
    }

    #[test]
    fn the_scan_peaks_at_the_running_maximum() {
        let refs = lattice_refs();
        let mut stars = vec![refs[0], refs[1]];
        stars.extend((0..20).map(|i| (5000.0 + 10.0 * i as f64, 3000.0)));
        let r = scan(&stars, &[], &refs, 1.5, 6e6);
        assert_eq!(r.peak_rank, 2);
        assert!((r.log10_odds - 2.0 * (D + c_of(1.5, 6e6, 50)).log10()).abs() < 1e-12);
        assert_eq!(r.test_stars, 22);
    }

    #[test]
    fn the_bar_rises_beyond_a_million_hypotheses() {
        assert_eq!(required_log10_odds(9.0, 1), 9.0);
        assert_eq!(required_log10_odds(9.0, 0), 9.0);
        assert_eq!(required_log10_odds(9.0, 1_000_000), 9.0);
        assert!((required_log10_odds(9.0, 10_000_000) - 10.0).abs() < 1e-12);
        assert_eq!(required_log10_odds(f64::INFINITY, 5), f64::INFINITY);
    }

    /// Reference and test stars uniform in the frame, no own stars: how many of
    /// `n` hypotheses reach 10^3 to 1.
    fn null_hits(sigma: f64, n_r: usize, (w, h): (f64, f64), n_stars: usize, n: u64) -> usize {
        (0..n)
            .filter(|&k| {
                let mut rng = Xoshiro256StarStar::seed_from_u64(k);
                let refs = uniform(&mut rng, n_r, w, h);
                let stars = uniform(&mut rng, n_stars, w, h);
                scan(&stars, &[], &refs, sigma, w * h).log10_odds >= 3.0
            })
            .count()
    }

    #[test]
    fn the_uniform_null_rarely_reaches_a_thousand_to_one() {
        // Ville's bound: at most 1e-3 of the hypotheses, so 10 of 10^4; Poisson q95 is 15.
        let wide = null_hits(13.8, 61, (6000.0, 4000.0), 600, 10_000);
        let dense = null_hits(1.6, 476, (6252.0, 4176.0), 584, 10_000);
        assert!(wide <= 15, "the documented false-solve geometry: {wide} of 10000");
        assert!(dense <= 15, "frame 28940's geometry: {dense} of 10000");
    }

    #[test]
    fn the_per_star_ratio_averages_below_one() {
        // One reference, its 3 sigma disc wholly inside a 16 x 16 frame; one test star
        // on a lattice (step 0.05 sigma). The mean per-star ratio is the integral of
        // d + c exp(-r^2/2s^2) over the disc plus d outside it, over the frame:
        // d + (1 - d)(1 - e^-4.5), the normalisation the bound rests on.
        let (sigma, side): (f64, f64) = (1.0, 16.0);
        let refs = [(8.0, 8.0)];
        let step = 0.05 * sigma;
        let n = (side / step).round() as usize;
        let mut total = 0.0;
        for i in 0..n {
            for j in 0..n {
                let star = [((i as f64 + 0.5) * step, (j as f64 + 0.5) * step)];
                total += 10f64.powf(scan(&star, &[], &refs, sigma, side * side).log10_odds);
            }
        }
        let mean = total / (n * n) as f64;
        let want = D + (1.0 - D) * (1.0 - (-4.5f64).exp());
        assert!(mean < 1.0, "the per-star ratio averages below one: {mean}");
        assert!((mean - want).abs() < 1e-3, "mean {mean} vs {want}");
    }

    #[test]
    fn a_true_alignment_clears_the_bar_by_far() {
        let (sigma, w, h) = (1.6, 6252.0, 4176.0);
        let mut rng = Xoshiro256StarStar::seed_from_u64(11);
        let refs = uniform(&mut rng, 476, w, h);
        let mut stars = Vec::new();
        for (i, r) in refs.iter().take(120).enumerate() {
            stars.push((r.0 + 0.5 * sigma, r.1));
            for _ in 0..3 + usize::from(i < 104) {
                stars.push((rng.random_range(0.0..w), rng.random_range(0.0..h)));
            }
        }
        assert_eq!(stars.len(), 584);
        let r = scan(&stars, &[], &refs, sigma, w * h);
        assert!(r.log10_odds >= 100.0, "got {}", r.log10_odds);
    }

    #[test]
    fn the_recorded_false_solve_scores_under_the_bar() {
        // The documented false solve: sigma 13.8 px, 61 references, 6000 x 4000, 600 stars,
        // the quad's four and two more on references. About 5 decades, never 9.
        let (sigma, w, h) = (13.8, 6000.0, 4000.0);
        let mut rng = Xoshiro256StarStar::seed_from_u64(5);
        let refs = uniform(&mut rng, 61, w, h);
        let mut stars: Vec<(f64, f64)> = refs[..6].to_vec();
        stars.extend(uniform(&mut rng, 594, w, h));
        let r = scan(&stars, &[0, 1, 2, 3], &refs, sigma, w * h);
        assert!(r.log10_odds > 3.0 && r.log10_odds < 9.0, "got {}", r.log10_odds);
    }
}
