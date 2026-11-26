//! Per-solve diagnostic trace.
//!
//! Each `solve()` builds a `SolveTrace` recording the funnel:
//! `detect → select → img_quads → for each rung { cone → cap → cat_quads → match → affine → verify }`.
//! A `Display` impl prints a compact summary at solve end (always) plus a
//! detailed post-mortem when the solve fails. Set `SOLVEMYASTRO_DEBUG=1` to
//! also dump every individual cell rather than just per-rung aggregates.

use std::fmt;

/// Honour `SOLVEMYASTRO_DEBUG=1` for verbose per-cell dumps.
pub fn debug_enabled() -> bool {
    std::env::var("SOLVEMYASTRO_DEBUG")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Honour `SOLVEMYASTRO_DUMP_RESIDUALS=1` — emit a per-star residual vector
/// field (catalog-predicted vs detected pixel positions) for each solved
/// frame. A pre-flight diagnostic to decide whether the precision ceiling is
/// **centroid-bound** (residuals scatter randomly) or **distortion-bound**
/// (residuals show radial / quadrant structure across the image plane).
pub fn dump_residuals_enabled() -> bool {
    std::env::var("SOLVEMYASTRO_DUMP_RESIDUALS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Write the per-star residual vector field for `frame` to
/// `<SOLVEMYASTRO_DUMP_DIR or .>/residual-vector-<frame>.csv`.
///
/// `pairs` are `((predicted_px, predicted_py), (detected_px, detected_py))`
/// exactly as produced by `count_inliers`. Emitted columns:
/// `det_x,det_y,res_x,res_y,res_mag` where `res = detected - predicted` — the
/// vector from where the catalog says the star is to where it was measured.
/// Plot `(det_x, det_y)` coloured/quivered by `(res_x, res_y)` to read the
/// spatial structure. Best-effort: I/O errors log to stderr and never
/// propagate (this is a diagnostic, not a solve dependency).
pub fn dump_residual_field(frame: &str, pairs: &[((f64, f64), (f64, f64))]) {
    use std::io::Write;
    let dir = std::env::var("SOLVEMYASTRO_DUMP_DIR").unwrap_or_else(|_| ".".to_string());
    let safe: String = frame
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let path = std::path::Path::new(&dir).join(format!("residual-vector-{safe}.csv"));
    let mut out = String::with_capacity(pairs.len() * 48 + 64);
    out.push_str("det_x,det_y,res_x,res_y,res_mag\n");
    for &((px, py), (dx, dy)) in pairs {
        let (rx, ry) = (dx - px, dy - py);
        let mag = (rx * rx + ry * ry).sqrt();
        out.push_str(&format!("{dx:.4},{dy:.4},{rx:.5},{ry:.5},{mag:.5}\n"));
    }
    match std::fs::File::create(&path).and_then(|mut f| f.write_all(out.as_bytes())) {
        Ok(()) => tracing::debug!(
            count = pairs.len(),
            path = %path.display(),
            "residual field written"
        ),
        Err(e) => tracing::warn!(
            path = %path.display(),
            error = %e,
            "failed to write residual field"
        ),
    }
}

#[derive(Debug, Clone, Default)]
pub struct SolveTrace {
    pub frame: String,
    pub n_detected: usize,
    pub n_selected: usize,
    pub snr_floor: f32,
    pub n_image_quads: usize,
    pub group_size: usize,
    pub hint_center: Option<(f64, f64)>,
    pub hint_radius_deg: f64,
    pub passes: Vec<PassTrace>,
    pub outcome: TraceOutcome,
    pub total_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PassTrace {
    pub balance: bool,
    pub rungs: Vec<RungTrace>,
}

impl PassTrace {
    pub fn new(balance: bool) -> Self {
        Self { balance, rungs: Vec::new() }
    }
}

#[derive(Debug, Clone)]
pub struct RungTrace {
    pub idx: usize,
    pub total: usize,
    pub fov_deg: f64,
    pub n_cells: usize,
    pub cells: Vec<CellTrace>,
    pub candidates_emitted: usize,
    pub verifies: Vec<VerifyTrace>,
    pub best_inliers_seen: usize,
}

impl RungTrace {
    pub fn new(idx: usize, total: usize, fov_deg: f64, n_cells: usize) -> Self {
        Self {
            idx, total, fov_deg, n_cells,
            cells: Vec::with_capacity(n_cells),
            candidates_emitted: 0,
            verifies: Vec::new(),
            best_inliers_seen: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CellTrace {
    pub ra: f64,
    pub dec: f64,
    pub cat_raw: usize,
    pub cat_after_cap: usize,
    pub cat_quads: usize,
    pub match_raw: usize,
    pub match_filtered: usize,
    pub seed: Option<SeedInfo>,
    pub stop: CellStop,
}

#[derive(Debug, Clone)]
pub struct SeedInfo {
    pub ra: f64,
    pub dec: f64,
    pub scale_arcsec_per_px: f64,
    pub dist_from_hint_deg: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CellStop {
    #[default]
    NotStarted,
    ConeTooSparse,
    CatQuadsEmpty,
    MatchBelowFloor,
    AffineFailed,
    ScaleSanityFailed,
    SeedEmitted,
}

impl CellStop {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NotStarted => "not_started",
            Self::ConeTooSparse => "cone_too_sparse",
            Self::CatQuadsEmpty => "cat_quads_empty",
            Self::MatchBelowFloor => "match_below_floor",
            Self::AffineFailed => "affine_failed",
            Self::ScaleSanityFailed => "scale_sanity_failed",
            Self::SeedEmitted => "seed_emitted",
        }
    }
}

#[derive(Debug, Clone)]
pub struct VerifyTrace {
    pub seed_ra: f64,
    pub seed_dec: f64,
    pub scale_arcsec_per_px: f64,
    pub cat_in_fov: usize,
    pub seed_inliers: usize,
    pub required: usize,
    /// RMS of the seed-stage inliers, in pixels. Recorded even when the trial
    /// is rejected — a tight seed_rms with low inliers means the gate threw
    /// away a valid near-miss; a loose seed_rms means the matches were noise.
    pub seed_rms_px: f64,
    /// Bayesian log-odds for the seed-stage inliers (the standard Bayesian verification formula).
    /// `f64::NEG_INFINITY` when the gate was not evaluated (e.g. legacy
    /// `use_bayesian_verify=false` path or pre-verify reject).
    pub log_odds: f64,
    pub final_inliers: usize,
    pub final_rms_px: f64,
    /// Angular distance from this trial's seed center to a known-correct WCS
    /// (e.g. an externally-verified solve from a comparison solver), in degrees.
    /// `None` if no known truth is registered for this frame. See
    /// [`known_truth_for`].
    pub dist_from_truth_deg: Option<f64>,
    /// Ratio of this trial's pixel scale to the known truth scale.
    /// `1.0` is perfect; values like 0.5 or 2.0 indicate a wrong-scale seed.
    /// `None` when no known truth is registered.
    pub scale_ratio_to_truth: Option<f64>,
    pub accepted: bool,
    pub reject_reason: Option<String>,
    /// The search that produced this candidate.
    pub path: SearchPath,
    /// True for a recentred re-fit of a spiral seed.
    pub recentered: bool,
    /// Hypotheses entering verification so far in this solve, this one
    /// included, across every pass and fallback step.
    pub hypotheses_so_far: usize,
    /// Image stars that chose this candidate (`own_stars.len()`).
    pub own_star_count: usize,
    /// Image stars of the frame the candidate was verified against.
    pub image_star_count: usize,
    /// Of the seed pre-filter's inliers (`seed_inliers`), those whose image
    /// star is not one of the candidate's own stars.
    pub inliers_outside_own: usize,
    /// The decision's σ in pixels (the matching tolerance of this candidate's
    /// scale); `0.0` when the verification cone was too sparse (set before the
    /// seed pre-filter, so every candidate past the cone carries it).
    pub tol_px: f64,
    /// `N_R`: verification stars that project into the frame on the final WCS.
    pub reference_stars: usize,
    /// The magnitude limit of the cone `reference_stars` came from (the ladder's level
    /// the frame's share reached the floor at, or the deepest); `0.0` when the
    /// decision was not reached.
    pub reference_mag_limit: f32,
    /// `|T|`: image stars minus the candidate's own stars.
    pub test_stars: usize,
    /// Scanned stars matched within 1σ of their claimed reference.
    pub scan_matched: usize,
    /// Own stars within 1σ plus `scan_matched` (the solution's `matched_stars`).
    pub matched_stars: usize,
    /// Scan rank (1-based) at which the running score peaked; 0 when none.
    pub peak_rank: usize,
    /// `log10` Bayes factor on the final WCS with the candidate's own stars;
    /// `None` when the decision was not reached.
    pub log10_odds: Option<f64>,
    /// The same with only the four stars of the first quad set aside.
    pub log10_odds_quad4: Option<f64>,
    /// The bar the candidate faced: `required_log10_odds(accept, hypotheses_so_far)`;
    /// `0.0` when the decision was not reached.
    pub required_log10_odds: f64,
    /// The final (SIP) WCS the decision scored; `None` when it was not reached.
    pub final_wcs: Option<crate::wcs::WcsSolution>,
}

/// Which search produced a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPath {
    Spiral,
    Index,
}

impl SearchPath {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Spiral => "spiral",
            Self::Index => "index",
        }
    }
}

/// A known-correct WCS for a specific frame, used to ground-truth-check the
/// solver's intermediate trial seeds. Populated by `known_truth_for` from
/// externally-verified solves (from another tool) so we can answer: "did
/// solvemyastro briefly find the right WCS and then reject it?"
#[derive(Debug, Clone, Copy)]
pub struct KnownTruth {
    pub ra: f64,
    pub dec: f64,
    pub scale_arcsec_per_px: f64,
    #[allow(dead_code)]
    pub rot_deg: f64,
}

/// Lookup table of known-correct WCS keyed by frame-name substring. Frames
/// not in this table return `None` and their VerifyTraces carry no truth
/// comparison — only present for specific failing frames where an external
/// solve has been captured.
pub fn known_truth_for(frame_name: &str) -> Option<KnownTruth> {
    if frame_name.contains("_0023.fits") {
        // A comparison solver solved this frame in 0.1s. Captured 2026-05-24.
        return Some(KnownTruth {
            ra: 334.0989556839,
            dec: 74.22973718397,
            scale_arcsec_per_px: 5.973,
            rot_deg: 121.4,
        });
    }
    None
}

#[derive(Debug, Clone, Default)]
pub enum TraceOutcome {
    #[default]
    Pending,
    Solved { ra: f64, dec: f64, inliers: usize, rms_px: f64 },
    Failed(String),
}

impl TraceOutcome {
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

/// Heuristic failure classification, computed from the trace contents.
#[derive(Debug, Clone, PartialEq)]
pub enum FailureClass {
    /// Solve succeeded.
    Solved,
    /// Detection returned < 4 stars (catastrophic — bad image).
    DetectTooFew,
    /// Built < 3 image quads.
    ImgQuadsEmpty,
    /// Every rung had match_raw_avg ≤ 3 — catalog density or descriptor
    /// mismatch.
    MatchStarved,
    /// Matches happened but no `fit_affine` ever succeeded.
    AffineNeverFit,
    /// Candidates emitted but no verify trace recorded — pre-verify reject
    /// (scale sanity, etc.).
    NoVerifyAttempted,
    /// Best verified inliers = 0 (WCS pointed at the wrong sky region entirely).
    VerifyZeroInliers,
    /// Best verified inliers > 0 but below the required floor.
    VerifyGap { best: usize, required: usize },
    /// A candidate reached the decision on its final WCS and scored below the
    /// bar; `best_log10_odds` is the highest finite score of the solve.
    NotConfident { best_log10_odds: f64 },
    /// Catch-all when nothing of the above fits.
    Unclassified,
}

impl FailureClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Solved => "SOLVED",
            Self::DetectTooFew => "DETECT_TOO_FEW",
            Self::ImgQuadsEmpty => "IMG_QUADS_EMPTY",
            Self::MatchStarved => "MATCH_STARVED",
            Self::AffineNeverFit => "AFFINE_NEVER_FIT",
            Self::NoVerifyAttempted => "NO_VERIFY_ATTEMPTED",
            Self::VerifyZeroInliers => "VERIFY_ZERO_INLIERS",
            Self::VerifyGap { .. } => "VERIFY_GAP",
            Self::NotConfident { .. } => "NOT_CONFIDENT",
            Self::Unclassified => "UNCLASSIFIED",
        }
    }
}

/// Structured failure returned by `solve()` so callers can render a precise
/// reason instead of a generic string. Carries the heuristic [`FailureClass`]
/// and the best partial attempt seen during the search. `Display` yields the
/// human `message`; callers can `downcast_ref::<SolveFailure>()` on the
/// returned `anyhow::Error` to inspect `class` / `best_attempt`.
#[derive(Debug, Clone)]
pub struct SolveFailure {
    pub class: FailureClass,
    pub best_attempt: Option<BestAttempt>,
    pub message: String,
    /// Hypotheses verified in the whole solve, every pass and fallback step.
    pub hypotheses_verified: usize,
    /// Candidates that passed the seed pre-filter (reached scoring).
    pub prefilter_passed: usize,
    /// The verify trace of every candidate of every step, in verification order.
    pub verifies: Vec<VerifyTrace>,
}

impl fmt::Display for SolveFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SolveFailure {}

/// The largest finite decision score among `verifies`, `None` when no
/// candidate reached the decision with a finite score. The one fold behind
/// [`FailureClass::NotConfident`] for a single trace and for a whole solve.
pub(crate) fn best_finite_log10_odds<'a>(
    verifies: impl IntoIterator<Item = &'a VerifyTrace>,
) -> Option<f64> {
    verifies
        .into_iter()
        .filter_map(|v| v.log10_odds)
        .filter(|o| o.is_finite())
        .fold(None, |acc: Option<f64>, o| Some(acc.map_or(o, |a| a.max(o))))
}

/// The failure message of a solve whose best candidate reached the decision and
/// scored below the bar (`FailureClass::NotConfident`).
pub(crate) fn chance_alignment_message(best_log10_odds: f64, required_log10_odds: f64) -> String {
    format!(
        "the best match could be a chance alignment (odds 10^{best_log10_odds:.1} to 1, \
         10^{required_log10_odds:.1} needed)"
    )
}

/// The bar the best-scoring candidate of `verifies` faced; `base` when none
/// reached the decision with a finite score.
pub(crate) fn best_decision_bar<'a>(
    verifies: impl IntoIterator<Item = &'a VerifyTrace>,
    base: f64,
) -> f64 {
    verifies
        .into_iter()
        .filter(|v| v.log10_odds.is_some_and(f64::is_finite))
        .fold(None, |acc: Option<&VerifyTrace>, v| match acc {
            Some(a) if a.log10_odds >= v.log10_odds => Some(a),
            _ => Some(v),
        })
        .map_or(base, |v| v.required_log10_odds)
}

impl SolveTrace {
    /// Every verify trace of the solve, in verification order.
    pub(crate) fn all_verifies(&self) -> Vec<&VerifyTrace> {
        self.passes.iter().flat_map(|p| p.rungs.iter()).flat_map(|r| r.verifies.iter()).collect()
    }

    pub fn classify(&self) -> FailureClass {
        if matches!(self.outcome, TraceOutcome::Solved { .. }) {
            return FailureClass::Solved;
        }
        if self.n_detected < 4 { return FailureClass::DetectTooFew; }
        if self.n_image_quads < 3 { return FailureClass::ImgQuadsEmpty; }

        let mut all_verifies: Vec<&VerifyTrace> = Vec::new();
        let mut any_candidate = false;
        let mut any_affine = false;
        let mut any_match = false;
        for p in &self.passes {
            for r in &p.rungs {
                if r.candidates_emitted > 0 { any_candidate = true; }
                for c in &r.cells {
                    if c.match_filtered >= 3 { any_match = true; }
                    if c.seed.is_some() { any_affine = true; }
                }
                all_verifies.extend(r.verifies.iter());
            }
        }
        if all_verifies.is_empty() {
            if any_candidate { return FailureClass::NoVerifyAttempted; }
            if any_affine { return FailureClass::NoVerifyAttempted; }
            if any_match { return FailureClass::AffineNeverFit; }
            return FailureClass::MatchStarved;
        }
        // A candidate that reached the decision and scored below the bar.
        if let Some(best_log10_odds) = best_finite_log10_odds(all_verifies.iter().copied()) {
            return FailureClass::NotConfident { best_log10_odds };
        }
        let (best_inl, best_req) = all_verifies.iter().fold((0usize, 0usize), |acc, v| {
            if v.seed_inliers > acc.0 { (v.seed_inliers, v.required) } else { acc }
        });
        if best_inl == 0 { return FailureClass::VerifyZeroInliers; }
        if best_inl < best_req { return FailureClass::VerifyGap { best: best_inl, required: best_req }; }
        FailureClass::Unclassified
    }

    /// Best-attempt summary across all passes/rungs — highest seed_inliers seen.
    pub fn best_attempt(&self) -> Option<BestAttempt> {
        let mut best: Option<BestAttempt> = None;
        for (pi, p) in self.passes.iter().enumerate() {
            for r in &p.rungs {
                for v in &r.verifies {
                    let ba = BestAttempt {
                        pass_balance: p.balance,
                        pass_idx: pi,
                        rung_idx: r.idx,
                        rung_total: r.total,
                        seed_ra: v.seed_ra,
                        seed_dec: v.seed_dec,
                        scale_arcsec_per_px: v.scale_arcsec_per_px,
                        cat_in_fov: v.cat_in_fov,
                        inliers: v.seed_inliers,
                        required: v.required,
                        rms_px: v.final_rms_px,
                        seed_rms_px: v.seed_rms_px,
                        log_odds: v.log_odds,
                        dist_from_truth_deg: v.dist_from_truth_deg,
                        scale_ratio_to_truth: v.scale_ratio_to_truth,
                    };
                    match &best {
                        None => best = Some(ba),
                        Some(cur) if v.seed_inliers > cur.inliers => best = Some(ba),
                        _ => {}
                    }
                }
            }
        }
        best
    }

    /// True if any registered known-truth WCS was within 0.1° of any trial
    /// seed. Indicates the solver found the right region but the verify
    /// gate rejected it — a strong signal that the gate is too strict.
    pub fn any_trial_near_truth(&self) -> Option<(f64, usize, usize, f64)> {
        let mut best: Option<(f64, usize, usize, f64)> = None;
        for p in &self.passes {
            for r in &p.rungs {
                for v in &r.verifies {
                    if let (Some(d), Some(s)) = (v.dist_from_truth_deg, v.scale_ratio_to_truth) {
                        let candidate = (d, v.seed_inliers, v.required, s);
                        match best {
                            None => best = Some(candidate),
                            Some((bd, _, _, _)) if d < bd => best = Some(candidate),
                            _ => {}
                        }
                    }
                }
            }
        }
        best
    }
}

#[derive(Debug, Clone)]
pub struct BestAttempt {
    pub pass_balance: bool,
    pub pass_idx: usize,
    pub rung_idx: usize,
    pub rung_total: usize,
    pub seed_ra: f64,
    pub seed_dec: f64,
    pub scale_arcsec_per_px: f64,
    pub cat_in_fov: usize,
    pub inliers: usize,
    pub required: usize,
    pub rms_px: f64,
    pub seed_rms_px: f64,
    pub log_odds: f64,
    pub dist_from_truth_deg: Option<f64>,
    pub scale_ratio_to_truth: Option<f64>,
}

impl fmt::Display for SolveTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let class = self.classify();
        writeln!(f, "[svy funnel] {}  OUTCOME={}", self.frame, class.as_str())?;
        writeln!(
            f,
            "  detect={} sel={} img_quads={} g={} hint={} hint_r={:.1}° wall={}ms",
            self.n_detected, self.n_selected, self.n_image_quads, self.group_size,
            self.hint_center.map(|(r, d)| format!("({:.3},{:.3})", r, d)).unwrap_or_else(|| "blind".into()),
            self.hint_radius_deg, self.total_ms,
        )?;
        for p in &self.passes {
            writeln!(f, "  pass={}", if p.balance { "fallback(balance)" } else { "primary" })?;
            for r in &p.rungs {
                let n_cells_with_data: Vec<&CellTrace> = r.cells.iter().filter(|c| c.cat_raw > 0).collect();
                let cat_avg = avg(n_cells_with_data.iter().map(|c| c.cat_raw));
                let cap_avg = avg(n_cells_with_data.iter().map(|c| c.cat_after_cap));
                let quads_avg = avg(n_cells_with_data.iter().map(|c| c.cat_quads));
                let mraw_avg = avg(n_cells_with_data.iter().map(|c| c.match_raw));
                let mfilt_avg = avg(n_cells_with_data.iter().map(|c| c.match_filtered));
                let verified_pass = r.verifies.iter().filter(|v| v.accepted).count();
                let verified_total = r.verifies.len();
                writeln!(
                    f,
                    "    rung {:>2}/{:<2} fov={:>6.2}° cells={}/{:<2} \
                     cat_avg={:>4} cap_avg={:>4} cat_quads_avg={:>4} \
                     match[raw/filt]={:>3}/{:<3} cand={} verified={}/{} best_inl={}",
                    r.idx, r.total, r.fov_deg, n_cells_with_data.len(), r.n_cells,
                    cat_avg, cap_avg, quads_avg, mraw_avg, mfilt_avg,
                    r.candidates_emitted, verified_pass, verified_total, r.best_inliers_seen,
                )?;
                if debug_enabled() {
                    for c in &r.cells {
                        writeln!(
                            f,
                            "        cell({:.3},{:.3}) cat={}→{} cq={} mraw={} mfilt={} stop={}{}",
                            c.ra, c.dec, c.cat_raw, c.cat_after_cap, c.cat_quads,
                            c.match_raw, c.match_filtered, c.stop.as_str(),
                            c.seed.as_ref().map(|s| format!(" seed=({:.3},{:.3}) scale={:.3}\"/px", s.ra, s.dec, s.scale_arcsec_per_px)).unwrap_or_default(),
                        )?;
                    }
                    for v in &r.verifies {
                        let decision = match v.log10_odds {
                            Some(o) => format!(
                                "log10_odds={o:.1}/{:.1} refs={} ",
                                v.required_log10_odds, v.reference_stars
                            ),
                            None => String::new(),
                        };
                        let verdict = if v.accepted {
                            "ACCEPTED".to_string()
                        } else {
                            format!("REJECTED({})", v.reject_reason.as_deref().unwrap_or("?"))
                        };
                        writeln!(
                            f,
                            "        verify seed=({:.3},{:.3}) scale={:.3}\"/px \
                             cat_in_fov={} inl={}/{} final_inl={} rms={:.2}px {}{}",
                            v.seed_ra, v.seed_dec, v.scale_arcsec_per_px,
                            v.cat_in_fov, v.seed_inliers, v.required,
                            v.final_inliers, v.final_rms_px,
                            decision, verdict,
                        )?;
                    }
                }
            }
        }
        if self.outcome.is_failure() {
            if let Some(ba) = self.best_attempt() {
                let dist_from_hint = self.hint_center.map(|(hr, hd)| {
                    crate::math::great_circle_deg(hr, hd, ba.seed_ra, ba.seed_dec)
                });
                let truth_bits = match (ba.dist_from_truth_deg, ba.scale_ratio_to_truth) {
                    (Some(d), Some(s)) => format!(" dist_from_truth={:.3}° scale_ratio={:.3}", d, s),
                    _ => String::new(),
                };
                let lo_str = if ba.log_odds.is_finite() {
                    format!(" log_odds={:.1}", ba.log_odds)
                } else {
                    String::new()
                };
                writeln!(
                    f,
                    "  best_attempt: pass={} rung={}/{} seed=({:.3},{:.3}) scale={:.3}\"/px {}{}\n              cat_in_fov={} inliers={}/{} seed_rms={:.2}px final_rms={:.2}px{}",
                    if ba.pass_balance { "fallback" } else { "primary" },
                    ba.rung_idx, ba.rung_total,
                    ba.seed_ra, ba.seed_dec, ba.scale_arcsec_per_px,
                    dist_from_hint.map(|d| format!("dist_from_hint={:.2}°", d)).unwrap_or_else(|| "no_hint".into()),
                    truth_bits,
                    ba.cat_in_fov, ba.inliers, ba.required, ba.seed_rms_px, ba.rms_px,
                    lo_str,
                )?;
            } else {
                writeln!(f, "  best_attempt: no candidate ever reached verification")?;
            }
            if let FailureClass::NotConfident { best_log10_odds } = class {
                writeln!(f, "  decision: best log10_odds={best_log10_odds:.1} below the bar")?;
            }
            // If a known truth was registered and ANY trial seed landed near
            // it, surface that explicitly — it's the strongest signal that the
            // verify gate (not matching) is what failed us.
            if let Some((d, inl, req, sr)) = self.any_trial_near_truth() {
                if d < 1.0 {
                    writeln!(
                        f,
                        "  TRUTH_PROXIMITY: closest seed was {:.3}° from known truth (inliers={}/{}, scale_ratio={:.3}) → likely a valid-solve REJECTED by the verify gate",
                        d, inl, req, sr,
                    )?;
                }
            }
        }
        Ok(())
    }
}

fn avg<I: IntoIterator<Item = usize>>(iter: I) -> usize {
    let v: Vec<usize> = iter.into_iter().collect();
    if v.is_empty() { 0 } else { v.iter().sum::<usize>() / v.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_path_names_are_stable() {
        assert_eq!(SearchPath::Spiral.as_str(), "spiral");
        assert_eq!(SearchPath::Index.as_str(), "index");
    }

    fn verify(seed_inliers: usize, required: usize, log10_odds: Option<f64>) -> VerifyTrace {
        VerifyTrace {
            seed_ra: 10.0,
            seed_dec: 20.0,
            scale_arcsec_per_px: 1.0,
            cat_in_fov: 100,
            seed_inliers,
            required,
            seed_rms_px: 1.0,
            log_odds: 15.0,
            final_inliers: 0,
            final_rms_px: 0.0,
            dist_from_truth_deg: None,
            scale_ratio_to_truth: None,
            accepted: false,
            reject_reason: None,
            path: SearchPath::Spiral,
            recentered: false,
            hypotheses_so_far: 1,
            own_star_count: 4,
            image_star_count: 300,
            inliers_outside_own: 0,
            tol_px: 1.5,
            reference_stars: 50,
            reference_mag_limit: 0.0,
            test_stars: 296,
            scan_matched: 0,
            matched_stars: 0,
            peak_rank: 0,
            log10_odds,
            log10_odds_quad4: None,
            required_log10_odds: 9.0,
            final_wcs: None,
        }
    }

    /// One pass, one rung, these verifies, enough upstream stages to reach the verify classes.
    fn trace_with(verifies: Vec<VerifyTrace>) -> SolveTrace {
        let mut rung = RungTrace::new(1, 1, 1.0, 1);
        rung.candidates_emitted = verifies.len();
        rung.verifies = verifies;
        let mut pass = PassTrace::new(false);
        pass.rungs.push(rung);
        SolveTrace {
            frame: "f".into(),
            n_detected: 300,
            n_selected: 300,
            n_image_quads: 100,
            passes: vec![pass],
            outcome: TraceOutcome::Failed("no".into()),
            ..Default::default()
        }
    }

    #[test]
    fn a_search_that_reached_the_decision_is_not_confident() {
        // The best score is neither the last verify's nor the best-seed one's.
        let t = trace_with(vec![
            verify(40, 10, Some(6.2)),
            verify(30, 10, Some(7.5)),
            verify(35, 10, Some(7.0)),
        ]);
        assert_eq!(t.classify(), FailureClass::NotConfident { best_log10_odds: 7.5 });
    }

    #[test]
    fn a_not_confident_failure_says_it_could_be_a_chance_alignment() {
        assert_eq!(
            chance_alignment_message(7.5, 9.0),
            "the best match could be a chance alignment (odds 10^7.5 to 1, 10^9.0 needed)"
        );
        // The bar is the one the best-scoring candidate faced; the base bar when none scored.
        let mut hi = verify(30, 10, Some(7.5));
        hi.required_log10_odds = 9.4;
        let lo = verify(40, 10, Some(6.2));
        assert_eq!(best_decision_bar([&lo, &hi], 9.0), 9.4);
        let none = verify(5, 10, Some(f64::NEG_INFINITY));
        assert_eq!(best_decision_bar([&none], 9.0), 9.0);
        let t = trace_with(vec![lo, hi]);
        assert_eq!(t.all_verifies().len(), 2);
    }

    #[test]
    fn a_non_finite_decision_falls_back_to_the_old_classes() {
        let t = trace_with(vec![verify(5, 10, Some(f64::NEG_INFINITY))]);
        assert_eq!(t.classify(), FailureClass::VerifyGap { best: 5, required: 10 });
    }

    #[test]
    fn failure_codes_are_stable() {
        assert_eq!(FailureClass::NotConfident { best_log10_odds: 7.5 }.as_str(), "NOT_CONFIDENT");
        assert_eq!(FailureClass::Solved.as_str(), "SOLVED");
        assert_eq!(FailureClass::DetectTooFew.as_str(), "DETECT_TOO_FEW");
        assert_eq!(FailureClass::ImgQuadsEmpty.as_str(), "IMG_QUADS_EMPTY");
        assert_eq!(FailureClass::MatchStarved.as_str(), "MATCH_STARVED");
        assert_eq!(FailureClass::AffineNeverFit.as_str(), "AFFINE_NEVER_FIT");
        assert_eq!(FailureClass::NoVerifyAttempted.as_str(), "NO_VERIFY_ATTEMPTED");
        assert_eq!(FailureClass::VerifyZeroInliers.as_str(), "VERIFY_ZERO_INLIERS");
        assert_eq!(FailureClass::VerifyGap { best: 1, required: 2 }.as_str(), "VERIFY_GAP");
        assert_eq!(FailureClass::Unclassified.as_str(), "UNCLASSIFIED");
    }
}
