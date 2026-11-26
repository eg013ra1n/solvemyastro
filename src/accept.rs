//! The acceptance gate: whether a solve's numbers are a real solution or a
//! chance alignment, and which check refuses when they are not.
//!
//! The gate never looks at pixels or the solver's internals: a caller builds
//! [`GateMetrics`] from a solve result and asks [`Gate::refusal`] (the named
//! check that fails, in a fixed order) or [`Gate::accepts`]. The chance test
//! is the solver's own: [`GateMetrics::log10_odds`] is the Bayes factor of
//! the final WCS over the image stars ([`crate::SolveSolution::log10_odds`])
//! and the gate holds it to [`crate::verify::ACCEPT_LOG10_ODDS`]; the solver
//! already refused anything below its own bar, so the gate re-reads the odds
//! as a check of what an embedding app stored or what `--no-gate` printed.
//! The RMS ceiling, the matched-star floor and the scale checks are backstops.
//!
//! A change to a threshold here (the solver's base bar
//! [`crate::verify::ACCEPT_LOG10_ODDS`] among them), or to the verification
//! magnitude limit (`orchestrate::VERIFY_MAG_LIMIT`), changes the verdicts an
//! embedding app has cached: it must bump its solve-recipe version
//! (Athenaeum's `config::SOLVE_RECIPE_VERSION`).

use std::fmt;

use crate::verify::ACCEPT_LOG10_ODDS;

/// The gate's thresholds. `Default` is Athenaeum's own defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    /// Base verification tolerance, arcseconds; [`gate_tol_px`] turns it into
    /// the per-frame pixel tolerance the RMS ceiling is relative to.
    pub base_tolerance_arcsec: f64,
    /// The RMS ceiling, as a multiple of the pixel tolerance.
    pub rms_max_px_mult: f64,
    /// Fewest stars matched on the final WCS (one-to-one, within 1 sigma).
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
    /// Matched stars ([`crate::SolveSolution::matched_stars`]).
    pub inliers: usize,
    /// Residual RMS, pixels.
    pub rms_px: f64,
    /// The pixel tolerance, [`gate_tol_px`] of the solve's scale.
    pub tol_px: f64,
    /// `log10` of the solver's odds that the match is real
    /// ([`crate::SolveSolution::log10_odds`]).
    pub log10_odds: f64,
    /// The solved scale, arcsec/px.
    pub recovered_scale_arcsec: f64,
    /// The scale the frame's own optics imply, when it declares them.
    pub header_scale_arcsec: Option<f64>,
}

/// The check that refused a solution, with the numbers it judged. `Display`
/// is the one wording of each refusal: the CLI prints `rejected: {r}` and an
/// embedding app can show the same text.
#[derive(Clone, Debug, PartialEq)]
pub enum GateRefusal {
    /// The residual RMS is above the ceiling.
    LooseFit { rms_px: f64, limit_px: f64 },
    /// Fewer stars matched than the floor.
    TooFewMatched { matched: usize, needed: usize },
    /// The odds are below the solver's bar, or not a number.
    NotConfident { log10_odds: f64, needed: f64, matched: usize },
    /// The solved scale is outside the plausible range.
    ImplausibleScale { scale_arcsec: f64, min: f64, max: f64 },
    /// The solved scale is too far from the header's.
    OffHeaderScale { scale_arcsec: f64, header_scale_arcsec: f64, factor: f64 },
}

impl fmt::Display for GateRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LooseFit { rms_px, limit_px } => {
                write!(f, "the stars fit too loosely ({rms_px:.2} px RMS, {limit_px:.2} px allowed)")
            }
            Self::TooFewMatched { matched, needed } => {
                write!(f, "only {matched} stars matched ({needed} needed)")
            }
            Self::NotConfident { log10_odds, needed, matched } => write!(
                f,
                "the match could be a chance alignment (odds 10^{log10_odds:.1} to 1, \
                 10^{needed} needed; {matched} stars matched)"
            ),
            Self::ImplausibleScale { scale_arcsec, min, max } => {
                write!(f, "the solved scale {scale_arcsec:.2}\"/px is outside {min}..{max}\"/px")
            }
            Self::OffHeaderScale { scale_arcsec, header_scale_arcsec, factor } => write!(
                f,
                "the solved scale {scale_arcsec:.2}\"/px is more than {factor}x off the header's \
                 {header_scale_arcsec:.2}\"/px"
            ),
        }
    }
}

impl Gate {
    /// The first check `m` fails, or `None` when it is an acceptable
    /// solution. The order is fixed: RMS, matched stars, odds, plausible
    /// scale, scale against the header. A non-finite RMS is `LooseFit`, a
    /// non-finite odds is `NotConfident`.
    pub fn refusal(&self, m: &GateMetrics) -> Option<GateRefusal> {
        // Geometric fit must be tight (scale-relative ceiling; loose backstop).
        let limit_px = self.rms_max_px_mult * m.tol_px;
        if !m.rms_px.is_finite() || m.rms_px > limit_px {
            return Some(GateRefusal::LooseFit { rms_px: m.rms_px, limit_px });
        }
        if m.inliers < self.inlier_floor {
            return Some(GateRefusal::TooFewMatched { matched: m.inliers, needed: self.inlier_floor });
        }
        // The primary false-positive gate: the solver's own odds, read at the
        // solver's base bar `ACCEPT_LOG10_ODDS`, not the hypothesis-raised
        // one; a non-finite value refuses.
        if !m.log10_odds.is_finite() || m.log10_odds < ACCEPT_LOG10_ODDS {
            return Some(GateRefusal::NotConfident {
                log10_odds: m.log10_odds,
                needed: ACCEPT_LOG10_ODDS,
                matched: m.inliers,
            });
        }
        // Recovered scale must be physically plausible.
        if !(self.scale_sanity_min..=self.scale_sanity_max).contains(&m.recovered_scale_arcsec) {
            return Some(GateRefusal::ImplausibleScale {
                scale_arcsec: m.recovered_scale_arcsec,
                min: self.scale_sanity_min,
                max: self.scale_sanity_max,
            });
        }
        // ...and not wildly off the header scale when the header had one.
        if let Some(hs) = m.header_scale_arcsec {
            if hs > 0.0 {
                let r = m.recovered_scale_arcsec / hs;
                if r < 1.0 / self.scale_header_tol || r > self.scale_header_tol {
                    return Some(GateRefusal::OffHeaderScale {
                        scale_arcsec: m.recovered_scale_arcsec,
                        header_scale_arcsec: hs,
                        factor: self.scale_header_tol,
                    });
                }
            }
        }
        None
    }

    /// Whether `m` is an acceptable solution: [`Gate::refusal`] finds nothing.
    pub fn accepts(&self, m: &GateMetrics) -> bool {
        self.refusal(m).is_none()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> GateMetrics {
        GateMetrics {
            inliers: 40,
            rms_px: 1.2,
            tol_px: 6.0,
            log10_odds: 300.0,
            recovered_scale_arcsec: 1.8,
            header_scale_arcsec: Some(1.9),
        }
    }

    #[test]
    fn gate_names_each_refusal_in_order() {
        let gate = Gate::default();
        assert_eq!(gate.refusal(&base()), None);
        assert!(matches!(
            gate.refusal(&GateMetrics { rms_px: 20.0, ..base() }),
            Some(GateRefusal::LooseFit { .. })
        ));
        assert_eq!(
            gate.refusal(&GateMetrics { inliers: 5, ..base() }),
            Some(GateRefusal::TooFewMatched { matched: 5, needed: 6 })
        );
        assert!(matches!(
            gate.refusal(&GateMetrics { log10_odds: 8.9, ..base() }),
            Some(GateRefusal::NotConfident { .. })
        ));
        assert!(matches!(
            gate.refusal(&GateMetrics { log10_odds: f64::NAN, ..base() }),
            Some(GateRefusal::NotConfident { .. })
        ));
        assert!(matches!(
            gate.refusal(&GateMetrics { log10_odds: f64::NEG_INFINITY, ..base() }),
            Some(GateRefusal::NotConfident { .. })
        ));
        // Scale 100 against the header's 1.9 also fails the header rule, so
        // this case pins plausible-scale before scale-against-header.
        assert!(matches!(
            gate.refusal(&GateMetrics { recovered_scale_arcsec: 100.0, ..base() }),
            Some(GateRefusal::ImplausibleScale { .. })
        ));
        assert!(matches!(
            gate.refusal(&GateMetrics { recovered_scale_arcsec: 20.0, ..base() }),
            Some(GateRefusal::OffHeaderScale { .. })
        ));
        // Non-finite RMS is never a real solve.
        for rms_px in [f64::NAN, f64::INFINITY] {
            assert!(matches!(
                gate.refusal(&GateMetrics { rms_px, ..base() }),
                Some(GateRefusal::LooseFit { .. })
            ));
        }
        // The first failing check wins: every adjacent pair of the order.
        assert!(matches!(
            gate.refusal(&GateMetrics { rms_px: 20.0, log10_odds: 3.0, ..base() }),
            Some(GateRefusal::LooseFit { .. })
        ));
        assert!(matches!(
            gate.refusal(&GateMetrics { rms_px: 20.0, inliers: 5, ..base() }),
            Some(GateRefusal::LooseFit { .. })
        ));
        assert_eq!(
            gate.refusal(&GateMetrics { inliers: 5, log10_odds: 3.0, ..base() }),
            Some(GateRefusal::TooFewMatched { matched: 5, needed: 6 })
        );
        assert!(matches!(
            gate.refusal(&GateMetrics { log10_odds: 3.0, recovered_scale_arcsec: 100.0, ..base() }),
            Some(GateRefusal::NotConfident { .. })
        ));
        // No header scale: the header rule is skipped entirely.
        assert_eq!(
            gate.refusal(&GateMetrics {
                header_scale_arcsec: None,
                recovered_scale_arcsec: 20.0,
                ..base()
            }),
            None
        );
        // The plausible-scale range, edges inclusive (no header to compare).
        let no_header = |recovered_scale_arcsec| GateMetrics {
            header_scale_arcsec: None,
            recovered_scale_arcsec,
            ..base()
        };
        assert!(matches!(
            gate.refusal(&no_header(0.01)),
            Some(GateRefusal::ImplausibleScale { .. })
        ));
        assert_eq!(gate.refusal(&no_header(0.05)), None);
        assert_eq!(gate.refusal(&no_header(60.0)), None);
    }

    /// The strict edges of the gate. The inputs are exact in f64 (4.0 * 2.5 =
    /// 10.0, 8.0 / 1.0 = 8.0, 1.0 / 8.0 = 0.125), so no assert is float-fragile.
    #[test]
    fn gate_boundaries_are_pinned() {
        let gate = Gate::default();
        let base = GateMetrics {
            inliers: 40,
            rms_px: 1.0,
            tol_px: 4.0,
            log10_odds: 50.0,
            recovered_scale_arcsec: 1.0,
            header_scale_arcsec: Some(1.0),
        };
        assert!(gate.accepts(&base));
        // rms exactly at the ceiling accepts; just above refuses.
        assert!(gate.accepts(&GateMetrics { rms_px: 10.0, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { rms_px: 10.5, ..base.clone() }));
        // Odds exactly at the bar accept; just below refuse.
        assert!(gate.accepts(&GateMetrics { log10_odds: 9.0, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { log10_odds: 8.999, ..base.clone() }));
        // A header ratio exactly at the tolerance accepts, either way round;
        // beyond it refuses.
        assert!(gate.accepts(&GateMetrics { recovered_scale_arcsec: 8.0, ..base.clone() }));
        assert!(gate.accepts(&GateMetrics { recovered_scale_arcsec: 0.125, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { recovered_scale_arcsec: 8.5, ..base.clone() }));
        assert!(!gate.accepts(&GateMetrics { recovered_scale_arcsec: 0.1, ..base.clone() }));
        // The default floor is 6: five refuse, six pass.
        assert!(!gate.accepts(&GateMetrics { inliers: 5, ..base.clone() }));
        assert!(gate.accepts(&GateMetrics { inliers: 6, ..base }));
    }

    /// The gate refuses the recorded false solve: a DSLR frame with no
    /// FOCALLEN/XPIXSZ in its header "solved" at Dec -68 deg, a sky it was
    /// never pointed at: 6 matches, RMS 1.159 px (2026-09-05). Its odds are
    /// what the solver's decision would have scored it (a chance alignment).
    #[test]
    fn gate_refuses_the_recorded_false_solve() {
        let gate = Gate::default();
        let noise = GateMetrics {
            inliers: 6,
            rms_px: 1.159,
            tol_px: gate_tol_px(0.2178, gate.base_tolerance_arcsec),
            log10_odds: 4.8,
            recovered_scale_arcsec: 0.2178,
            header_scale_arcsec: None,
        };
        let r = gate.refusal(&noise);
        assert_eq!(
            r,
            Some(GateRefusal::NotConfident { log10_odds: 4.8, needed: 9.0, matched: 6 })
        );
        assert_eq!(
            r.unwrap().to_string(),
            "the match could be a chance alignment (odds 10^4.8 to 1, 10^9 needed; 6 stars matched)"
        );
        // The same field with a real solve's odds passes.
        assert!(gate.accepts(&GateMetrics { log10_odds: 120.0, ..noise }));
    }

    #[test]
    fn refusal_texts_are_pinned() {
        let text = |r: GateRefusal| r.to_string();
        assert_eq!(
            text(GateRefusal::LooseFit { rms_px: 20.0, limit_px: 10.53 }),
            "the stars fit too loosely (20.00 px RMS, 10.53 px allowed)"
        );
        assert_eq!(
            text(GateRefusal::TooFewMatched { matched: 5, needed: 6 }),
            "only 5 stars matched (6 needed)"
        );
        assert_eq!(
            text(GateRefusal::NotConfident { log10_odds: 6.2, needed: 9.0, matched: 12 }),
            "the match could be a chance alignment (odds 10^6.2 to 1, 10^9 needed; 12 stars matched)"
        );
        assert_eq!(
            text(GateRefusal::ImplausibleScale { scale_arcsec: 100.0, min: 0.05, max: 60.0 }),
            "the solved scale 100.00\"/px is outside 0.05..60\"/px"
        );
        assert_eq!(
            text(GateRefusal::OffHeaderScale {
                scale_arcsec: 29.6,
                header_scale_arcsec: 1.873,
                factor: 8.0
            }),
            "the solved scale 29.60\"/px is more than 8x off the header's 1.87\"/px"
        );
    }

    #[test]
    fn gate_defaults_are_the_apps_defaults() {
        let g = Gate::default();
        assert_eq!(g.base_tolerance_arcsec, 8.0);
        assert_eq!(g.rms_max_px_mult, 2.5);
        assert_eq!(g.inlier_floor, 6);
        assert_eq!(g.scale_sanity_min, 0.05);
        assert_eq!(g.scale_sanity_max, 60.0);
        assert_eq!(g.scale_header_tol, 8.0);
        assert_eq!(
            ACCEPT_LOG10_ODDS, 9.0,
            "ACCEPT_LOG10_ODDS is the gate's odds bar: moving it changes embedding apps' cached \
             verdicts: bump their solve-recipe version (Athenaeum's config::SOLVE_RECIPE_VERSION)"
        );
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

    /// The decision's references are the catalogue stars to the verification
    /// magnitude limit ([`crate::orchestrate::VERIFY_MAG_LIMIT`], the first
    /// level of the decision's list): moving it moves `N_R`, the odds and so
    /// every embedding app's verdicts.
    #[test]
    fn the_verify_magnitude_limit_is_pinned() {
        assert_eq!(
            crate::orchestrate::VERIFY_MAG_LIMIT,
            12.0,
            "VERIFY_MAG_LIMIT is the decision's first reference level: moving it moves N_R, the \
             odds and embedding apps' verdicts: bump their solve-recipe version (Athenaeum's \
             config::SOLVE_RECIPE_VERSION)"
        );
    }
}
