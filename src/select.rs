//! Star selection for the quad-building pool.
//!
//! # Algorithm
//!
//! 1. Discard detections whose `snr ≤ SNR_THRESHOLD` (= 10.0).
//! 2. Rank survivors by SNR descending (brightest first).
//! 3. Cap at the pool-size ceiling (default 600).
//!
//! `FastStar.snr` (rustafits `detect_fast`) is the aperture-photometry
//! signal-to-noise ratio with background noise already folded in, so the
//! aperture-SNR rank inherently demotes extended structure — no separate
//! HFD gate is needed for this pool.

/// SNR floor below which a detection is treated as noise.
const SNR_THRESHOLD: f32 = 10.0;

/// Select the quad-building star pool from raw `detect_fast` detections.
///
/// Steps:
/// 1. Discard any star with `snr ≤ SNR_THRESHOLD`.
/// 2. Rank survivors by SNR descending (brightest first).
/// 3. Return at most `max` stars as `(x, y)` image positions (`f64`).
///
/// # Parameters
///
/// - `stars` — raw detections from `astroimage::ImageAnalyzer::detect_fast`.
///   Each element must expose `.snr: f32`, `.x: f32`, `.y: f32`.
/// - `max` — pool-size ceiling (default via [`DEFAULT_MAX`]).
///
/// # Returns
///
/// `Vec<(f64, f64)>` of selected star positions, SNR-ordered, length ≤ `max`.
///
/// # Examples
///
/// ```
/// // (No external dep in doctest — just show the contract.)
/// use solvemyastro::select::select_stars_for_quads;
///
/// struct FakeStar { x: f32, y: f32, snr: f32 }
///
/// let raw = vec![
///     FakeStar { x: 10.0, y: 20.0, snr: 50.0 },
///     FakeStar { x: 30.0, y: 40.0, snr: 5.0 },   // below threshold
///     FakeStar { x: 50.0, y: 60.0, snr: 200.0 },
/// ];
///
/// let pool = select_stars_for_quads(&raw, 600,
///     |s: &FakeStar| (s.x, s.y, s.snr));
///
/// assert_eq!(pool.len(), 2);                      // low-SNR dropped
/// assert!(pool[0].0 == 50.0_f64);                 // highest SNR first
/// ```
pub fn select_stars_for_quads<T, F>(stars: &[T], max: usize, accessor: F) -> Vec<(f64, f64)>
where
    F: Fn(&T) -> (f32, f32, f32), // (x, y, snr)
{
    // Gate: keep only SNR > threshold.
    let mut passing: Vec<(f32, f32, f32)> = stars
        .iter()
        .map(|s| accessor(s))
        .filter(|&(_, _, snr)| snr > SNR_THRESHOLD)
        .collect();

    // Rank by SNR descending (brightest first).
    passing.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Cap at max and convert to f64 positions.
    passing
        .into_iter()
        .take(max)
        .map(|(x, y, _snr)| (x as f64, y as f64))
        .collect()
}

/// Default pool-size ceiling.
pub const DEFAULT_MAX: usize = 600;

/// Convenience wrapper for `astroimage::FastStar`.
///
/// Call this when you have a `&[astroimage::FastStar]` — it wires the
/// accessor for you.  The `astroimage` dep is always present (rustafits path
/// dep), so no feature gate is needed.
///
/// ```ignore
/// use solvemyastro::select::select_fast_stars;
/// let pool = select_fast_stars(&result.stars, solvemyastro::select::DEFAULT_MAX);
/// ```
pub fn select_fast_stars(
    stars: &[astroimage::FastStar],
    max: usize,
) -> Vec<(f64, f64)> {
    select_stars_for_quads(stars, max, |s| (s.x, s.y, s.snr))
}

/// Select the quad-building star pool from `detect_fast` detections, returning
/// per-star centroid σ alongside positions.  Used by the precise SIP path to
/// supply inverse-variance weights to [`crate::sip::fit_sip_annealed`].
///
/// Applies the same SNR gate and ranking as [`select_fast_stars`].  The
/// returned tuple is `(x, y, σx, σy)` in pixels.  `σx`/`σy` mirror
/// `FastStar::sx`/`sy` — they are `0.0` when the star was not PSF-refined
/// (i.e. refinement is off or the per-star gate fired); `fit_sip_annealed`
/// treats `0` / non-finite σ as "unknown" and applies uniform weight for that
/// star.
///
/// # Contract
///
/// The returned `Vec` is **always the same length and in the same order** as
/// the companion `Vec<(f64,f64)>` produced by calling `select_fast_stars` on
/// the same `stars` slice with the same `max`.  Callers that need both can
/// call this function and build the position slice with `.map(|(x,y,_,_)| (x,y))`.
pub fn select_fast_stars_with_sigma(
    stars: &[astroimage::FastStar],
    max: usize,
) -> Vec<(f64, f64, f64, f64)> {
    // Gate: keep only SNR > threshold, same predicate as select_fast_stars.
    let mut passing: Vec<(f32, f32, f32, f32, f32)> = stars
        .iter()
        .map(|s| (s.x, s.y, s.snr, s.sx, s.sy))
        .filter(|&(_, _, snr, _, _)| snr > SNR_THRESHOLD)
        .collect();

    // Rank by SNR descending — byte-identical ordering to select_fast_stars.
    passing.sort_unstable_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    passing
        .into_iter()
        .take(max)
        .map(|(x, y, _snr, sx, sy)| (x as f64, y as f64, sx as f64, sy as f64))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct S {
        x: f32,
        y: f32,
        snr: f32,
    }

    fn acc(s: &S) -> (f32, f32, f32) {
        (s.x, s.y, s.snr)
    }

    #[test]
    fn snr_gate_filters_below_threshold() {
        let stars = vec![
            S { x: 1.0, y: 1.0, snr: 5.0 },   // below → dropped
            S { x: 2.0, y: 2.0, snr: 10.0 },  // equal → dropped (> not >=)
            S { x: 3.0, y: 3.0, snr: 10.1 },  // above → kept
            S { x: 4.0, y: 4.0, snr: 50.0 },  // above → kept
        ];
        let pool = select_stars_for_quads(&stars, 600, acc);
        assert_eq!(pool.len(), 2, "only SNR>10 pass the gate");
        // No star with x==1 or x==2 in pool.
        assert!(pool.iter().all(|&(x, _)| x > 2.0));
    }

    #[test]
    fn snr_ordering_descending() {
        let stars = vec![
            S { x: 10.0, y: 0.0, snr: 30.0 },
            S { x: 20.0, y: 0.0, snr: 100.0 },
            S { x: 30.0, y: 0.0, snr: 50.0 },
            S { x: 40.0, y: 0.0, snr: 11.0 },
        ];
        let pool = select_stars_for_quads(&stars, 600, acc);
        assert_eq!(pool.len(), 4);
        // Verify SNR descending by checking x order: 20,30,10,40
        assert_eq!(pool[0].0, 20.0, "highest SNR=100 first");
        assert_eq!(pool[1].0, 30.0, "SNR=50 second");
        assert_eq!(pool[2].0, 10.0, "SNR=30 third");
        assert_eq!(pool[3].0, 40.0, "SNR=11 last");
    }

    #[test]
    fn max_cap_respected() {
        let stars: Vec<S> = (0..100)
            .map(|i| S { x: i as f32, y: 0.0, snr: 100.0 + i as f32 })
            .collect();
        let pool = select_stars_for_quads(&stars, 10, acc);
        assert_eq!(pool.len(), 10, "pool capped at max=10");
    }

    #[test]
    fn max_cap_takes_highest_snr() {
        // 20 stars with SNR 11..30; cap at 5 → should be the 5 with highest SNR
        let stars: Vec<S> = (0..20)
            .map(|i| S { x: i as f32, y: 0.0, snr: 11.0 + i as f32 })
            .collect();
        let pool = select_stars_for_quads(&stars, 5, acc);
        assert_eq!(pool.len(), 5);
        // Highest 5 are SNR 30,29,28,27,26 → x = 19,18,17,16,15
        for (i, &(x, _)) in pool.iter().enumerate() {
            let expected_x = (19 - i) as f64;
            assert_eq!(x, expected_x, "rank {i}: expected x={expected_x}, got {x}");
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let stars: Vec<S> = vec![];
        let pool = select_stars_for_quads(&stars, 600, acc);
        assert!(pool.is_empty());
    }

    #[test]
    fn all_below_threshold_returns_empty() {
        let stars = vec![
            S { x: 1.0, y: 1.0, snr: 1.0 },
            S { x: 2.0, y: 2.0, snr: 9.9 },
        ];
        let pool = select_stars_for_quads(&stars, 600, acc);
        assert!(pool.is_empty());
    }

    #[test]
    fn positions_are_f64_upcast() {
        let stars = vec![S { x: 123.5, y: 456.7, snr: 50.0 }];
        let pool = select_stars_for_quads(&stars, 600, acc);
        assert_eq!(pool.len(), 1);
        assert!((pool[0].0 - 123.5_f64).abs() < 1e-4);
        assert!((pool[0].1 - 456.7_f64).abs() < 1e-4);
    }
}
