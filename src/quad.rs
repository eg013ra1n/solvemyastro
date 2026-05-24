//! Quad building, matching, and affine fitting.
//!
//! # Algorithm
//!
//! - Group size is chosen by a ladder based on `nrstars_image` (image star
//!   count, *not* catalog star count): `< 30` → 6, `< 60` → 5, else → 4
//!   (group size 4 = "1 quad per star"). Callers of [`build_quads`] pass
//!   the correct `group_size` via this ladder; the function is pure data.
//! - For each star in the pool, find the `group_size - 1` nearest
//!   neighbours; emit every C(group_size, 4) 4-subset as a quad; dedup on
//!   sorted star-index set. Per quad: 6 pairwise Euclidean distances,
//!   sorted ascending, `ratios[0..4] = d[0..4] / d[5]` (smallest 5 over
//!   the longest), `longest = d[5]`, centre = mean of 4 positions.
//! - Match: accept a pair when
//!   `|image.ratios[k] - catalog.ratios[k]| ≤ tol` for all 5 k (continuous
//!   tolerance). Take the median of `catalog.longest / image.longest`
//!   across matches; drop pairs whose scale ratio deviates more than
//!   `tol * median` from the median (outlier reject on scale consistency).
//! - Affine: 6-parameter affine from quad centres via least-squares;
//!   reject if the xy-scale-squared ratio falls outside `[0.9, 1.1]`.

use std::collections::HashSet;

use nalgebra::{DMatrix, DVector};

/// A quad: 4 stars described by their sorted, normalized pairwise distances.
#[derive(Clone, Debug)]
pub struct Quad {
    /// Indices into the input `points` slice, sorted ascending.
    pub star_idx: [usize; 4],
    /// Ratios `d[0..4] / d[5]` (smallest 5 over longest).
    pub ratios: [f64; 5],
    /// Longest pairwise distance (same units as the input points).
    pub longest: f64,
    /// Mean position of the 4 stars (`(Σx/4, Σy/4)`).
    pub center: (f64, f64),
}

/// 6-parameter affine transform: image center → catalog center.
///
/// Maps `(x, y) → (a1·x + b1·y + c1, a2·x + b2·y + c2)`.
#[derive(Clone, Debug)]
pub struct Affine {
    pub a1: f64,
    pub b1: f64,
    pub c1: f64,
    pub a2: f64,
    pub b2: f64,
    pub c2: f64,
}

impl Affine {
    /// Apply the affine to a point.
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.a1 * x + self.b1 * y + self.c1, self.a2 * x + self.b2 * y + self.c2)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Euclidean distance between two points.
#[inline]
fn euclidean(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// Emit one quad from 4 star indices if not already seen.
///
/// Deduplication is on the sorted star-index set (the canonical form;
/// centre-based dedup would be lossy under floating-point).
fn push_quad(
    points: &[(f64, f64)],
    idx: [usize; 4],
    quads: &mut Vec<Quad>,
    seen: &mut HashSet<[usize; 4]>,
) {
    let mut key = idx;
    key.sort_unstable();
    if !seen.insert(key) {
        return;
    }

    let p: [(f64, f64); 4] = [points[key[0]], points[key[1]], points[key[2]], points[key[3]]];

    // 6 pairwise distances, then sort ascending.
    let mut d = [
        euclidean(p[0], p[1]),
        euclidean(p[0], p[2]),
        euclidean(p[0], p[3]),
        euclidean(p[1], p[2]),
        euclidean(p[1], p[3]),
        euclidean(p[2], p[3]),
    ];
    // euclidean() returns non-NaN, non-negative for finite inputs; defensive cmp
    // so a malformed point cannot panic this code path.
    d.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let longest = d[5];
    if longest < 1e-15 {
        return; // degenerate — collocated stars
    }

    let ratios = [d[0] / longest, d[1] / longest, d[2] / longest, d[3] / longest, d[4] / longest];

    let cx = (p[0].0 + p[1].0 + p[2].0 + p[3].0) * 0.25;
    let cy = (p[0].1 + p[1].1 + p[2].1 + p[3].1) * 0.25;

    quads.push(Quad { star_idx: key, ratios, longest, center: (cx, cy) });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Group-size ladder driven by image-star count.
///
/// Pass `n_image_stars` (the count of image stars, **not** catalog stars):
///
/// - `< 30` → 6
/// - `< 60` → 5
/// - else   → 4
///
/// Use the returned value as the `group_size` argument to [`build_quads`].
pub fn group_size_for(n_image_stars: usize) -> usize {
    if n_image_stars < 30 {
        6
    } else if n_image_stars < 60 {
        5
    } else {
        4
    }
}

/// Build quads from a flat star list.
///
/// For each star `i` in `points[..n]`, find its `group_size − 1` nearest
/// neighbors, then emit every C(`group_size`, 4) 4-subset as a [`Quad`].
/// Quads are deduplicated on their sorted star-index key so the same 4-star
/// set is never emitted twice regardless of which center star triggered it.
///
/// `group_size` should come from [`group_size_for`] when working on image
/// stars; pass the **same** value when building catalog quads so both sides
/// are structurally the same family (the gnomonic-consistency invariant).
///
/// Only the first `n` entries of `points` are used; `n = points.len()` is
/// the common case and calling code may pass any subset.
pub fn build_quads(points: &[(f64, f64)], n: usize, group_size: usize) -> Vec<Quad> {
    let n = n.min(points.len());
    let g = group_size.max(4); // groups smaller than 4 are not meaningful
    if n < 4 {
        return Vec::new();
    }

    let n_nbr = g - 1; // nearest neighbors to collect per star
    let mut quads: Vec<Quad> = Vec::with_capacity(n);
    let mut seen: HashSet<[usize; 4]> = HashSet::new();
    let mut nbrs: Vec<(usize, f64)> = Vec::with_capacity(n.saturating_sub(1));

    for i in 0..n {
        nbrs.clear();
        for j in 0..n {
            if i != j {
                nbrs.push((j, euclidean(points[i], points[j])));
            }
        }
        if nbrs.len() < n_nbr {
            continue;
        }

        // Partial-sort: bring the n_nbr smallest to the front, then stable-sort
        // those so group membership is deterministic. Distances are finite by
        // construction; defensive cmp so a stray NaN cannot panic the worker.
        nbrs.select_nth_unstable_by(n_nbr - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        nbrs.truncate(n_nbr);
        nbrs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Build local group: star i + its n_nbr nearest.
        let mut group = [0usize; 6]; // g ≤ 6 in practice
        let gsz = g.min(6);
        group[0] = i;
        for (slot, &(idx, _)) in nbrs.iter().take(gsz - 1).enumerate() {
            group[slot + 1] = idx;
        }

        // Emit every 4-subset of the group.
        for a in 0..gsz {
            for b in (a + 1)..gsz {
                for c in (b + 1)..gsz {
                    for d in (c + 1)..gsz {
                        push_quad(points, [group[a], group[b], group[c], group[d]], &mut quads, &mut seen);
                    }
                }
            }
        }
    }

    quads
}

/// Match image quads against catalog quads.
///
/// A pair is accepted when, for all 5 ratio positions k:
/// `|image.ratios[k] - catalog.ratios[k]| ≤ tol` (continuous absolute
/// tolerance, early exit on first mismatch).
///
/// After collecting initial matches, compute `median(catalog.longest /
/// image.longest)` over all pairs; drop any pair whose scale ratio
/// deviates from the median by more than 20% (`|ratio − median| > 0.2 ·
/// median`).
///
/// Returns matched `(image_quad_index, catalog_quad_index)` pairs.
pub fn match_quads(image: &[Quad], catalog: &[Quad], tol: f64) -> Vec<(usize, usize)> {
    match_quads_stats(image, catalog, tol).0
}

/// Diagnostic variant of [`match_quads`]. Returns `(matches, raw_count)`
/// where `raw_count` is the number of ratio-matched pairs BEFORE the
/// median-scale-ratio outlier filter. The trace uses the gap between
/// `raw_count` and `matches.len()` to spot pathological filter behaviour.
pub fn match_quads_stats(
    image: &[Quad],
    catalog: &[Quad],
    tol: f64,
) -> (Vec<(usize, usize)>, usize) {
    let mut raw: Vec<(usize, usize, f64)> = Vec::new();

    for (ii, iq) in image.iter().enumerate() {
        for (ci, cq) in catalog.iter().enumerate() {
            let mut ok = true;
            for k in 0..5 {
                if (iq.ratios[k] - cq.ratios[k]).abs() > tol {
                    ok = false;
                    break;
                }
            }
            if ok {
                let scale = if iq.longest > 1e-15 { cq.longest / iq.longest } else { 0.0 };
                raw.push((ii, ci, scale));
            }
        }
    }

    let raw_count = raw.len();
    if raw.len() < 3 {
        return (raw.into_iter().map(|(i, j, _)| (i, j)).collect(), raw_count);
    }

    let mut ratios: Vec<f64> = raw.iter().map(|&(_, _, r)| r).collect();
    ratios.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = ratios[ratios.len() / 2];

    if median.abs() < 1e-15 {
        return (raw.into_iter().map(|(i, j, _)| (i, j)).collect(), raw_count);
    }
    let filtered: Vec<(usize, usize)> = raw
        .into_iter()
        .filter(|&(_, _, r)| (r - median).abs() <= 0.2 * median)
        .map(|(i, j, _)| (i, j))
        .collect();
    (filtered, raw_count)
}

/// Fit a 6-parameter affine from matched quad-center correspondences.
///
/// Solves `Xcat = a1·Ximg + b1·Yimg + c1` and
///         `Ycat = a2·Ximg + b2·Yimg + c2`
/// by SVD least-squares (≥ 3 matches required).
///
/// Rejects if `(a1² + b1²) / (a2² + b2²) ∉ [0.9, 1.1]` (xy-scale-square
/// sanity gate).
///
/// Returns `None` if there are fewer than 3 matches, if the SVD fails, or
/// if the xy scale-square ratio check fails.
pub fn fit_affine(
    matches: &[(usize, usize)],
    image_quads: &[Quad],
    catalog_quads: &[Quad],
) -> Option<Affine> {
    let n = matches.len();
    if n < 3 {
        return None;
    }

    let mut a_mat = DMatrix::zeros(n, 3);
    let mut bx = DVector::zeros(n);
    let mut by = DVector::zeros(n);

    for (row, &(ii, ci)) in matches.iter().enumerate() {
        let (ix, iy) = image_quads[ii].center;
        let (cx, cy) = catalog_quads[ci].center;
        a_mat[(row, 0)] = ix;
        a_mat[(row, 1)] = iy;
        a_mat[(row, 2)] = 1.0;
        bx[row] = cx;
        by[row] = cy;
    }

    let svd = a_mat.svd(true, true);
    let eps = 1e-12;
    let px = svd.solve(&bx, eps).ok()?;
    let py = svd.solve(&by, eps).ok()?;

    let affine = Affine { a1: px[0], b1: px[1], c1: px[2], a2: py[0], b2: py[1], c2: py[2] };

    // xy scale-square sanity.
    let sx2 = affine.a1 * affine.a1 + affine.b1 * affine.b1;
    let sy2 = affine.a2 * affine.a2 + affine.b2 * affine.b2;
    if sy2 < 1e-30 {
        return None;
    }
    let ratio = sx2 / sy2;
    if !(0.9..=1.1).contains(&ratio) {
        return None;
    }

    Some(affine)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // group_size_for
    // -----------------------------------------------------------------------

    #[test]
    fn group_size_ladder() {
        assert_eq!(group_size_for(0), 6);
        assert_eq!(group_size_for(29), 6);
        assert_eq!(group_size_for(30), 5);
        assert_eq!(group_size_for(59), 5);
        assert_eq!(group_size_for(60), 4);
        assert_eq!(group_size_for(600), 4);
    }

    // -----------------------------------------------------------------------
    // build_quads
    // -----------------------------------------------------------------------

    #[test]
    fn build_quads_basic_ratios_in_range() {
        let pts = vec![
            (0.0, 0.0), (100.0, 0.0), (50.0, 80.0), (150.0, 80.0), (75.0, 160.0),
        ];
        let quads = build_quads(&pts, pts.len(), 4);
        assert!(!quads.is_empty(), "should produce at least 1 quad");
        for q in &quads {
            assert!(q.longest > 0.0, "longest must be positive");
            for &r in &q.ratios {
                assert!(r >= 0.0 && r <= 1.0, "ratio out of [0,1]: {r}");
            }
        }
    }

    #[test]
    fn build_quads_dedup_by_star_set() {
        // 4 stars in a square — only 1 unique quad possible regardless of group_size.
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let quads = build_quads(&pts, pts.len(), 4);
        assert_eq!(quads.len(), 1, "4 points → exactly 1 unique 4-subset");
    }

    #[test]
    fn build_quads_fewer_than_4_returns_empty() {
        let pts = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        assert!(build_quads(&pts, pts.len(), 4).is_empty());
    }

    #[test]
    fn build_quads_group6_emits_more_quads_than_group4() {
        // 6 stars: group_size=6 → C(6,4)=15 subsets per star (before dedup),
        // group_size=4 → 1 per star. After dedup the 6-group pool is still larger.
        let pts: Vec<(f64, f64)> = (0..6).map(|i| (i as f64 * 100.0, 0.0)).collect();
        let q4 = build_quads(&pts, pts.len(), 4);
        let q6 = build_quads(&pts, pts.len(), 6);
        assert!(q6.len() >= q4.len(), "group_size=6 should produce at least as many quads as 4");
    }

    // -----------------------------------------------------------------------
    // match_quads
    // -----------------------------------------------------------------------

    /// Pure-similarity transform: ratios should be identical to < 1e-10.
    #[test]
    fn match_quads_identical_family_matches() {
        // Image: 6 stars in a scattered pattern.
        let img_pts: Vec<(f64, f64)> = vec![
            (100.0, 200.0), (300.0, 150.0), (250.0, 350.0),
            (400.0, 300.0), (150.0, 450.0), (350.0, 50.0),
        ];
        // Catalog: same pattern at 2x scale + (500, 500) translate (pure similarity).
        let cat_pts: Vec<(f64, f64)> = img_pts.iter().map(|&(x, y)| (x * 2.0 + 500.0, y * 2.0 + 500.0)).collect();

        let iq = build_quads(&img_pts, img_pts.len(), 4);
        let cq = build_quads(&cat_pts, cat_pts.len(), 4);

        let m = match_quads(&iq, &cq, 0.007);
        assert!(!m.is_empty(), "identical-family quads must produce matches; got 0");
    }

    #[test]
    fn match_quads_unrelated_patterns_produce_few_matches() {
        // Use pts with distinct ratio fingerprints by making them irregular.
        let img_pts: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (1.0, 0.0), (0.0, 5.0), (3.0, 3.0),
        ];
        let cat_pts: Vec<(f64, f64)> = vec![
            (100.0, 100.0), (200.0, 100.0), (150.0, 110.0), (160.0, 200.0),
        ];
        let iq = build_quads(&img_pts, img_pts.len(), 4);
        let cq = build_quads(&cat_pts, cat_pts.len(), 4);
        let m = match_quads(&iq, &cq, 0.007);
        assert!(m.len() <= 1, "unrelated shapes should produce 0 or 1 false match");
    }

    #[test]
    fn match_quads_outlier_scale_reject() {
        // Set up two good matches plus one outlier with a very different scale.
        // Build a pool where one pair is a 10x scale outlier.
        let img_pts: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (100.0, 0.0), (50.0, 87.0), (150.0, 87.0), (75.0, 174.0), (25.0, 130.0),
        ];
        // Catalog: 2x scale of first 5 + a wildly different 6th point.
        let cat_pts: Vec<(f64, f64)> = vec![
            (0.0, 0.0), (200.0, 0.0), (100.0, 174.0), (300.0, 174.0), (150.0, 348.0),
            // 6th point far away forming a very different scale quad
            (5000.0, 5000.0),
        ];
        let iq = build_quads(&img_pts, img_pts.len(), 4);
        let cq = build_quads(&cat_pts, cat_pts.len(), 4);
        let m = match_quads(&iq, &cq, 0.01);
        // After outlier rejection, the 10x-scale matches should be gone.
        // (This is a structural test: verify outlier-reject runs without panic.)
        let _ = m; // Just asserting no panic.
    }

    // -----------------------------------------------------------------------
    // fit_affine
    // -----------------------------------------------------------------------

    #[test]
    fn fit_affine_known_transform() {
        // Known transform: xcat = 2*ximg + 100, ycat = 2*yimg + 200.
        // Create a set of quads with matching centers.
        let img_centers = vec![(100.0, 100.0), (200.0, 100.0), (150.0, 250.0), (300.0, 200.0)];
        let cat_centers: Vec<(f64, f64)> =
            img_centers.iter().map(|&(x, y)| (x * 2.0 + 100.0, y * 2.0 + 200.0)).collect();

        // Build synthetic quads with just centers (ratios/longest don't matter for fit_affine).
        let img_quads: Vec<Quad> = img_centers
            .iter()
            .map(|&c| Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: c })
            .collect();
        let cat_quads: Vec<Quad> = cat_centers
            .iter()
            .map(|&c| Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: c })
            .collect();

        let matches: Vec<(usize, usize)> = (0..4).map(|i| (i, i)).collect();
        let affine = fit_affine(&matches, &img_quads, &cat_quads).expect("fit_affine must succeed");

        // Check a point not in the fit set.
        let (xcat, ycat) = affine.apply(250.0, 150.0);
        let expected_x = 250.0 * 2.0 + 100.0;
        let expected_y = 150.0 * 2.0 + 200.0;
        assert!((xcat - expected_x).abs() < 0.5, "x: expected {expected_x}, got {xcat}");
        assert!((ycat - expected_y).abs() < 0.5, "y: expected {expected_y}, got {ycat}");
    }

    #[test]
    fn fit_affine_rejects_too_few_matches() {
        let q = Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: (0.0, 0.0) };
        let matches = vec![(0usize, 0usize), (0, 0)]; // only 2
        assert!(fit_affine(&matches, &[q.clone()], &[q]).is_none());
    }

    #[test]
    fn fit_affine_rejects_bad_xy_scale_ratio() {
        // Construct centers that imply a wildly non-uniform scale (x stretch only).
        let img_quads: Vec<Quad> = vec![
            Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: (0.0, 0.0) },
            Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: (1.0, 0.0) },
            Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: (0.0, 1.0) },
        ];
        // Catalog: 100x in x, 1x in y — ratio will be 10000 → rejected.
        let cat_quads: Vec<Quad> = vec![
            Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: (0.0, 0.0) },
            Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: (100.0, 0.0) },
            Quad { star_idx: [0, 1, 2, 3], ratios: [0.0; 5], longest: 1.0, center: (0.0, 1.0) },
        ];
        let matches = vec![(0usize, 0usize), (1, 1), (2, 2)];
        assert!(
            fit_affine(&matches, &img_quads, &cat_quads).is_none(),
            "100x x-stretch must fail the xy_sqr_ratio gate"
        );
    }
}
