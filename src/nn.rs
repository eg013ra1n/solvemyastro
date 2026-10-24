//! Uniform-grid nearest-neighbour lookup over image detections.
//!
//! Catalog→detection matching (`count_inliers`, the SIP pair collection) asks
//! "nearest detection to this projected catalog star, if within `reach`" for
//! every catalog star in the field — tens of thousands on wide frames, each a
//! full scan over ~600 detections. The grid answers it from the 3×3 cells
//! around the query, which hold every point within `reach` because a cell is
//! at least `reach` wide.
//!
//! Exactness: [`PointGrid::nearest_by`] returns the same point a linear scan
//! with a strict `<` would (lowest index among equal metrics) whenever that
//! point lies within `reach`; callers reject anything farther, so their
//! results are unchanged.

pub(crate) struct PointGrid<'a> {
    points: &'a [(f64, f64)],
    min_x: f64,
    min_y: f64,
    cell: f64,
    ncols: usize,
    nrows: usize,
    /// CSR layout: indices of the points in cell `c` are
    /// `idx[start[c]..start[c + 1]]`, ascending.
    start: Vec<u32>,
    idx: Vec<u32>,
}

impl<'a> PointGrid<'a> {
    /// Grid over `points` for queries whose accepted match lies within
    /// `reach` of the query point.
    pub(crate) fn new(points: &'a [(f64, f64)], reach: f64) -> Self {
        let n = points.len();
        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(x, y) in points {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
        if n == 0 {
            return Self {
                points,
                min_x: 0.0,
                min_y: 0.0,
                cell: 1.0,
                ncols: 0,
                nrows: 0,
                start: vec![0],
                idx: Vec::new(),
            };
        }
        // ~1 point per cell, but never narrower than `reach`.
        let area = ((max_x - min_x) * (max_y - min_y)).max(1e-9);
        let cell = reach.max((area / n as f64).sqrt()).max(1e-9);
        let ncols = ((max_x - min_x) / cell) as usize + 1;
        let nrows = ((max_y - min_y) / cell) as usize + 1;

        let cell_of = |x: f64, y: f64| -> usize {
            let c = (((x - min_x) / cell) as usize).min(ncols - 1);
            let r = (((y - min_y) / cell) as usize).min(nrows - 1);
            r * ncols + c
        };
        let mut start = vec![0u32; ncols * nrows + 1];
        for &(x, y) in points {
            start[cell_of(x, y) + 1] += 1;
        }
        for c in 0..ncols * nrows {
            start[c + 1] += start[c];
        }
        let mut fill = start.clone();
        let mut idx = vec![0u32; n];
        for (i, &(x, y)) in points.iter().enumerate() {
            let c = cell_of(x, y);
            idx[fill[c] as usize] = i as u32;
            fill[c] += 1;
        }
        Self { points, min_x, min_y, cell, ncols, nrows, start, idx }
    }

    /// Point minimising `metric(point)` among the cells around `(px, py)`,
    /// ties to the lowest index, as `(index, metric)`. Every point within
    /// `reach` of the query is considered; `None` when there are none nearby.
    pub(crate) fn nearest_by(
        &self,
        px: f64,
        py: f64,
        metric: impl Fn((f64, f64)) -> f64,
    ) -> Option<(usize, f64)> {
        if self.ncols == 0 {
            return None;
        }
        let c = ((px - self.min_x) / self.cell).floor();
        let r = ((py - self.min_y) / self.cell).floor();
        if !c.is_finite() || !r.is_finite() {
            return None;
        }
        let (c, r) = (c as i64, r as i64);
        let mut best: Option<(usize, f64)> = None;
        for rr in (r - 1).max(0)..=(r + 1).min(self.nrows as i64 - 1) {
            for cc in (c - 1).max(0)..=(c + 1).min(self.ncols as i64 - 1) {
                let cell = rr as usize * self.ncols + cc as usize;
                for &i in &self.idx[self.start[cell] as usize..self.start[cell + 1] as usize] {
                    let i = i as usize;
                    let m = metric(self.points[i]);
                    let better = match best {
                        None => m < f64::MAX,
                        Some((bi, bm)) => m < bm || (m == bm && i < bi),
                    };
                    if better {
                        best = Some((i, m));
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    /// The grid returns exactly what a strict-`<` linear scan returns for
    /// every query whose nearest point is within reach.
    #[test]
    fn matches_linear_scan() {
        let mut s = 7u64;
        for &(n, reach) in &[(1usize, 2.0), (37, 1.5), (600, 1.5), (600, 20.0), (600, 300.0)] {
            let mut pts: Vec<(f64, f64)> =
                (0..n).map(|_| (lcg(&mut s) * 6000.0, lcg(&mut s) * 4000.0)).collect();
            // Duplicates exercise the lowest-index tie rule.
            if n > 10 {
                pts[5] = pts[3];
            }
            let grid = PointGrid::new(&pts, reach);
            for q in 0..5000 {
                let (px, py) = if q % 3 == 0 && n > 0 {
                    let p = pts[q % n];
                    (p.0 + (lcg(&mut s) - 0.5) * reach, p.1 + (lcg(&mut s) - 0.5) * reach)
                } else {
                    (lcg(&mut s) * 6100.0 - 50.0, lcg(&mut s) * 4100.0 - 50.0)
                };
                let d2 = |(x, y): (f64, f64)| (x - px).powi(2) + (y - py).powi(2);
                let mut best = (usize::MAX, f64::MAX);
                for (i, &p) in pts.iter().enumerate() {
                    if d2(p) < best.1 {
                        best = (i, d2(p));
                    }
                }
                let got = grid.nearest_by(px, py, d2);
                if best.1 <= reach * reach {
                    assert_eq!(got, Some(best), "n={n} reach={reach} q=({px},{py})");
                } else if let Some((_, m)) = got {
                    assert!(m > reach * reach);
                }
            }
        }
    }

    #[test]
    fn empty_grid_finds_nothing() {
        let grid = PointGrid::new(&[], 3.0);
        assert_eq!(grid.nearest_by(1.0, 1.0, |_| 0.0), None);
    }
}
