//! The density tiers: the Gaia bins ([`crate::bins`]) cut by rank into four
//! additive `stars.smac` caches.
//!
//! Tier `k` owns the stars of rank `[counts[k], counts[k+1])` in every level-6
//! cell (a bin is sorted by magnitude, so the bands are prefixes of it), which
//! makes a tier a density limit: 500, 2000, 5000 and 8000 stars per square
//! degree, each added on top of the ones before. Every cell is cut by the same
//! counts, so a cell the files reached only in part gives a thinner band than
//! a whole one; [`build_tiers`] says so when the coverage record shows it.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};

use crate::bins::{read_records_until_mag, RECORD_SIZE};
use crate::cache::{build_cache_parallel, StarRecord};
use crate::gaia::Coverage;
use crate::layout::tier_dir_name;

/// Cumulative density targets (stars/deg²) for the 4 additive tiers.
pub const TIER_DENSITIES: [u32; 4] = [500, 2000, 5000, 8000];

/// Full-sphere area / HEALPix-6 cell count = per-cell solid angle in deg².
const CELL_AREA_DEG2: f64 = 41_252.961_25 / 49_152.0;

/// Cumulative per-cell record counts `[0, b1, b2, b3, b4]`. Tier `k` owns the
/// rank band `[counts[k], counts[k+1])`.
pub fn cell_cum_counts() -> [usize; 5] {
    let mut out = [0usize; 5];
    for (i, d) in TIER_DENSITIES.iter().enumerate() {
        out[i + 1] = (*d as f64 * CELL_AREA_DEG2).round() as usize;
    }
    out
}

/// The slice of `records` for rank band `[lo, hi)`, clamped to the data length.
/// Records must be mag-sorted ascending; bands are disjoint and tile the cell.
pub fn slice_select<T>(records: &[T], lo: usize, hi: usize) -> &[T] {
    let end = hi.min(records.len());
    let start = lo.min(end);
    &records[start..end]
}

fn to_smac(r: &crate::bins::BinRecord) -> StarRecord {
    StarRecord {
        ra: r.ra as f64,
        dec: r.dec as f64,
        mag: r.mag(),
        pmra_mas_yr: r.pmra_mas_yr(),
        pmdec_mas_yr: r.pmdec_mas_yr(),
    }
}

/// Read pixel `px`'s band `[lo, hi)` from its `healpix_<px>.bin` tile (records
/// mag-sorted). Reads only the first `min(hi, total)` records — the bands are
/// prefixes of the sorted cell, so dense cells are never fully parsed. Missing
/// tiles (empty pixels) yield an empty band.
fn read_band(bins_dir: &Path, px: u64, lo: usize, hi: usize) -> Vec<StarRecord> {
    let path = bins_dir.join(crate::bins::bin_file_name(px));
    let total = std::fs::metadata(&path).map(|m| m.len() as usize / RECORD_SIZE).unwrap_or(0);
    let n = hi.min(total); // records to actually read (prefix)
    if n == 0 || lo >= n {
        return Vec::new();
    }
    let mut buf = vec![0u8; n * RECORD_SIZE];
    if let Err(e) = File::open(&path).and_then(|mut f| f.read_exact(&mut buf)) {
        tracing::warn!(path = %path.display(), error = %e, "bin unreadable, its stars are skipped");
        return Vec::new();
    }
    // The bin is mag-sorted, so the first n records are the n brightest.
    let cell = read_records_until_mag(&buf, f32::MAX);
    slice_select(&cell, lo, hi).iter().map(to_smac).collect()
}

/// What the coverage record says is wrong with the bins, if anything.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoverageGap {
    /// No `coverage.json`: the bins' origin is unknown.
    NoRecord,
    /// A file without a range in its name: no cell can be called whole.
    UnknownFiles,
    /// This many cells were reached by some of their files only.
    PartialCells(usize),
}

fn coverage_gap(coverage: Option<&Coverage>) -> Option<CoverageGap> {
    let Some(cov) = coverage else { return Some(CoverageGap::NoRecord) };
    match cov.partial_cells() {
        None => Some(CoverageGap::UnknownFiles),
        Some(cells) if cells.is_empty() => None,
        Some(cells) => Some(CoverageGap::PartialCells(cells.len())),
    }
}

/// Build `out_dir/tier_<density>/stars.smac` for each tier from the bins.
/// Returns `(density, star_count)` per tier in [`TIER_DENSITIES`] order. Each
/// tier build saturates all cores; tiers run sequentially.
///
/// A bins folder with no coverage record, or one whose record shows cells the
/// files reached in part only, still builds: the tiers are then thinner there
/// than the density they are named for, and the log says so.
pub fn build_tiers(bins_dir: &Path, out_dir: &Path, epoch: f64) -> Result<Vec<(u32, usize)>> {
    let coverage = Coverage::read(bins_dir)?;
    match coverage_gap(coverage.as_ref()) {
        None => {}
        Some(CoverageGap::NoRecord) => {
            tracing::warn!(bins = %bins_dir.display(), "no coverage record, so which cells are whole is unknown")
        }
        Some(CoverageGap::UnknownFiles) => {
            tracing::warn!(bins = %bins_dir.display(), "coverage record has files without a range, so which cells are whole is unknown")
        }
        Some(CoverageGap::PartialCells(n)) => {
            tracing::warn!(bins = %bins_dir.display(), partial_cells = n, "some cells were reached by part of their files only, their tiers are thinner")
        }
    }

    let bounds = cell_cum_counts();
    let mut out = Vec::with_capacity(TIER_DENSITIES.len());
    for (k, &density) in TIER_DENSITIES.iter().enumerate() {
        let (lo, hi) = (bounds[k], bounds[k + 1]);
        let tier_dir = out_dir.join(tier_dir_name(density));
        tracing::info!(density, lo, hi, "building tier");
        let n = build_cache_parallel(&tier_dir, epoch, |px| read_band(bins_dir, px, lo, hi))
            .with_context(|| format!("build {}", tier_dir_name(density)))?;
        tracing::info!(density, stars = n, "tier built");
        out.push((density, n));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaia::coverage::COVERAGE_LEVEL;

    #[test]
    fn boundaries_match_density_times_cell_area() {
        // cell area = 41252.96125 / 49152 = 0.8392876 deg²
        let c = cell_cum_counts();
        assert_eq!(c, [0, 420, 1679, 4196, 6714]);
    }

    #[test]
    fn slice_select_returns_the_band() {
        let v: Vec<u32> = (0..100).collect();
        assert_eq!(slice_select(&v, 10, 30), &v[10..30]);
    }

    #[test]
    fn slice_select_clamps_to_len_and_is_disjoint() {
        let v: Vec<u32> = (0..15).collect();
        // band beyond the data → empty; bands tile without overlap
        assert_eq!(slice_select(&v, 20, 40), &[] as &[u32]);
        let a = slice_select(&v, 0, 10);
        let b = slice_select(&v, 10, 40); // clamps hi to 15
        assert_eq!(a.len() + b.len(), v.len());
        assert_eq!(b, &v[10..15]);
    }

    fn names(n: &[&str]) -> Vec<String> {
        n.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_coverage_record_decides_what_the_build_warns_about() {
        assert_eq!(coverage_gap(None), Some(CoverageGap::NoRecord));

        // Pixels 0..=15 are level-6 cell 0, whole.
        let whole = Coverage::from_names(&names(&["GaiaSource_000000-000015.csv.gz"]));
        assert_eq!(whole.level, COVERAGE_LEVEL);
        assert_eq!(coverage_gap(Some(&whole)), None);

        // Pixels 0..=20 reach cell 1 in part.
        let part = Coverage::from_names(&names(&["GaiaSource_000000-000020.csv.gz"]));
        assert_eq!(coverage_gap(Some(&part)), Some(CoverageGap::PartialCells(1)));

        let unknown = Coverage::from_names(&names(&["GaiaSource_000000-000015.csv.gz", "other.csv.gz"]));
        assert_eq!(coverage_gap(Some(&unknown)), Some(CoverageGap::UnknownFiles));
    }
}
