//! Which parts of the sky a set of Gaia files covered.
//!
//! A `GaiaSource_FROM-TO.csv.gz` file holds every source whose level-8 nested
//! HEALPix pixel (the source id divided by 2^43) lies in `FROM..=TO`. The
//! ingest records the ranges next to the bins, so a later tier build can tell a
//! level-6 cell that is whole from one that only some of the files reached.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::COVERAGE_FILE;
use crate::healpix;

/// The level the file names count in.
pub const COVERAGE_LEVEL: u8 = 8;

/// Level-8 pixels in one level-6 cell (`4^(8-6)`).
const PIXELS_PER_CELL: u64 = 16;

/// One file with a range in its name: level-8 pixels `from..=to`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoveredFile {
    pub name: String,
    pub from: u64,
    pub to: u64,
}

/// The coverage record (`coverage.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coverage {
    /// The HEALPix level of the `from`/`to` ranges: 8.
    pub level: u8,
    /// Files whose name carries a valid range, ordered by range.
    pub files: Vec<CoveredFile>,
    /// Files whose name carries none (or an impossible one), sorted: while any
    /// exists, no cell can be called complete.
    pub unknown: Vec<String>,
}

/// `GaiaSource_000000-000007.csv.gz` to `(0, 7)`; `None` for any other shape.
fn parse_range(name: &str) -> Option<(u64, u64)> {
    let body = name.strip_prefix("GaiaSource_")?.strip_suffix(".csv.gz")?;
    let (a, b) = body.split_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    if !digits(a) || !digits(b) {
        return None;
    }
    let (from, to): (u64, u64) = (a.parse().ok()?, b.parse().ok()?);
    let n_pixels = 12u64 << (2 * COVERAGE_LEVEL);
    (from <= to && to < n_pixels).then_some((from, to))
}

impl Coverage {
    /// The record for these file names.
    pub fn from_names(names: &[String]) -> Self {
        let mut files = Vec::new();
        let mut unknown = Vec::new();
        for name in names {
            match parse_range(name) {
                Some((from, to)) => files.push(CoveredFile { name: name.clone(), from, to }),
                None => unknown.push(name.clone()),
            }
        }
        files.sort_by(|a, b| (a.from, a.to, &a.name).cmp(&(b.from, b.to, &b.name)));
        unknown.sort();
        Self { level: COVERAGE_LEVEL, files, unknown }
    }

    /// Per level-6 cell, how many of its 16 level-8 pixels the files cover.
    fn covered_counts(&self) -> Vec<u8> {
        let mut counts = vec![0u8; healpix::n_pixels_at_depth(healpix::HEALPIX_DEPTH) as usize];
        // Merge the ranges first, so overlapping files do not count twice.
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for f in &self.files {
            match merged.last_mut() {
                Some(last) if f.from <= last.1.saturating_add(1) => last.1 = last.1.max(f.to),
                _ => merged.push((f.from, f.to)),
            }
        }
        for (from, to) in merged {
            for cell in from / PIXELS_PER_CELL..=to / PIXELS_PER_CELL {
                let lo = from.max(cell * PIXELS_PER_CELL);
                let hi = to.min(cell * PIXELS_PER_CELL + PIXELS_PER_CELL - 1);
                counts[cell as usize] += (hi - lo + 1) as u8;
            }
        }
        counts
    }

    /// The level-6 cells all of whose level-8 pixels lie in the files' ranges,
    /// ascending. `None` when some file has no range: the record cannot say.
    pub fn complete_cells(&self) -> Option<Vec<u64>> {
        self.cells_where(|n| n == PIXELS_PER_CELL as u8)
    }

    /// The level-6 cells the files reach in part only, ascending. `None` when
    /// some file has no range.
    pub fn partial_cells(&self) -> Option<Vec<u64>> {
        self.cells_where(|n| n > 0 && n < PIXELS_PER_CELL as u8)
    }

    fn cells_where(&self, keep: impl Fn(u8) -> bool) -> Option<Vec<u64>> {
        if !self.unknown.is_empty() {
            return None;
        }
        Some(
            self.covered_counts()
                .iter()
                .enumerate()
                .filter(|(_, n)| keep(**n))
                .map(|(cell, _)| cell as u64)
                .collect(),
        )
    }

    /// Read `coverage.json` from a bins folder. `Ok(None)` when there is none.
    pub fn read(bins_dir: &Path) -> Result<Option<Coverage>> {
        let path = bins_dir.join(COVERAGE_FILE);
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "coverage record unreadable");
                return Err(e).with_context(|| format!("read {}", path.display()));
            }
        };
        let cov: Coverage = serde_json::from_str(&text).map_err(|e| {
            tracing::warn!(path = %path.display(), error = %e, "coverage record malformed");
            anyhow::Error::new(e).context(format!("parse {}", path.display()))
        })?;
        if cov.level != COVERAGE_LEVEL {
            tracing::warn!(path = %path.display(), level = cov.level, "coverage record has an unsupported level");
            anyhow::bail!("{} counts level {}, expected {COVERAGE_LEVEL}", path.display(), cov.level);
        }
        Ok(Some(cov))
    }

    /// Write `coverage.json` into a bins folder (whole file or nothing).
    pub(crate) fn write(&self, bins_dir: &Path) -> Result<()> {
        let path = bins_dir.join(COVERAGE_FILE);
        let tmp = bins_dir.join(format!("{COVERAGE_FILE}.tmp"));
        let text = serde_json::to_string_pretty(self).context("serialize coverage")?;
        std::fs::write(&tmp, text).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, &path).with_context(|| format!("rename to {}", path.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn ranges_parse_from_file_names() {
        assert_eq!(parse_range("GaiaSource_000000-000007.csv.gz"), Some((0, 7)));
        assert_eq!(parse_range("GaiaSource_786097-786431.csv.gz"), Some((786097, 786431)));
        for bad in [
            "mine.csv.gz",
            "GaiaSource_000007-000000.csv.gz",
            "GaiaSource_000000-786432.csv.gz",
            "GaiaSource_+1-5.csv.gz",
            "GaiaSource_1-.csv.gz",
            "GaiaSource_1-5.csv",
        ] {
            assert_eq!(parse_range(bad), None, "{bad}");
        }
    }

    #[test]
    fn cells_are_complete_or_partial_by_their_sixteen_pixels() {
        // 0..=20: cell 0 (pixels 0..=15) whole, cell 1 (16..=31) partial.
        let c = Coverage::from_names(&names(&[
            "GaiaSource_000008-000020.csv.gz",
            "GaiaSource_000000-000007.csv.gz",
        ]));
        assert_eq!(c.level, 8);
        assert_eq!(c.files[0].name, "GaiaSource_000000-000007.csv.gz", "ordered by range");
        assert_eq!(c.complete_cells(), Some(vec![0]));
        assert_eq!(c.partial_cells(), Some(vec![1]));
    }

    #[test]
    fn overlapping_files_do_not_count_twice_and_ends_are_inclusive() {
        // Two files cover 0..=7 each: cell 0 is only half covered.
        let c = Coverage::from_names(&names(&[
            "GaiaSource_000000-000007.csv.gz",
            "GaiaSource_000000-000007.csv.gz",
        ]));
        assert_eq!(c.complete_cells(), Some(vec![]));
        assert_eq!(c.partial_cells(), Some(vec![0]));
        // 15 is the last pixel of cell 0, 16 the first of cell 1.
        let c = Coverage::from_names(&names(&["GaiaSource_000000-000015.csv.gz"]));
        assert_eq!(c.complete_cells(), Some(vec![0]));
        assert_eq!(c.partial_cells(), Some(vec![]));
    }

    #[test]
    fn the_whole_sky_is_all_cells_and_the_last_range_ends_at_the_last_pixel() {
        let c = Coverage::from_names(&names(&["GaiaSource_000000-786431.csv.gz"]));
        let all = c.complete_cells().unwrap();
        assert_eq!(all.len() as u64, healpix::n_pixels_at_depth(6));
        assert_eq!(all.last().copied(), Some(49151));
    }

    #[test]
    fn a_file_without_a_range_makes_the_record_unknown() {
        let c = Coverage::from_names(&names(&["GaiaSource_000000-000015.csv.gz", "mine.csv.gz"]));
        assert_eq!(c.unknown, vec!["mine.csv.gz".to_string()]);
        assert_eq!(c.complete_cells(), None);
        assert_eq!(c.partial_cells(), None);
    }

    #[test]
    fn write_then_read_round_trips_and_a_missing_record_is_none() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(Coverage::read(tmp.path()).unwrap(), None);
        let c = Coverage::from_names(&names(&["GaiaSource_000000-000015.csv.gz", "mine.csv.gz"]));
        c.write(tmp.path()).unwrap();
        assert_eq!(Coverage::read(tmp.path()).unwrap(), Some(c));
        assert!(!tmp.path().join("coverage.json.tmp").exists());
        std::fs::write(tmp.path().join(COVERAGE_FILE), "{").unwrap();
        assert!(Coverage::read(tmp.path()).is_err());
    }
}
