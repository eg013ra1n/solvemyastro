//! Gaia DR3 ingest from a folder: the `GaiaSource_*.csv.gz` files the user
//! downloaded become a folder of HEALPix bins ([`crate::bins`]), plus a
//! coverage record naming which parts of the sky the files covered and an
//! optional MD5 check against the archive's `_MD5SUM.txt`.
//!
//! The raw files are the long-term archive; the bins can be rebuilt from them
//! at any time (another magnitude limit, another RUWE rule) without a new
//! download. Nothing here downloads.
//!
//! The output does not depend on file order or thread count: every bin is
//! written through [`crate::bins::write_records`], which sorts by a total key.

pub mod binner;
pub mod coverage;
pub mod ingest;
pub mod md5_list;

pub use binner::HealpixBinner;
pub use coverage::{Coverage, CoveredFile};
pub use ingest::{ingest, IngestOptions, IngestProgress, IngestSummary};

/// RUWE (renormalised unit weight error) ceiling, applied locally at bin
/// time. A Gaia DR3 source with RUWE >= 1.4 has an unreliable astrometric
/// solution (typically an unresolved binary, so the catalog position itself is
/// biased) and is dropped at ingest. A row with no RUWE (a 2-parameter
/// solution has none by construction) is kept: a missing RUWE is not a quality
/// failure.
pub const GAIA_RUWE_MAX: f64 = 1.4;

/// Default magnitude ceiling: a star at or past this G is dropped.
pub const DEFAULT_MAG_LIMIT: f32 = 21.0;

/// Default number of files read in parallel.
pub const DEFAULT_JOBS: usize = 4;

/// The archive's checksum list, kept next to the data files.
pub const MD5_LIST_FILE: &str = "_MD5SUM.txt";

/// The coverage record inside a bins folder; written last, after the bins.
pub const COVERAGE_FILE: &str = "coverage.json";
