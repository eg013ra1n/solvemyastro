//! Ingest a folder of `GaiaSource_*.csv.gz` files into HEALPix bins.
//!
//! Each file is stream-gunzipped, the six columns that matter are projected out
//! of the ~150-column row, the row filters apply, and the records go to the
//! [`HealpixBinner`]. The output depends on neither the order of the files nor
//! the number of worker threads.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Mutex};

use anyhow::{anyhow, Context, Result};
use flate2::read::GzDecoder;

use super::binner::HealpixBinner;
use super::coverage::Coverage;
use super::md5_list::{md5_of_file, read_md5_list};
use super::{COVERAGE_FILE, DEFAULT_JOBS, DEFAULT_MAG_LIMIT, GAIA_RUWE_MAX, MD5_LIST_FILE};
use crate::bins::BinRecord;

/// Records handed from a worker to the binner in one message. Big enough that
/// the channel cost is amortised, small enough that a worker never sits on much
/// memory waiting for the consumer.
const INGEST_CHUNK_SIZE: usize = 64 * 1024;

/// The scratch folder inside the output folder.
const SCRATCH_DIR: &str = ".scratch";

/// How an ingest runs.
#[derive(Clone, Debug)]
pub struct IngestOptions {
    /// A star at or past this G magnitude is dropped.
    pub mag_limit: f32,
    /// Files read (and checked) in parallel.
    pub jobs: usize,
    /// Check every file against `_MD5SUM.txt` before reading any. A missing
    /// list is then an error.
    pub verify_md5: bool,
}

impl Default for IngestOptions {
    fn default() -> Self {
        Self { mag_limit: DEFAULT_MAG_LIMIT, jobs: DEFAULT_JOBS, verify_md5: false }
    }
}

/// What the ingest reports while it runs.
#[derive(Clone, Debug)]
pub enum IngestProgress {
    /// Once, after the files are listed.
    Started { total_files: usize },
    /// One file matched its checksum.
    Verified { name: String },
    /// One file fully read.
    FileDone { name: String, completed: usize, total_files: usize, stars_kept: usize },
    /// The scratch is being turned into bins.
    Finalizing,
    /// The bins and the coverage record are written.
    Complete { total_stars: usize },
}

/// What an ingest produced.
#[derive(Clone, Debug)]
pub struct IngestSummary {
    pub files: usize,
    pub stars: usize,
    pub coverage: Coverage,
}

/// Ingest every `*.csv.gz` file of `gaia_dir` into `out_bins`.
///
/// The files are taken in the order of `_MD5SUM.txt` (those present) when the
/// list exists, else sorted by name. `out_bins` must not already hold bins or a
/// coverage record. `<out_bins>/.scratch/` is cleared at the start and removed
/// at the end. `coverage.json` is written last: a folder without one is not a
/// finished ingest. Any file error stops the run with the file's name in it.
pub fn ingest(
    gaia_dir: &Path,
    out_bins: &Path,
    opts: &IngestOptions,
    progress: &(dyn Fn(IngestProgress) + Sync),
) -> Result<IngestSummary> {
    match run(gaia_dir, out_bins, opts, progress) {
        Ok(summary) => {
            tracing::info!(
                files = summary.files,
                stars = summary.stars,
                out = %out_bins.display(),
                "gaia ingest complete"
            );
            Ok(summary)
        }
        Err(e) => {
            tracing::warn!(
                gaia_dir = %gaia_dir.display(),
                out = %out_bins.display(),
                error = %format!("{e:#}"),
                "gaia ingest failed"
            );
            Err(e)
        }
    }
}

/// A file to read, with the checksum the list gives it.
struct Input {
    name: String,
    path: PathBuf,
    md5_hex: Option<String>,
}

/// The first error of a parallel stage, and the flag that stops the rest.
struct FirstError {
    slot: Mutex<Option<anyhow::Error>>,
    stop: AtomicBool,
}

impl FirstError {
    fn new() -> Self {
        Self { slot: Mutex::new(None), stop: AtomicBool::new(false) }
    }

    fn record(&self, e: anyhow::Error) {
        let mut slot = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        if slot.is_none() {
            *slot = Some(e);
        }
        self.stop.store(true, Ordering::Relaxed);
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn take(&self) -> Option<anyhow::Error> {
        self.slot.lock().unwrap_or_else(|p| p.into_inner()).take()
    }
}

fn run(
    gaia_dir: &Path,
    out_bins: &Path,
    opts: &IngestOptions,
    progress: &(dyn Fn(IngestProgress) + Sync),
) -> Result<IngestSummary> {
    std::fs::create_dir_all(out_bins).with_context(|| format!("create {}", out_bins.display()))?;
    refuse_filled_output(out_bins)?;

    let scratch = out_bins.join(SCRATCH_DIR);
    remove_scratch(&scratch)?;
    std::fs::create_dir_all(&scratch).with_context(|| format!("create scratch {}", scratch.display()))?;

    let result = work(gaia_dir, out_bins, &scratch, opts, progress);
    if let Err(e) = remove_scratch(&scratch) {
        if result.is_ok() {
            return Err(e);
        }
        tracing::warn!(error = %format!("{e:#}"), "gaia scratch not removed after a failed ingest");
    }
    result
}

fn work(
    gaia_dir: &Path,
    out_bins: &Path,
    scratch: &Path,
    opts: &IngestOptions,
    progress: &(dyn Fn(IngestProgress) + Sync),
) -> Result<IngestSummary> {
    let inputs = list_inputs(gaia_dir, opts.verify_md5)?;
    let total_files = inputs.len();
    let jobs = opts.jobs.max(1);
    progress(IngestProgress::Started { total_files });

    if opts.verify_md5 {
        verify_all(&inputs, jobs, progress)?;
    }

    let binner = HealpixBinner::open(scratch)?;
    read_all(&inputs, jobs, opts.mag_limit, &binner, progress)?;

    progress(IngestProgress::Finalizing);
    let stars = binner.finalize(out_bins)?;

    let names: Vec<String> = inputs.iter().map(|i| i.name.clone()).collect();
    let coverage = Coverage::from_names(&names);
    remove_scratch(scratch)?;
    // Last: a folder with a coverage record is a finished ingest.
    coverage.write(out_bins)?;
    progress(IngestProgress::Complete { total_stars: stars });
    Ok(IngestSummary { files: total_files, stars, coverage })
}

fn refuse_filled_output(out_bins: &Path) -> Result<()> {
    for entry in std::fs::read_dir(out_bins).with_context(|| format!("list {}", out_bins.display()))? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if name == COVERAGE_FILE || (name.starts_with("healpix_") && name.ends_with(".bin")) {
            anyhow::bail!("output folder already holds bins: {}; choose an empty folder", out_bins.display());
        }
    }
    Ok(())
}

fn remove_scratch(scratch: &Path) -> Result<()> {
    match std::fs::remove_dir_all(scratch) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("remove scratch {}", scratch.display())),
    }
}

/// The files to read: the list's order (those present) when `_MD5SUM.txt`
/// exists, else every `*.csv.gz` sorted by name.
fn list_inputs(gaia_dir: &Path, verify_md5: bool) -> Result<Vec<Input>> {
    let list_path = gaia_dir.join(MD5_LIST_FILE);
    let inputs: Vec<Input> = if list_path.is_file() {
        let listed = read_md5_list(&list_path)?;
        let present: Vec<Input> = listed
            .into_iter()
            .map(|f| Input { path: gaia_dir.join(&f.name), name: f.name, md5_hex: Some(f.md5_hex) })
            .filter(|i| i.path.is_file())
            .collect();
        let unlisted = csv_gz_names(gaia_dir)?
            .into_iter()
            .filter(|n| !present.iter().any(|i| &i.name == n))
            .count();
        if unlisted > 0 {
            tracing::warn!(unlisted, list = MD5_LIST_FILE, "gaia files not in the checksum list are not ingested");
        }
        present
    } else {
        if verify_md5 {
            anyhow::bail!("cannot verify checksums: {} has no {MD5_LIST_FILE}", gaia_dir.display());
        }
        csv_gz_names(gaia_dir)?
            .into_iter()
            .map(|name| Input { path: gaia_dir.join(&name), name, md5_hex: None })
            .collect()
    };
    if inputs.is_empty() {
        anyhow::bail!("no .csv.gz files in {}", gaia_dir.display());
    }
    Ok(inputs)
}

/// Sorted names of the `*.csv.gz` files in a folder.
fn csv_gz_names(dir: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("list {}", dir.display()))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".csv.gz") && entry.path().is_file() {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// Check every file against its listed checksum, `jobs` at a time.
fn verify_all(inputs: &[Input], jobs: usize, progress: &(dyn Fn(IngestProgress) + Sync)) -> Result<()> {
    let next = AtomicUsize::new(0);
    let first_err = FirstError::new();
    std::thread::scope(|s| {
        for _ in 0..jobs {
            s.spawn(|| loop {
                if first_err.stopped() {
                    break;
                }
                let Some(input) = inputs.get(next.fetch_add(1, Ordering::Relaxed)) else {
                    break;
                };
                match check_one(input) {
                    Ok(()) => progress(IngestProgress::Verified { name: input.name.clone() }),
                    Err(e) => first_err.record(e),
                }
            });
        }
    });
    match first_err.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

fn check_one(input: &Input) -> Result<()> {
    let Some(listed) = &input.md5_hex else {
        return Err(anyhow!("no checksum listed for {}", input.name));
    };
    let got = md5_of_file(&input.path).with_context(|| format!("checksum {}", input.name))?;
    if &got != listed {
        anyhow::bail!("md5 mismatch for {}: listed {listed}, file {got}", input.name);
    }
    Ok(())
}

/// Read every file, `jobs` at a time, into the binner.
fn read_all(
    inputs: &[Input],
    jobs: usize,
    mag_limit: f32,
    binner: &HealpixBinner,
    progress: &(dyn Fn(IngestProgress) + Sync),
) -> Result<()> {
    let total_files = inputs.len();
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let first_err = FirstError::new();
    let (tx, rx) = mpsc::sync_channel::<(String, Vec<BinRecord>)>(jobs * 2);

    std::thread::scope(|s| {
        for _ in 0..jobs {
            let tx = tx.clone();
            let (next, completed, first_err) = (&next, &completed, &first_err);
            s.spawn(move || loop {
                if first_err.stopped() {
                    break;
                }
                let Some(input) = inputs.get(next.fetch_add(1, Ordering::Relaxed)) else {
                    break;
                };
                match ingest_one_file(input, &tx, &first_err.stop, mag_limit) {
                    Ok(kept) => {
                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        tracing::info!(
                            file = %input.name,
                            completed = done,
                            total_files,
                            stars_kept = kept,
                            "gaia file ingested"
                        );
                        progress(IngestProgress::FileDone {
                            name: input.name.clone(),
                            completed: done,
                            total_files,
                            stars_kept: kept,
                        });
                    }
                    Err(e) => first_err.record(e.context(format!("ingest {}", input.name))),
                }
            });
        }
        drop(tx);

        // Consumer: the binner's writes are serialised here. After an error it
        // keeps draining so no worker blocks on a full channel.
        while let Ok((name, recs)) = rx.recv() {
            if first_err.stopped() {
                continue;
            }
            if let Err(e) = binner.push_tile(&recs) {
                first_err.record(e.context(format!("binner push from {name}")));
            }
        }
    });

    match first_err.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Indices of the six columns we keep, from the file's header row.
#[derive(Clone, Copy, Debug)]
struct ColumnIndex {
    ra: usize,
    dec: usize,
    mag: usize,
    pmra: usize,
    pmdec: usize,
    ruwe: usize,
}

impl ColumnIndex {
    fn from_header(header: &str) -> Result<Self> {
        let mut map = HashMap::new();
        for (i, name) in header.split(',').enumerate() {
            map.insert(name.trim().to_ascii_lowercase(), i);
        }
        let get = |k: &str| -> Result<usize> {
            map.get(k).copied().ok_or_else(|| anyhow!("header missing column `{k}` (found: {header})"))
        };
        Ok(Self {
            ra: get("ra")?,
            dec: get("dec")?,
            mag: get("phot_g_mean_mag")?,
            pmra: get("pmra")?,
            pmdec: get("pmdec")?,
            ruwe: get("ruwe")?,
        })
    }

    /// The largest column index we read.
    fn max(&self) -> usize {
        self.ra.max(self.dec).max(self.mag).max(self.pmra).max(self.pmdec).max(self.ruwe)
    }
}

/// Parse one row given the column indices.
///
/// - `ra`, `dec` and `phot_g_mean_mag` must be present and parse as numbers.
/// - `phot_g_mean_mag >= mag_limit` drops the row.
/// - `pmra` / `pmdec` empty or `null` read 0.
/// - `ruwe` present and `>= GAIA_RUWE_MAX` drops the row; empty or `null` keeps it.
fn parse_row(row: &str, idx: ColumnIndex, mag_limit: f32) -> Option<BinRecord> {
    let cap = idx.max() + 1;
    // The file has ~150 columns and we touch 6: stop splitting at the last one
    // we need.
    let mut cols: Vec<&str> = Vec::with_capacity(cap.min(160));
    for (i, c) in row.split(',').enumerate() {
        if i >= cap {
            break;
        }
        cols.push(c);
    }
    if cols.len() <= idx.max() {
        return None;
    }
    let ra: f64 = cols[idx.ra].trim().parse().ok()?;
    let dec: f64 = cols[idx.dec].trim().parse().ok()?;
    let g: f32 = cols[idx.mag].trim().parse().ok()?;
    if !g.is_finite() || g >= mag_limit {
        return None;
    }
    let pm = |s: &str| -> Option<f64> {
        match s.trim() {
            "" | "null" | "NULL" => Some(0.0),
            s => s.parse().ok(),
        }
    };
    let pmra = pm(cols[idx.pmra])?;
    let pmdec = pm(cols[idx.pmdec])?;
    let ruwe_s = cols[idx.ruwe].trim();
    if !ruwe_s.is_empty() && ruwe_s != "null" && ruwe_s != "NULL" {
        let ruwe: f64 = ruwe_s.parse().ok()?;
        if ruwe >= GAIA_RUWE_MAX {
            return None;
        }
    }
    Some(BinRecord::from_values(ra as f32, dec as f32, g, pmra, pmdec))
}

/// Stream-ingest one `.csv.gz` into chunks of records, sent to the binner's
/// consumer. Returns the kept-record count.
fn ingest_one_file(
    input: &Input,
    sender: &mpsc::SyncSender<(String, Vec<BinRecord>)>,
    stop: &AtomicBool,
    mag_limit: f32,
) -> Result<usize> {
    let path = &input.path;
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let gz = GzDecoder::new(BufReader::with_capacity(1 << 20, f));
    let reader = BufReader::with_capacity(1 << 20, gz);

    // Skip the leading `#` preamble (copyright and citation), then take the
    // next non-empty line as the header.
    let mut lines = reader.lines();
    let mut header: Option<String> = None;
    for line in &mut lines {
        let l = line.with_context(|| format!("read header in {}", path.display()))?;
        let t = l.trim_start();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        header = Some(l);
        break;
    }
    let header = header.ok_or_else(|| anyhow!("no CSV header in {}", path.display()))?;
    let idx = ColumnIndex::from_header(&header).with_context(|| format!("parse header of {}", path.display()))?;

    let mut chunk: Vec<BinRecord> = Vec::with_capacity(INGEST_CHUNK_SIZE);
    let mut kept = 0usize;
    for line in lines {
        if stop.load(Ordering::Relaxed) {
            anyhow::bail!("stopped");
        }
        let l = line.with_context(|| format!("read row in {}", path.display()))?;
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if let Some(rec) = parse_row(&l, idx, mag_limit) {
            chunk.push(rec);
            kept += 1;
            if chunk.len() >= INGEST_CHUNK_SIZE {
                let to_send = std::mem::replace(&mut chunk, Vec::with_capacity(INGEST_CHUNK_SIZE));
                if sender.send((input.name.clone(), to_send)).is_err() {
                    anyhow::bail!("binner consumer hung up");
                }
            }
        }
    }
    if !chunk.is_empty() && sender.send((input.name.clone(), chunk)).is_err() {
        anyhow::bail!("binner consumer hung up");
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn column_index_resolves_known_header() {
        let header = "solution_id,designation,source_id,ref_epoch,ra,ra_error,dec,dec_error,parallax,pmra,pmra_error,pmdec,pmdec_error,ruwe,phot_g_mean_mag,phot_g_mean_flux";
        let idx = ColumnIndex::from_header(header).unwrap();
        assert_eq!(idx.ra, 4);
        assert_eq!(idx.dec, 6);
        assert_eq!(idx.pmra, 9);
        assert_eq!(idx.pmdec, 11);
        assert_eq!(idx.ruwe, 13);
        assert_eq!(idx.mag, 14);
        // A header missing a needed column is a hard error.
        assert!(ColumnIndex::from_header("solution_id,ra,dec,pmra,pmdec,ruwe").is_err());
    }

    #[test]
    fn parse_row_applies_filters() {
        let idx = ColumnIndex::from_header("ra,dec,phot_g_mean_mag,pmra,pmdec,ruwe").unwrap();
        let s = parse_row("86.682,0.042,14.231,12.5,-7.25,1.05", idx, 19.0).expect("kept");
        assert!((s.ra - 86.682).abs() < 1e-3);
        assert!((s.mag() - 14.231).abs() < 1e-3);
        // RUWE at the cap and above is dropped; empty RUWE (a 2-parameter solution) is kept.
        assert!(parse_row("86.682,0.042,14.231,12.5,-7.25,1.4", idx, 19.0).is_none());
        assert!(parse_row("86.682,0.042,14.231,12.5,-7.25,3.7", idx, 19.0).is_none());
        assert!(parse_row("86.682,0.042,14.231,12.5,-7.25,", idx, 19.0).is_some());
        // Empty proper motion reads 0.
        let s2 = parse_row("10,20,15.9,,,", idx, 19.0).expect("empty pm kept");
        assert_eq!(s2.pmra_mas_yr(), 0.0);
        assert_eq!(s2.pmdec_mas_yr(), 0.0);
        // Past the limit, a NaN magnitude and a short row are dropped.
        assert!(parse_row("10,20,19.5,0,0,1.0", idx, 19.0).is_none());
        assert!(parse_row("10,20,nan,0,0,1.0", idx, 19.0).is_none());
        assert!(parse_row("not,enough", idx, 19.0).is_none());
    }

    #[test]
    fn null_proper_motion_and_ruwe_read_as_missing() {
        let idx = ColumnIndex::from_header("ra,dec,phot_g_mean_mag,pmra,pmdec,ruwe").unwrap();
        let s = parse_row("10,20,15.9,null,NULL,null", idx, 19.0).expect("nulls kept");
        assert_eq!((s.pmra_raw, s.pmdec_raw), (0, 0));
    }

    #[test]
    fn parse_row_honours_the_mag_limit_argument() {
        let idx = ColumnIndex::from_header("ra,dec,phot_g_mean_mag,pmra,pmdec,ruwe").unwrap();
        let row = "10.0,20.0,20.5,0,0,1.0";
        assert!(parse_row(row, idx, 19.0).is_none(), "dropped at limit 19");
        assert!(parse_row(row, idx, 21.0).is_some(), "kept at limit 21");
        assert!(parse_row("10.0,20.0,21.0,0,0,1.0", idx, 21.0).is_none(), "the limit itself is out");
    }

    #[test]
    fn default_options_are_the_documented_ones() {
        let o = IngestOptions::default();
        assert_eq!((o.mag_limit, o.jobs, o.verify_md5), (21.0, 4, false));
    }
}
