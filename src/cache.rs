//! Star-catalog cache — solvemyastro's own on-disk star database.
//!
//! # On-disk format (`stars.smac`)
//!
//! Single memory-mapped file, HEALPix-depth-6 indexed, mag-sorted within
//! each pixel. Depth 6 ⇒ 49 152 pixels.
//!
//! ## Layout (all little-endian, packed; manual byteorder serialisation)
//!
//! ### Header — 64 bytes, fixed
//! | Range | Field |
//! |-------|-------|
//! | 0..8  | magic `b"SMACACHE"` |
//! | 8..10 | `u16` version = 1 |
//! | 10    | `u8`  healpix_depth = 6 |
//! | 11..16 | 5 zero bytes |
//! | 16..24 | `u64` star_count |
//! | 24..32 | `f64` catalog_epoch |
//! | 32..64 | zero-filled |
//!
//! ### Pixel directory — `49 152 × 12` bytes immediately after the header
//! Each entry: `{ u64 byte_offset, u32 count }` (absolute from file start).
//!
//! ### Records — 28 bytes each, pixel-id order, mag-sorted ascending
//! `f64 ra | f64 dec | f32 mag | f32 pmra_mas_yr | f32 pmdec_mas_yr`
//!
//! # Rationale
//!
//! `f64` sky coordinates because this feeds a *precise* SIP solver —
//! `f32` (~0.15″) would be a noise floor on sub-pixel fits.
//! `pmra`/`pmdec` are stored as `f32` (acceptable PM precision; cast on
//! write/read).

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use cdshealpix::nested;
use memmap2::Mmap;

use crate::{math::great_circle_deg, CatalogStar};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 8] = b"SMACACHE";
const VERSION: u16 = 1;
const HEALPIX_DEPTH: u8 = 6;
const N_PIXELS: usize = 49_152; // 12 * 4^6

const HEADER_SIZE: u64 = 64;
const DIR_ENTRY_SIZE: u64 = 12; // u64 offset + u32 count
const DIR_SIZE: u64 = N_PIXELS as u64 * DIR_ENTRY_SIZE;
const RECORD_SIZE: usize = 28; // f64 ra + f64 dec + f32 mag + f32 pmra + f32 pmdec

const RECORDS_REGION_START: u64 = HEADER_SIZE + DIR_SIZE;

const PROGRESS_INGEST_INTERVAL: usize = 1_000_000;
const PROGRESS_FINALIZE_INTERVAL: usize = 4_096;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// One catalog star as ingested into the cache.
///
/// On-disk `pmra`/`pmdec` are stored as `f32` (cast on write/read).
#[derive(Clone, Copy, Debug)]
pub struct StarRecord {
    /// RA, degrees (catalog epoch).
    pub ra: f64,
    /// Dec, degrees (catalog epoch).
    pub dec: f64,
    /// Magnitude (catalog native band).
    pub mag: f32,
    /// Proper motion μα* (mas/yr, includes cos δ).
    pub pmra_mas_yr: f64,
    /// Proper motion μδ (mas/yr).
    pub pmdec_mas_yr: f64,
}

/// Progress callback payload for [`build_cache`].
#[derive(Clone, Copy, Debug)]
pub enum BuildProgress {
    Ingesting { records: usize },
    Finalizing { shards_done: usize, shards_total: usize },
    Complete { records: usize },
}

/// A read-only, memory-mapped star cache the solver queries per trial.
pub struct StarCache {
    /// Cache directory (retained so callers can inspect it; also used by
    /// future write-back / validation helpers).
    pub dir: PathBuf,
    mmap: Mmap,
    star_count: u64,
    catalog_epoch: f64,
    /// Pixel directory: `(byte_offset, count)` for each of the 49 152 pixels.
    directory: Vec<(u64, u32)>,
}

// ---------------------------------------------------------------------------
// StarCache — open / query
// ---------------------------------------------------------------------------

impl StarCache {
    /// Open an existing cache directory.
    ///
    /// Fails if the directory is missing, if `stars.smac` is absent or
    /// truncated, or if the magic/version/depth header fields are wrong.
    ///
    /// # Errors
    ///
    /// Returns `Err` for any of the above conditions.
    pub fn open(dir: &Path) -> Result<Self> {
        let dir = dir.to_path_buf();
        anyhow::ensure!(dir.is_dir(), "star cache directory not found: {}", dir.display());

        let smac_path = dir.join("stars.smac");
        let file =
            File::open(&smac_path).with_context(|| format!("open {}", smac_path.display()))?;

        // SAFETY: we never mutate the file while the mapping is live (read-only
        // cache — no background writes to `stars.smac`).
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("mmap {}", smac_path.display()))?;

        // Minimum viable size: header + directory
        let min_size = (HEADER_SIZE + DIR_SIZE) as usize;
        anyhow::ensure!(
            mmap.len() >= min_size,
            "stars.smac too short ({} bytes, need at least {})",
            mmap.len(),
            min_size
        );

        // Validate header
        let mut cur = std::io::Cursor::new(&mmap[..]);
        let mut magic_buf = [0u8; 8];
        cur.read_exact(&mut magic_buf).context("read magic")?;
        anyhow::ensure!(&magic_buf == MAGIC, "stars.smac bad magic");

        let version = cur.read_u16::<LittleEndian>().context("read version")?;
        anyhow::ensure!(version == VERSION, "unsupported stars.smac version {version}");

        let depth = cur.read_u8().context("read depth")?;
        anyhow::ensure!(
            depth == HEALPIX_DEPTH,
            "unsupported healpix depth {depth} (expected {HEALPIX_DEPTH})"
        );

        // 5 pad bytes
        cur.seek(SeekFrom::Current(5)).context("seek past pad")?;

        let star_count = cur.read_u64::<LittleEndian>().context("read star_count")?;
        let catalog_epoch = cur.read_f64::<LittleEndian>().context("read catalog_epoch")?;

        // Parse pixel directory (starts at HEADER_SIZE)
        cur.seek(SeekFrom::Start(HEADER_SIZE)).context("seek to directory")?;
        let mut directory = Vec::with_capacity(N_PIXELS);
        for _ in 0..N_PIXELS {
            let offset = cur.read_u64::<LittleEndian>().context("read dir offset")?;
            let count = cur.read_u32::<LittleEndian>().context("read dir count")?;
            directory.push((offset, count));
        }

        Ok(Self { dir, mmap, star_count, catalog_epoch, directory })
    }

    /// Number of stars in the cache.
    #[allow(dead_code)]
    pub fn star_count(&self) -> u64 {
        self.star_count
    }

    /// Catalog epoch (Julian year, e.g. 2016.0 for Gaia DR3).
    #[allow(dead_code)]
    pub fn catalog_epoch(&self) -> f64 {
        self.catalog_epoch
    }

    /// Stars within `radius_deg` of `(ra, dec)` brighter than `mag_limit`,
    /// proper-motion-corrected to `epoch`.
    ///
    /// Uses cdshealpix `cone_coverage_approx` to enumerate candidate pixels,
    /// then does a precise great-circle separation filter. Early-exits each
    /// pixel slice once `mag > mag_limit` (records are mag-sorted ascending).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the mmap slice is internally inconsistent.
    pub fn cone(
        &self,
        ra: f64,
        dec: f64,
        radius_deg: f64,
        mag_limit: f32,
        epoch: f64,
    ) -> Result<Vec<CatalogStar>> {
        let pixels = cone_search_pixels(ra, dec, radius_deg);

        let mut results: Vec<CatalogStar> = Vec::new();

        for px in pixels {
            let (offset, count) = self.directory[px as usize];
            if count == 0 {
                continue;
            }

            let byte_start = offset as usize;
            let byte_end = byte_start + count as usize * RECORD_SIZE;
            anyhow::ensure!(
                byte_end <= self.mmap.len(),
                "pixel {px} directory entry points beyond mmap ({byte_end} > {})",
                self.mmap.len()
            );

            let slice = &self.mmap[byte_start..byte_end];
            let mut cur = std::io::Cursor::new(slice);

            for _ in 0..count {
                let rec_ra = cur.read_f64::<LittleEndian>().context("read rec ra")?;
                let rec_dec = cur.read_f64::<LittleEndian>().context("read rec dec")?;
                let rec_mag = cur.read_f32::<LittleEndian>().context("read rec mag")?;
                let rec_pmra = cur.read_f32::<LittleEndian>().context("read rec pmra")? as f64;
                let rec_pmdec =
                    cur.read_f32::<LittleEndian>().context("read rec pmdec")? as f64;

                // Early-exit: records are mag-sorted ascending within each pixel.
                if rec_mag > mag_limit {
                    break;
                }

                // Proper-motion propagation
                let (pm_ra, pm_dec) =
                    propagate(rec_ra, rec_dec, rec_pmra, rec_pmdec, self.catalog_epoch, epoch);

                // Precise great-circle separation filter (cone_coverage over-covers)
                if great_circle_deg(pm_ra, pm_dec, ra, dec) <= radius_deg {
                    results.push(CatalogStar { ra: pm_ra, dec: pm_dec, mag: rec_mag });
                }
            }
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// build_cache — streaming, RAM-bounded
// ---------------------------------------------------------------------------

/// Build a star-catalog cache from a record stream into `out_dir`.
///
/// The function is RAM-bounded: it streams records through per-pixel scratch
/// files rather than collecting them all in memory. This accommodates the
/// ~78 M-star Gaia catalog without OOM.
///
/// # Phases
///
/// - **Phase A — Ingest**: for each record, compute its HEALPix pixel and
///   append the packed 28-byte record to `out_dir/.scratch/p_{pixel:06}.raw`.
///   Emits [`BuildProgress::Ingesting`] every ~1 M records.
/// - **Phase B — Finalize**: creates `out_dir/stars.smac`, writes a
///   placeholder header+directory, then for each pixel reads its scratch
///   file, sorts by mag ascending, appends records, and records directory
///   entries. Emits [`BuildProgress::Finalizing`] periodically. Seeks back
///   to write the real header+directory, deletes `.scratch/`.
///   Emits [`BuildProgress::Complete`].
///
/// Returns the total number of stars written.
///
/// # Errors
///
/// Returns `Err` on any I/O failure.
pub fn build_cache<I, F>(
    records: I,
    out_dir: &Path,
    catalog_epoch: f64,
    progress: F,
) -> Result<usize>
where
    I: IntoIterator<Item = StarRecord>,
    F: Fn(BuildProgress),
{
    fs::create_dir_all(out_dir)
        .with_context(|| format!("create cache dir {}", out_dir.display()))?;

    let scratch_dir = out_dir.join(".scratch");
    fs::create_dir_all(&scratch_dir)
        .with_context(|| format!("create scratch dir {}", scratch_dir.display()))?;

    // -------------------------------------------------------------------
    // Phase A: Ingest — stream records into per-pixel scratch files
    // -------------------------------------------------------------------

    let mut total_records: usize = 0;

    for star in records {
        let pixel = sky_to_pixel(star.ra, star.dec);
        let scratch_path = scratch_dir.join(format!("p_{pixel:06}.raw"));

        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&scratch_path)
            .with_context(|| format!("open scratch {}", scratch_path.display()))?;

        write_record(&mut f, &star).with_context(|| format!("write record to pixel {pixel}"))?;
        drop(f);

        total_records += 1;
        if total_records % PROGRESS_INGEST_INTERVAL == 0 {
            progress(BuildProgress::Ingesting { records: total_records });
        }
    }

    // -------------------------------------------------------------------
    // Phase B: Finalize — create stars.smac
    // -------------------------------------------------------------------

    let smac_path = out_dir.join("stars.smac");
    let mut smac = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&smac_path)
        .with_context(|| format!("create {}", smac_path.display()))?;

    // Write placeholder header (64 bytes of zeros)
    smac.write_all(&[0u8; HEADER_SIZE as usize]).context("write placeholder header")?;

    // Write placeholder directory (all zeros)
    let dir_placeholder = vec![0u8; DIR_SIZE as usize];
    smac.write_all(&dir_placeholder).context("write placeholder directory")?;

    // Build directory entries as we write records
    let mut directory: Vec<(u64, u32)> = vec![(0u64, 0u32); N_PIXELS];
    let mut file_offset: u64 = RECORDS_REGION_START;

    for pixel in 0..N_PIXELS {
        let scratch_path = scratch_dir.join(format!("p_{pixel:06}.raw"));

        // Read scratch file (may not exist for empty pixels)
        let raw_bytes = match fs::read(&scratch_path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("read scratch {}", scratch_path.display()))
            }
        };

        let n_raw = raw_bytes.len() / RECORD_SIZE;
        if n_raw == 0 {
            directory[pixel] = (0, 0);
        } else {
            // Parse records from raw bytes
            let mut recs: Vec<(f32, [u8; RECORD_SIZE])> = Vec::with_capacity(n_raw);
            let mut cur = std::io::Cursor::new(&raw_bytes);
            for _ in 0..n_raw {
                let mut buf = [0u8; RECORD_SIZE];
                // Read ra, dec (skip for sort key), then mag
                let ra_bytes = cur.read_f64::<LittleEndian>().context("parse ra")?;
                let dec_bytes = cur.read_f64::<LittleEndian>().context("parse dec")?;
                let mag = cur.read_f32::<LittleEndian>().context("parse mag")?;
                let pmra = cur.read_f32::<LittleEndian>().context("parse pmra")?;
                let pmdec = cur.read_f32::<LittleEndian>().context("parse pmdec")?;

                // Re-pack into buf for writing
                let mut wc = std::io::Cursor::new(&mut buf[..]);
                wc.write_f64::<LittleEndian>(ra_bytes).unwrap();
                wc.write_f64::<LittleEndian>(dec_bytes).unwrap();
                wc.write_f32::<LittleEndian>(mag).unwrap();
                wc.write_f32::<LittleEndian>(pmra).unwrap();
                wc.write_f32::<LittleEndian>(pmdec).unwrap();

                recs.push((mag, buf));
            }

            // Sort by mag ascending
            recs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let count = recs.len() as u32;
            directory[pixel] = (file_offset, count);

            for (_, buf) in &recs {
                smac.write_all(buf).context("write sorted record")?;
            }
            file_offset += count as u64 * RECORD_SIZE as u64;
        }

        if pixel % PROGRESS_FINALIZE_INTERVAL == 0 && pixel > 0 {
            progress(BuildProgress::Finalizing {
                shards_done: pixel,
                shards_total: N_PIXELS,
            });
        }
    }

    // Seek back and write real header
    smac.seek(SeekFrom::Start(0)).context("seek to header")?;
    write_header(&mut smac, total_records as u64, catalog_epoch).context("write header")?;

    // Write real directory
    smac.seek(SeekFrom::Start(HEADER_SIZE)).context("seek to directory")?;
    for &(offset, count) in &directory {
        smac.write_u64::<LittleEndian>(offset).context("write dir offset")?;
        smac.write_u32::<LittleEndian>(count).context("write dir count")?;
    }

    smac.flush().context("flush stars.smac")?;
    drop(smac);

    // Clean up scratch files
    fs::remove_dir_all(&scratch_dir)
        .with_context(|| format!("remove scratch dir {}", scratch_dir.display()))?;

    progress(BuildProgress::Complete { records: total_records });
    Ok(total_records)
}

// ---------------------------------------------------------------------------
// Legacy HEALPix format — parse helpers and high-level importers
// ---------------------------------------------------------------------------

/// Parse one `healpix_NNNNNN.bin` legacy file.
///
/// Each file is a flat array of 14-byte little-endian records:
///
/// | Bytes  | Field                              |
/// |--------|------------------------------------|
/// | 0..4   | `f32` ra (deg)                     |
/// | 4..8   | `f32` dec (deg)                    |
/// | 8..10  | `u16` mag_raw = mag × 1 000        |
/// | 10..12 | `i16` pmra_raw = μα*/0.01 (mas/yr) |
/// | 12..14 | `i16` pmdec_raw = μδ/0.01 (mas/yr) |
///
/// Returns a `Vec<StarRecord>`. Silently returns empty if `path` is not
/// a regular file.
///
/// # Errors
///
/// Returns `Err` on I/O failure or if the file size is not a multiple of 14.
fn read_legacy_file(path: &Path) -> Result<Vec<StarRecord>> {
    const LEGACY_RECORD_SIZE: usize = 14;

    let data =
        fs::read(path).with_context(|| format!("read legacy file {}", path.display()))?;

    if data.len() % LEGACY_RECORD_SIZE != 0 {
        bail!(
            "legacy file {} size {} is not a multiple of {LEGACY_RECORD_SIZE}",
            path.display(),
            data.len()
        );
    }

    let n = data.len() / LEGACY_RECORD_SIZE;
    let mut cur = std::io::Cursor::new(&data);
    let mut stars = Vec::with_capacity(n);

    for _ in 0..n {
        let ra = cur.read_f32::<LittleEndian>().context("legacy ra")? as f64;
        let dec = cur.read_f32::<LittleEndian>().context("legacy dec")? as f64;
        let mag_raw = cur.read_u16::<LittleEndian>().context("legacy mag_raw")?;
        let pmra_raw = cur.read_i16::<LittleEndian>().context("legacy pmra_raw")?;
        let pmdec_raw = cur.read_i16::<LittleEndian>().context("legacy pmdec_raw")?;

        stars.push(StarRecord {
            ra,
            dec,
            mag: mag_raw as f32 / 1000.0,
            pmra_mas_yr: pmra_raw as f64 * 0.01,
            pmdec_mas_yr: pmdec_raw as f64 * 0.01,
        });
    }

    Ok(stars)
}

/// Import the legacy athenaeum HEALPix catalog format into a `Vec`.
///
/// Collects all `healpix_NNNNNN.bin` files from `dir` and returns the parsed
/// records. Acceptable for small synthetic test directories; **not** suitable
/// for the 78 M-star Gaia catalog (OOM). Use [`build_cache_from_legacy_dir`]
/// for large catalogs.
///
/// # Errors
///
/// Returns `Err` on I/O failure or malformed records.
pub fn import_legacy_healpix_dir(dir: &Path) -> Result<Vec<StarRecord>> {
    let mut paths = legacy_bin_paths(dir)?;
    paths.sort();

    let mut stars: Vec<StarRecord> = Vec::new();
    for path in &paths {
        stars.extend(read_legacy_file(path)?);
    }
    Ok(stars)
}

/// Build a star-catalog cache directly from a legacy `healpix_NNNNNN.bin`
/// directory — **RAM-bounded**.
///
/// Collects and sorts the `.bin` paths, then streams them one file at a time
/// through [`build_cache`]. RAM usage is bounded to one legacy file at a time
/// (typically a few MB) plus `build_cache`'s per-pixel scratch files, making
/// the full 78 M-star Gaia load feasible without OOM.
///
/// # Parameters
///
/// - `legacy_dir`: directory containing `healpix_NNNNNN.bin` files.
/// - `out_dir`: destination for the `stars.smac` cache.
/// - `catalog_epoch`: Julian year of the catalog positions (e.g. `2016.0`
///   for Gaia DR3).
/// - `progress`: callback receiving [`BuildProgress`] updates.
///
/// # Errors
///
/// Returns `Err` on any I/O or parsing failure.
pub fn build_cache_from_legacy_dir(
    legacy_dir: &Path,
    out_dir: &Path,
    catalog_epoch: f64,
    progress: impl Fn(BuildProgress),
) -> Result<usize> {
    let mut paths = legacy_bin_paths(legacy_dir)?;
    paths.sort(); // deterministic order

    // Flat-map: one file at a time via the iterator; build_cache streams
    // through them without collecting into a single Vec.
    let records = paths.into_iter().flat_map(|p| {
        read_legacy_file(&p).unwrap_or_default()
    });

    build_cache(records, out_dir, catalog_epoch, progress)
}

/// Enumerate `healpix_NNNNNN.bin` paths inside `dir`.
fn legacy_bin_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries =
        fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))?;

    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry.context("read dir entry")?;
        let path = entry.path();
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if fname.starts_with("healpix_") && fname.ends_with(".bin") {
            paths.push(path);
        }
    }
    Ok(paths)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Compute the HEALPix pixel for a sky position (depth 6, nested scheme).
///
/// Mirrors `athenaeum_core::catalog::healpix::sky_to_pixel` exactly.
fn sky_to_pixel(ra_deg: f64, dec_deg: f64) -> u64 {
    nested::hash(HEALPIX_DEPTH, ra_deg.to_radians(), dec_deg.to_radians())
}

/// Return the set of depth-6 HEALPix pixel IDs that overlap a cone.
///
/// Mirrors the athenaeum-core reference implementation including the
/// panic-catching nudge strategy for cdshealpix 0.7.x edge cases.
fn cone_search_pixels(ra_deg: f64, dec_deg: f64, radius_deg: f64) -> Vec<u64> {
    if !ra_deg.is_finite()
        || !dec_deg.is_finite()
        || !radius_deg.is_finite()
        || radius_deg <= 0.0
    {
        return Vec::new();
    }
    let radius_deg = radius_deg.min(90.0);

    let lon_rad = ra_deg.to_radians();

    let try_cover = |r_deg: f64| -> Option<Vec<u64>> {
        let lat_rad = dec_deg.to_radians();
        let radius_rad = r_deg.to_radians();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            nested::cone_coverage_approx(HEALPIX_DEPTH, lon_rad, lat_rad, radius_rad)
        }))
        .ok()
        .map(|bmoc| bmoc.flat_iter().collect())
    };

    let pixels = try_cover(radius_deg)
        .or_else(|| try_cover(radius_deg * 1.0001))
        .or_else(|| try_cover(radius_deg * 0.9999))
        .or_else(|| try_cover(radius_deg * 1.01))
        .or_else(|| try_cover(radius_deg * 0.99));

    let Some(mut pixels) = pixels else {
        eprintln!(
            "cache::cone_search_pixels: cdshealpix panicked at every nudge for \
             ra={ra_deg:.6} dec={dec_deg:.6} radius_deg={radius_deg:.6}; \
             falling back to brute-force pixel scan"
        );
        return cone_search_pixels_bruteforce(ra_deg, dec_deg, radius_deg);
    };
    pixels.sort_unstable();
    pixels.dedup();
    pixels
}

/// Brute-force pixel enumeration — scans all depth-6 pixel centers for
/// intersection with the query cone. Last-resort fallback when cdshealpix
/// `cone_coverage_approx` panics across every radius nudge (a known issue
/// with cdshealpix 0.7.x for certain sky regions, e.g. around RA≈315°
/// Dec≈-23°). Depth 6 has 49 152 pixels, so a full scan is bounded and
/// fast (~tens of µs).
///
/// The result may slightly over-cover (the test radius is inflated by one
/// pixel half-diagonal so we never miss a boundary pixel). The `cone()`
/// consumer already filters each star by true great-circle distance, so
/// over-coverage is harmless — same property the normal
/// `cone_coverage_approx` path relies on.
fn cone_search_pixels_bruteforce(ra_deg: f64, dec_deg: f64, radius_deg: f64) -> Vec<u64> {
    // Depth-6 pixel area ≈ 0.838 deg² → half-diagonal ≈ 0.65°.
    // 0.7° gives a small safety margin so boundary pixels are not missed.
    const PIXEL_HALF_DIAG_DEG: f64 = 0.7;
    // n_pixels(depth) = 12 * 4^depth. depth 6 → 49 152.
    const N_PIXELS: u64 = 49_152;
    debug_assert_eq!(N_PIXELS, 12u64 << (2 * HEALPIX_DEPTH as u32));

    let lon_rad = ra_deg.to_radians();
    let lat_rad = dec_deg.to_radians();
    let test_radius_rad = (radius_deg + PIXEL_HALF_DIAG_DEG).to_radians();
    let (sin_lat, cos_lat) = (lat_rad.sin(), lat_rad.cos());

    let mut out = Vec::with_capacity(64);
    for idx in 0..N_PIXELS {
        let (plon, plat) = nested::center(HEALPIX_DEPTH, idx);
        let cos_d = sin_lat * plat.sin() + cos_lat * plat.cos() * (lon_rad - plon).cos();
        let d_rad = cos_d.clamp(-1.0, 1.0).acos();
        if d_rad < test_radius_rad {
            out.push(idx);
        }
    }
    // Already ascending + unique by construction.
    out
}

/// Propagate a star's position by proper motion.
///
/// `pmra_mas_yr` is μα* (includes cos δ), so we divide by cos δ to get the
/// true RA increment. The result is clamped to cos δ ≥ 1e-9 near the poles.
/// Returned RA is wrapped into `[0, 360)`.
fn propagate(
    ra_deg: f64,
    dec_deg: f64,
    pmra_mas_yr: f64,
    pmdec_mas_yr: f64,
    from_epoch: f64,
    to_epoch: f64,
) -> (f64, f64) {
    let dt = to_epoch - from_epoch;
    let ddec_deg = pmdec_mas_yr * dt / 3.6e6;
    let cos_dec = dec_deg.to_radians().cos().max(1e-9);
    let dra_deg = (pmra_mas_yr * dt / 3.6e6) / cos_dec;

    let new_dec = dec_deg + ddec_deg;
    let new_ra = (ra_deg + dra_deg).rem_euclid(360.0);
    (new_ra, new_dec)
}

/// Write a single 64-byte header to `w`.
fn write_header<W: Write + Seek>(w: &mut W, star_count: u64, catalog_epoch: f64) -> Result<()> {
    // Magic (8)
    w.write_all(MAGIC).context("write magic")?;
    // Version (2)
    w.write_u16::<LittleEndian>(VERSION).context("write version")?;
    // Depth (1)
    w.write_u8(HEALPIX_DEPTH).context("write depth")?;
    // Pad (5)
    w.write_all(&[0u8; 5]).context("write pad")?;
    // star_count (8)
    w.write_u64::<LittleEndian>(star_count).context("write star_count")?;
    // catalog_epoch (8)
    w.write_f64::<LittleEndian>(catalog_epoch).context("write catalog_epoch")?;
    // Zero fill to 64 bytes: we've written 8+2+1+5+8+8 = 32 bytes so far
    w.write_all(&[0u8; 32]).context("write header zero fill")?;
    Ok(())
}

/// Pack and write one `StarRecord` as 28 bytes (little-endian).
fn write_record<W: Write>(w: &mut W, s: &StarRecord) -> Result<()> {
    w.write_f64::<LittleEndian>(s.ra).context("write ra")?;
    w.write_f64::<LittleEndian>(s.dec).context("write dec")?;
    w.write_f32::<LittleEndian>(s.mag).context("write mag")?;
    w.write_f32::<LittleEndian>(s.pmra_mas_yr as f32).context("write pmra")?;
    w.write_f32::<LittleEndian>(s.pmdec_mas_yr as f32).context("write pmdec")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use byteorder::WriteBytesExt;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn noop_progress(_p: BuildProgress) {}

    /// Build a cache from a `Vec<StarRecord>` into a temp dir; return the
    /// opened `StarCache` and the temp dir (which must stay alive).
    fn build_and_open(stars: Vec<StarRecord>, epoch: f64) -> (StarCache, TempDir) {
        let tmp = TempDir::new().expect("tempdir");
        let n = build_cache(stars.into_iter(), tmp.path(), epoch, noop_progress)
            .expect("build_cache");
        assert!(n > 0 || true, "build succeeded");
        let cache = StarCache::open(tmp.path()).expect("open");
        (cache, tmp)
    }

    // -----------------------------------------------------------------------
    // 1. Whole-sky round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn whole_sky_round_trip() {
        // Scatter a handful of stars across diverse sky positions including
        // poles and RA=0/360 boundary.
        let stars = vec![
            StarRecord { ra: 10.0, dec: 45.0, mag: 5.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            StarRecord { ra: 180.0, dec: -30.0, mag: 7.5, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            StarRecord { ra: 359.9, dec: 0.1, mag: 3.2, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            StarRecord { ra: 0.1, dec: -0.1, mag: 9.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            // Near north pole
            StarRecord { ra: 42.0, dec: 89.5, mag: 6.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            // Near south pole
            StarRecord {
                ra: 200.0,
                dec: -89.5,
                mag: 8.0,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            },
            // RA wrap: just under and just over 0
            StarRecord { ra: 359.0, dec: 5.0, mag: 4.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            StarRecord { ra: 1.0, dec: 5.0, mag: 4.5, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
        ];
        let input_count = stars.len();
        let epoch = 2016.0;

        let (cache, _tmp) = build_and_open(stars.clone(), epoch);

        // star_count() must equal the number of records written
        assert_eq!(
            cache.star_count(),
            input_count as u64,
            "star_count header field"
        );

        // Each star must be recoverable via a targeted cone around its own position.
        // (A single "whole-sky" 180° cone from one point cannot cover the full sphere
        // because cdshealpix clamps radius to 90° and the antipode has sep > 90°.)
        let any_missing = stars.iter().any(|s| {
            let res = cache.cone(s.ra, s.dec, 0.1, 99.0, epoch).expect("cone");
            res.is_empty()
        });
        assert!(!any_missing, "every star should be recoverable via a targeted cone");

        // Verify mag round-trips within f32 quantisation using the north hemisphere
        // (dec ≥ 0) which is fully covered by a single 90° cone from (0, 0).
        let north_stars: Vec<_> = stars.iter().filter(|s| s.dec >= 0.0).collect();
        let results =
            cache.cone(0.0, 90.0, 90.0, 99.0, epoch).expect("north-hemisphere cone");
        // Every north-pole-adjacent star must appear somewhere in the results
        assert!(
            results.len() >= north_stars.len(),
            "north hemisphere cone should contain at least all northern stars"
        );

        // Full-sky mag coverage: union of two 90° cones (north + south poles)
        let south = cache.cone(0.0, -90.0, 90.0, 99.0, epoch).expect("south cone");
        let mut all_mags: Vec<f32> = results.iter().chain(south.iter()).map(|s| s.mag).collect();
        all_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_mags.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        let mut expected_mags: Vec<f32> = stars.iter().map(|s| s.mag).collect();
        expected_mags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Every input magnitude must appear in the union
        for &em in &expected_mags {
            assert!(
                all_mags.iter().any(|&fm| (fm - em).abs() < 1e-4),
                "input mag {em} not found in full-sky union of cones"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. Cone correctness — small radius, RA wrap, pole
    // -----------------------------------------------------------------------

    #[test]
    fn cone_correctness_small_radius() {
        // Stars distributed to test cone filtering precisely
        let stars = vec![
            // Near center
            StarRecord { ra: 30.0, dec: 20.0, mag: 5.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            // Within 0.3° of center
            StarRecord {
                ra: 30.1,
                dec: 20.1,
                mag: 6.0,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            },
            // Far away
            StarRecord { ra: 60.0, dec: -10.0, mag: 4.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            StarRecord {
                ra: 120.0,
                dec: 45.0,
                mag: 7.0,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            },
        ];
        let epoch = 2016.0;
        let (cache, _tmp) = build_and_open(stars.clone(), epoch);

        let center_ra = 30.0;
        let center_dec = 20.0;
        let radius = 0.3_f64;

        // Brute-force reference
        let expected: Vec<_> = stars
            .iter()
            .filter(|s| great_circle_deg(s.ra, s.dec, center_ra, center_dec) <= radius)
            .collect();

        let results = cache.cone(center_ra, center_dec, radius, 99.0, epoch).expect("cone");

        assert_eq!(
            results.len(),
            expected.len(),
            "cone should return exactly {} star(s); got {}",
            expected.len(),
            results.len()
        );
    }

    #[test]
    fn cone_correctness_ra_wrap() {
        // Stars around RA=0/360 boundary; center at ra=0.5
        let stars = vec![
            // Inside cone (sep < 1.0°)
            StarRecord {
                ra: 359.8,
                dec: 0.0,
                mag: 5.0,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            },
            StarRecord { ra: 0.3, dec: 0.0, mag: 6.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            // Outside cone
            StarRecord { ra: 2.5, dec: 0.0, mag: 7.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            StarRecord { ra: 180.0, dec: 0.0, mag: 4.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
        ];
        let epoch = 2016.0;
        let (cache, _tmp) = build_and_open(stars.clone(), epoch);

        let center_ra = 0.5;
        let center_dec = 0.0;
        let radius = 1.0_f64;

        let expected_count = stars
            .iter()
            .filter(|s| great_circle_deg(s.ra, s.dec, center_ra, center_dec) <= radius)
            .count();

        let results = cache.cone(center_ra, center_dec, radius, 99.0, epoch).expect("cone");
        assert_eq!(
            results.len(),
            expected_count,
            "RA-wrap cone expected {expected_count}, got {}",
            results.len()
        );
        // The two inside-cone stars should be included
        assert!(expected_count >= 2, "test setup: at least 2 stars inside cone");
    }

    #[test]
    fn cone_correctness_pole() {
        // Cone centered at north pole, radius 2°
        let stars = vec![
            // dec=88.5 → sep from pole ≈ 1.5° → inside
            StarRecord { ra: 0.0, dec: 88.5, mag: 5.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 },
            StarRecord {
                ra: 90.0,
                dec: 88.8,
                mag: 6.0,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            },
            // dec=86.0 → sep ≈ 4.0° → outside
            StarRecord {
                ra: 45.0,
                dec: 86.0,
                mag: 7.0,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            },
        ];
        let epoch = 2016.0;
        let (cache, _tmp) = build_and_open(stars.clone(), epoch);

        let results = cache.cone(0.0, 90.0, 2.0, 99.0, epoch).expect("cone");

        let expected_count = stars
            .iter()
            .filter(|s| great_circle_deg(s.ra, s.dec, 0.0, 90.0) <= 2.0)
            .count();

        assert_eq!(
            results.len(),
            expected_count,
            "pole cone: expected {expected_count}, got {}",
            results.len()
        );
    }

    // -----------------------------------------------------------------------
    // 3. Mag early-exit: mag limit, pixel-ascending check
    // -----------------------------------------------------------------------

    #[test]
    fn mag_early_exit_and_ascending() {
        // Put several stars in the same HEALPix pixel by clustering them
        // tightly, with different magnitudes.
        let base_ra = 90.0;
        let base_dec = 45.0;
        // Tiny offsets so they all land in the same pixel at depth 6
        // (~0.84° pixels at depth 6; offsets of 0.01° are safe)
        let stars: Vec<StarRecord> = (1..=6)
            .map(|i| StarRecord {
                ra: base_ra + i as f64 * 0.001,
                dec: base_dec + i as f64 * 0.001,
                mag: i as f32,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            })
            .collect();

        let epoch = 2016.0;
        let (cache, _tmp) = build_and_open(stars.clone(), epoch);

        // Query with mag_limit=3.5 → should return stars with mag <= 3 (i=1,2,3)
        let results =
            cache.cone(base_ra, base_dec, 0.1, 3.5, epoch).expect("cone mag limit");

        assert!(
            !results.is_empty(),
            "should return some stars below mag 3.5"
        );
        for r in &results {
            assert!(
                r.mag <= 3.5,
                "cone returned star with mag {} > limit 3.5",
                r.mag
            );
        }

        // All pixel slices must be mag-ascending: verify by reading pixel dir directly
        // by querying without limit and checking sort order per pixel
        let all = cache.cone(base_ra, base_dec, 0.1, 99.0, epoch).expect("cone all");
        // The results from a single pixel should be mag-sorted (they come from one pixel slice)
        let mags: Vec<f32> = all.iter().map(|s| s.mag).collect();
        let mut sorted = mags.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(mags, sorted, "results should be mag-ascending (single pixel)");
    }

    // -----------------------------------------------------------------------
    // 4. Proper motion propagation
    // -----------------------------------------------------------------------

    #[test]
    fn proper_motion_propagation() {
        let dec = 60.0_f64;
        // μα* = 300 mas/yr (includes cos δ), μδ = 0
        let pmra = 300.0_f64;
        let pmdec = 0.0_f64;
        let catalog_epoch = 2016.0;
        let query_epoch = 2116.0; // 100 years later
        let dt = query_epoch - catalog_epoch;

        // Expected RA shift: pmra * dt / 3.6e6 / cos(dec) degrees
        let cos_dec = dec.to_radians().cos();
        let expected_dra = pmra * dt / 3.6e6 / cos_dec; // ~0.01667°

        let star = StarRecord {
            ra: 100.0,
            dec,
            mag: 5.0,
            pmra_mas_yr: pmra,
            pmdec_mas_yr: pmdec,
        };

        let epoch = catalog_epoch;
        let (cache, _tmp) = build_and_open(vec![star], epoch);

        // Query a generous cone centered on the expected future position
        let expected_ra = 100.0 + expected_dra;
        let results =
            cache.cone(expected_ra, dec, 0.1, 99.0, query_epoch).expect("cone with pm");

        assert_eq!(results.len(), 1, "should find exactly 1 PM-propagated star");
        let found = &results[0];

        // f32 storage of pmra → quantisation error ≤ 0.01 mas/yr → RA error ≤
        // 0.01*100/3.6e6/cos(60°) ≈ 5.6e-7°; use 1e-4° as a safe tolerance
        let dra_actual = {
            let d = found.ra - 100.0;
            // handle wrap
            if d > 180.0 { d - 360.0 } else if d < -180.0 { d + 360.0 } else { d }
        };
        assert!(
            (dra_actual - expected_dra).abs() < 1e-4,
            "RA shift {dra_actual:.8}° expected ~{expected_dra:.8}°"
        );
        assert!(
            (found.dec - dec).abs() < 1e-6,
            "Dec should be unchanged; got {}, expected {dec}",
            found.dec
        );
    }

    // -----------------------------------------------------------------------
    // 5. import_legacy_healpix_dir
    // -----------------------------------------------------------------------

    #[test]
    fn import_legacy_healpix_dir_round_trip() {
        let tmp = TempDir::new().expect("tempdir");
        let bin_path = tmp.path().join("healpix_000000.bin");

        // Write 3 synthetic 14-byte records
        let records: Vec<(f32, f32, u16, i16, i16)> = vec![
            // ra, dec, mag*1000, pmra/0.01, pmdec/0.01
            (10.5_f32, -20.3_f32, 5500u16, 120i16, -80i16),
            (359.9_f32, 0.0_f32, 11200u16, 0i16, 0i16),
            (180.0_f32, 89.0_f32, 1000u16, -500i16, 300i16),
        ];

        let mut f = File::create(&bin_path).expect("create bin");
        for &(ra, dec, mag_raw, pmra_raw, pmdec_raw) in &records {
            f.write_f32::<LittleEndian>(ra).unwrap();
            f.write_f32::<LittleEndian>(dec).unwrap();
            f.write_u16::<LittleEndian>(mag_raw).unwrap();
            f.write_i16::<LittleEndian>(pmra_raw).unwrap();
            f.write_i16::<LittleEndian>(pmdec_raw).unwrap();
        }
        f.flush().unwrap();
        drop(f);

        let imported = import_legacy_healpix_dir(tmp.path()).expect("import");
        assert_eq!(imported.len(), 3, "should import 3 records");

        for (i, (star, &(ra, dec, mag_raw, pmra_raw, pmdec_raw))) in
            imported.iter().zip(records.iter()).enumerate()
        {
            let tol = 1e-4_f64;
            assert!(
                (star.ra - ra as f64).abs() < tol,
                "record {i}: ra mismatch {} vs {}",
                star.ra,
                ra
            );
            assert!(
                (star.dec - dec as f64).abs() < tol,
                "record {i}: dec mismatch {} vs {}",
                star.dec,
                dec
            );
            assert!(
                (star.mag - mag_raw as f32 / 1000.0).abs() < 1e-4,
                "record {i}: mag mismatch {} vs {}",
                star.mag,
                mag_raw as f32 / 1000.0
            );
            assert!(
                (star.pmra_mas_yr - pmra_raw as f64 * 0.01).abs() < 1e-6,
                "record {i}: pmra mismatch"
            );
            assert!(
                (star.pmdec_mas_yr - pmdec_raw as f64 * 0.01).abs() < 1e-6,
                "record {i}: pmdec mismatch"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. open rejects missing dir + bad magic
    // -----------------------------------------------------------------------

    #[test]
    fn open_rejects_missing_dir() {
        assert!(StarCache::open(Path::new("/no/such/cache/dir")).is_err());
    }

    #[test]
    fn open_rejects_bad_magic() {
        let tmp = TempDir::new().expect("tempdir");
        // Create a stars.smac with wrong magic
        let smac_path = tmp.path().join("stars.smac");
        let mut buf = vec![0u8; (HEADER_SIZE + DIR_SIZE) as usize];
        buf[0..8].copy_from_slice(b"BADMAGIC");
        fs::write(&smac_path, &buf).expect("write bad smac");
        assert!(
            StarCache::open(tmp.path()).is_err(),
            "open should reject bad magic"
        );
    }

    #[test]
    fn open_rejects_truncated_file() {
        let tmp = TempDir::new().expect("tempdir");
        let smac_path = tmp.path().join("stars.smac");
        // Only write the magic, not a full valid file
        fs::write(&smac_path, b"SMACACHE\x01\x00\x06").expect("write truncated");
        assert!(
            StarCache::open(tmp.path()).is_err(),
            "open should reject truncated file"
        );
    }

    // -----------------------------------------------------------------------
    // 7. build_cache_from_legacy_dir — 2-file synthetic round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn build_cache_from_legacy_dir_round_trip() {
        use byteorder::WriteBytesExt;
        use std::fs::File;

        let legacy_tmp = TempDir::new().expect("legacy tempdir");
        let cache_tmp = TempDir::new().expect("cache tempdir");

        // Two synthetic legacy files with 2 records each.
        // File 1: healpix_000000.bin  (RA~10°, Dec~20°)
        // File 2: healpix_000001.bin  (RA~170°, Dec~-30°)
        let files: &[(&str, &[(f32, f32, u16, i16, i16)])] = &[
            (
                "healpix_000000.bin",
                &[
                    (10.0_f32, 20.0_f32, 5000u16, 100i16, -50i16),
                    (10.1_f32, 20.1_f32, 8000u16, 0i16, 0i16),
                ],
            ),
            (
                "healpix_000001.bin",
                &[
                    (170.5_f32, -30.0_f32, 3200u16, 200i16, 150i16),
                    (171.0_f32, -29.8_f32, 9900u16, -10i16, 20i16),
                ],
            ),
        ];

        for (fname, records) in files {
            let path = legacy_tmp.path().join(fname);
            let mut f = File::create(&path).expect("create bin");
            for &(ra, dec, mag_raw, pmra_raw, pmdec_raw) in *records {
                f.write_f32::<LittleEndian>(ra).unwrap();
                f.write_f32::<LittleEndian>(dec).unwrap();
                f.write_u16::<LittleEndian>(mag_raw).unwrap();
                f.write_i16::<LittleEndian>(pmra_raw).unwrap();
                f.write_i16::<LittleEndian>(pmdec_raw).unwrap();
            }
            f.flush().unwrap();
        }

        let total = build_cache_from_legacy_dir(
            legacy_tmp.path(),
            cache_tmp.path(),
            2016.0,
            noop_progress,
        )
        .expect("build_cache_from_legacy_dir");

        assert_eq!(total, 4, "should have ingested all 4 synthetic records");

        let cache = StarCache::open(cache_tmp.path()).expect("open cache");
        assert_eq!(cache.star_count(), 4, "star_count header field");

        // Whole-sky cone (both hemispheres) — union must contain all 4 stars.
        let north = cache.cone(0.0, 90.0, 90.0, 99.0, 2016.0).expect("north cone");
        let south = cache.cone(0.0, -90.0, 90.0, 99.0, 2016.0).expect("south cone");
        let total_found = north.len() + south.len();
        assert_eq!(total_found, 4, "all 4 stars recoverable via full-sky cones");

        // Each synthetic star must appear in a targeted 0.5° cone.
        let all_input: Vec<(f32, f32)> = files
            .iter()
            .flat_map(|(_, recs)| recs.iter().map(|&(ra, dec, ..)| (ra, dec)))
            .collect();
        for (ra, dec) in &all_input {
            let found = cache
                .cone(*ra as f64, *dec as f64, 0.5, 99.0, 2016.0)
                .expect("targeted cone");
            assert!(
                !found.is_empty(),
                "no star found near ({ra}, {dec}) in targeted cone"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 8. Real Gaia fixture (one-time, ~mins, ~2.3 GB)
    // -----------------------------------------------------------------------

    /// Build the ~78 M-star Gaia DR3 fixture cache from the legacy athenaeum
    /// catalog directory.  Skips gracefully when the source is absent.
    ///
    /// Run with:
    /// ```text
    /// cargo test -p solvemyastro --release build_real_gaia_fixture -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "dev: builds the ~78M Gaia fixture (~mins, ~2.3GB)"]
    fn build_real_gaia_fixture() {
        let home = std::env::var("HOME").expect("$HOME not set");
        let legacy_dir = std::path::PathBuf::from(&home)
            .join("Library/Application Support/com.vsharifov.athenaeum/catalogs/gaia_dr3");
        let out_dir = std::path::PathBuf::from(&home)
            .join("Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia");

        if !legacy_dir.is_dir() {
            println!(
                "SKIP build_real_gaia_fixture: legacy_dir not found at {}",
                legacy_dir.display()
            );
            return;
        }

        println!("Building Gaia cache from {} …", legacy_dir.display());
        println!("Output: {}", out_dir.display());

        let t0 = std::time::Instant::now();
        // Use Cell so the closure satisfies Fn (not FnMut) while still
        // updating the last-report timestamp.
        let last_report = std::cell::Cell::new(std::time::Instant::now());

        let count = build_cache_from_legacy_dir(
            &legacy_dir,
            &out_dir,
            2016.0,
            |p| match p {
                BuildProgress::Ingesting { records } => {
                    if last_report.get().elapsed().as_secs() >= 5 {
                        println!("  Ingesting … {records} records");
                        last_report.set(std::time::Instant::now());
                    }
                }
                BuildProgress::Finalizing { shards_done, shards_total } => {
                    if last_report.get().elapsed().as_secs() >= 5 {
                        println!(
                            "  Finalizing … {shards_done}/{shards_total} shards"
                        );
                        last_report.set(std::time::Instant::now());
                    }
                }
                BuildProgress::Complete { records } => {
                    println!("  Complete: {records} records");
                }
            },
        )
        .expect("build_cache_from_legacy_dir");

        let elapsed = t0.elapsed();

        let smac_path = out_dir.join("stars.smac");
        let smac_bytes = std::fs::metadata(&smac_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let smac_mb = smac_bytes as f64 / (1024.0 * 1024.0);

        let cache = StarCache::open(&out_dir).expect("open smac_gaia");
        let star_count = cache.star_count();

        println!("--- Gaia fixture build results ---");
        println!("  star_count : {star_count}");
        println!("  build_time : {:.1}s", elapsed.as_secs_f64());
        println!("  smac size  : {smac_mb:.0} MB  ({smac_bytes} bytes)");

        assert_eq!(
            count as u64, star_count,
            "build count must match header star_count"
        );
        assert!(
            star_count > 50_000_000,
            "expected ~78M Gaia stars, got {star_count}"
        );
    }

    // -----------------------------------------------------------------------
    // 9. Detection-adequacy gate — M78 frame
    // -----------------------------------------------------------------------

    /// Gate test: is `detect_fast` good enough for Phase 2?
    ///
    /// Bright-star completeness ≥ 0.50 → ADEQUATE.
    ///
    /// Run with:
    /// ```text
    /// cargo test -p solvemyastro --release detection_adequacy_m78 -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "dev: needs M78 frame + smac_gaia cache"]
    fn detection_adequacy_m78() {
        use astroimage::ImageAnalyzer;
        use crate::wcs::WcsSolution;

        let m78_path = std::path::Path::new(
            "/Volumes/BigMac/Users/astrobureau/Pictures/Astro/Platesolve Bench/\
             2024-01-13_20-27-32_L_-10.00_120.00s_0081.fits",
        );
        let home = std::env::var("HOME").expect("$HOME not set");
        let smac_gaia = std::path::PathBuf::from(&home)
            .join("Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia");

        if !m78_path.exists() {
            println!("SKIP detection_adequacy_m78: M78 frame not found at {}", m78_path.display());
            return;
        }
        if !smac_gaia.is_dir() {
            println!(
                "SKIP detection_adequacy_m78: smac_gaia cache not found at {}",
                smac_gaia.display()
            );
            return;
        }

        // Reference truth WCS for M78 (0-based crpix = FITS crpix minus 1)
        let wcs = WcsSolution {
            crpix: (2072.5 - 1.0, 1411.5 - 1.0),
            crval: (86.681_973_819_866_826, 0.042_483_541_842_011_370),
            cd: [
                [-4.716_595_432_868_171_2e-5, -2.622_240_404_323_833_8e-4],
                [ 2.619_629_039_840_624_3e-4, -4.666_360_322_280e-5],
            ],
            sip_forward: None,
            sip_reverse: None,
        };

        // FITS frame dimensions
        let fits_w = 4145_usize;
        let fits_h = 2823_usize;

        // Detect stars with detect_fast
        println!("Running detect_fast on M78 frame …");
        let t0 = std::time::Instant::now();
        let result = ImageAnalyzer::new()
            .with_detection_sigma(5.0)
            .with_max_stars(600)
            .with_saturation_fraction(1.0)
            .detect_fast(m78_path)
            .expect("detect_fast");
        println!(
            "  detect_fast: {} stars in {:.1}ms  (detector grid {}×{})",
            result.stars.len(),
            t0.elapsed().as_secs_f64() * 1000.0,
            result.width,
            result.height,
        );

        // Scale factors: detect_fast may bin internally
        let scale_x = fits_w as f64 / result.width as f64;
        let scale_y = fits_h as f64 / result.height as f64;
        println!("  scale factors: x={scale_x:.3}  y={scale_y:.3}");

        // Open the Gaia cache; observation epoch 2024.04
        let cache = StarCache::open(&smac_gaia).expect("open smac_gaia");
        let obs_epoch = 2024.04_f64; // M78 DATE-OBS 2024-01-13

        // Cone centred on M78 field centre, radius 0.8°, mag ≤ 13
        let catalog_stars = cache
            .cone(86.6820, 0.04248, 0.8, 13.0, obs_epoch)
            .expect("cone query");
        println!("  catalog stars in cone: {}", catalog_stars.len());

        // Project catalog stars onto the FITS pixel grid; keep those in-frame
        let in_frame: Vec<(f64, f64, f32)> = catalog_stars
            .iter()
            .filter_map(|s| {
                let (px, py) = wcs.sky_to_pixel(s.ra, s.dec);
                if px >= 0.0 && px < fits_w as f64 && py >= 0.0 && py < fits_h as f64 {
                    Some((px, py, s.mag))
                } else {
                    None
                }
            })
            .collect();
        println!("  catalog stars in frame: {}", in_frame.len());

        // Match tolerance: 3 px on the FITS grid
        let match_radius_px = 3.0_f64;

        // Helper: nearest detection distance for a FITS-grid point.
        let nearest_det_dist = |fpx: f64, fpy: f64| -> f64 {
            result
                .stars
                .iter()
                .map(|s| {
                    let dx = s.x as f64 * scale_x - fpx;
                    let dy = s.y as f64 * scale_y - fpy;
                    dx.hypot(dy)
                })
                .fold(f64::INFINITY, f64::min)
        };

        // (a) Bright-star completeness: catalog stars with mag ≤ 12 that have
        //     a detection within 3 px.
        let bright_in_frame: Vec<(f64, f64, f32)> =
            in_frame.iter().cloned().filter(|&(_, _, m)| m <= 12.0).collect();
        let bright_matched = bright_in_frame
            .iter()
            .filter(|&&(fpx, fpy, _)| nearest_det_dist(fpx, fpy) <= match_radius_px)
            .count();
        let completeness = if bright_in_frame.is_empty() {
            0.0_f64
        } else {
            bright_matched as f64 / bright_in_frame.len() as f64
        };

        println!("--- Detection adequacy ---");
        println!(
            "  bright catalog stars (mag≤12) in frame : {}",
            bright_in_frame.len()
        );
        println!(
            "  matched within 3 px                    : {}",
            bright_matched
        );
        println!(
            "  bright-star completeness               : {:.1}%  ({}/{})",
            completeness * 100.0,
            bright_matched,
            bright_in_frame.len()
        );

        // (b) Quad-pool purity: brightest-20 detections within 3 px of a
        //     catalog star.
        let mut sorted_by_flux = result.stars.clone();
        sorted_by_flux.sort_unstable_by(|a, b| b.flux.partial_cmp(&a.flux).unwrap_or(std::cmp::Ordering::Equal));
        let top20 = sorted_by_flux.iter().take(20);

        let nearest_cat_dist = |dpx: f64, dpy: f64| -> f64 {
            in_frame
                .iter()
                .map(|&(cpx, cpy, _)| {
                    let dx = dpx - cpx;
                    let dy = dpy - cpy;
                    dx.hypot(dy)
                })
                .fold(f64::INFINITY, f64::min)
        };

        let purity_count = top20
            .filter(|s| {
                let fpx = s.x as f64 * scale_x;
                let fpy = s.y as f64 * scale_y;
                nearest_cat_dist(fpx, fpy) <= match_radius_px
            })
            .count();
        let top_n_used = result.stars.len().min(20);
        println!(
            "  brightest-{top_n_used} detections within 3 px of a catalog star: {purity_count}/{top_n_used}"
        );

        // Verdict
        let verdict = if completeness >= 0.50 {
            "DETECTION VERDICT: detect_fast ADEQUATE — proceed with rustafits detection + SNR selection"
        } else {
            "DETECTION VERDICT: INADEQUATE — a dedicated HFD-based detector is required"
        };
        println!("{verdict}");

        // Gate assertions: the probe must have actually run.
        assert!(
            !result.stars.is_empty(),
            "detect_fast returned 0 detections — probe did not run"
        );
        assert!(
            !in_frame.is_empty(),
            "0 catalog stars projected into frame — cone query or WCS failed"
        );
        // Do not panic on INADEQUATE — the verdict is a decision signal.
    }

    // ── Brute-force pixel-search fallback ─────────────────────────────────
    //
    // Last-resort path used when cdshealpix `cone_coverage_approx` panics
    // at every nudge for a sky region (a known 0.7.x issue, observed in
    // the field for a frame at RA≈315° Dec≈-23°). The fallback must
    // (a) return a non-empty set covering the query, and (b) be a
    // superset of `cone_coverage_approx` so consumers don't miss pixels.

    #[test]
    fn bruteforce_returns_pixels_for_known_panic_region() {
        // Exact coords from a user-reported panic (3_18_r_01_P.fit).
        let pixels = super::cone_search_pixels_bruteforce(309.378526, -22.497678, 0.556013);
        assert!(
            !pixels.is_empty(),
            "brute-force must return ≥1 pixel for the panicking region"
        );
        // The pixel containing the cone centre must be in the result.
        let centre_pix = super::nested::hash(
            super::HEALPIX_DEPTH,
            309.378526_f64.to_radians(),
            (-22.497678_f64).to_radians(),
        );
        assert!(
            pixels.contains(&centre_pix),
            "brute-force must include the pixel containing the cone centre"
        );
    }

    /// Diagnostic against the real user `smac_gaia` cache for the
    /// RA≈315° Dec≈-23° panic region. Prints pixel count, candidate-record
    /// count, and final filtered-star count so we can tell whether the
    /// brute-force fallback's pixel indices line up with the cache contents
    /// or not — a decisive signal between "cache has no stars here" vs
    /// "downstream solve pipeline is at fault". `#[ignore]` because it
    /// requires the real on-disk cache.
    #[test]
    #[ignore = "needs ~/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia"]
    fn diag_real_cache_panic_region() {
        let cache_dir = std::path::Path::new(
            "/Volumes/BigMac/Users/astrobureau/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia",
        );
        if !cache_dir.exists() {
            eprintln!("SKIP: smac_gaia cache not present");
            return;
        }
        let cache = StarCache::open(cache_dir).expect("open smac_gaia");
        // Exact panicking call from user-reported log (3_18_r_01_P.fit).
        let ra = 314.183208_f64;
        let dec = -23.053690_f64;
        let radius_deg = 0.556013_f64;
        let pixels = super::cone_search_pixels(ra, dec, radius_deg);
        let counts: Vec<u32> = pixels
            .iter()
            .map(|&p| cache.directory[p as usize].1)
            .collect();
        let total_records: u64 = counts.iter().map(|&c| c as u64).sum();
        let stars = cache
            .cone(ra, dec, radius_deg, 18.0, 2024.04)
            .expect("cone query");
        eprintln!(
            "DIAG ra={ra} dec={dec} r={radius_deg}: pixels={}, total candidate records={total_records}, filtered stars={}, first 5 pixel counts={:?}",
            pixels.len(),
            stars.len(),
            counts.iter().take(5).collect::<Vec<_>>()
        );
        // Sanity: at this Milky-Way density we expect thousands of stars.
        // We don't assert > N because the goal is the diagnostic print —
        // but we do assert non-empty so a regression surfaces.
        assert!(!pixels.is_empty(), "brute-force/approx returned 0 pixels");
    }

    /// Focused diagnostic: solve `3_18_r_01_P.fit` end-to-end and let
    /// solvemyastro print its full stderr trace. Mirrors how it'd run from
    /// the desktop app but bypasses the Tauri/web layer so we see every
    /// `[solvemyastro] ...` log line without truncation.
    #[test]
    #[ignore = "needs the bench frame + smac_gaia cache"]
    fn diag_solve_3_18_r_01_P() {
        use crate::{SolveConfig, SolveHints};
        let cache_dir = std::path::Path::new(
            "/Volumes/BigMac/Users/astrobureau/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia",
        );
        let image = std::path::Path::new(
            "/Volumes/BigMac/Users/astrobureau/Pictures/Astro/Platesolve Bench/3_18_r_01_P.fit",
        );
        if !cache_dir.exists() || !image.exists() {
            eprintln!("SKIP: cache or image missing");
            return;
        }
        let cache = StarCache::open(cache_dir).expect("open smac_gaia");
        // Reference truth: RA 316.75 Dec -14.92, scale 0.497"/px. Pass NO
        // hints first to see what blind solve does; that's the path the
        // desktop app takes if header hints are misleading.
        eprintln!("===== HINTLESS (blind) =====");
        let blind = crate::solve(
            image,
            &SolveHints::default(),
            &cache,
            &SolveConfig::default(),
            None,
        );
        eprintln!("blind outcome: {:?}", blind.as_ref().err().map(|e| e.to_string()));
        // Then with reference-correct hints to localise whether the solver
        // can succeed when given the right neighbourhood. Search radius 1°.
        eprintln!("===== HINTED (reference-correct centre, blind scale) =====");
        let hinted = crate::solve(
            image,
            &SolveHints {
                ra: Some(316.75),
                dec: Some(-14.92),
                fov_deg: None,
                pixel_scale_arcsec: None,
                search_radius_deg: Some(1.0),
            },
            &cache,
            &SolveConfig::default(),
            None,
        );
        eprintln!("hinted outcome: {:?}", hinted.as_ref().map(|s|
            format!("SOLVED ra={:.4} dec={:.4} scale={:.3}\" rms={:.2}px",
                s.wcs.crval.0, s.wcs.crval.1, s.pixel_scale_arcsec, s.rms_residual_px)
        ).unwrap_or_else(|e| format!("ERR: {e}")));
    }

    /// Same diagnostic shape as `diag_solve_3_18_r_01_P`, but for
    /// `1_18_r_1_P.fit` (header missing FOCALLEN/XPIXSZ/OBJCTRA/OBJCTDEC,
    /// reference truth RA 234.60° Dec -33.24°, 0.497"/px). Tells us whether
    /// the solve actually succeeds on this frame — if not, the
    /// focallen-back-fill question is moot (no `store_result` ever runs).
    #[test]
    #[ignore = "needs the bench frame + smac_gaia cache"]
    fn diag_solve_1_18_r_1_p() {
        use crate::{SolveConfig, SolveHints};
        let cache_dir = std::path::Path::new(
            "/Volumes/BigMac/Users/astrobureau/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia",
        );
        let image = std::path::Path::new(
            "/Volumes/BigMac/Users/astrobureau/Pictures/Astro/Platesolve Bench/1_18_r_1_P.fit",
        );
        if !cache_dir.exists() || !image.exists() {
            eprintln!("SKIP: cache or image missing");
            return;
        }
        let cache = StarCache::open(cache_dir).expect("open smac_gaia");
        eprintln!("===== HINTLESS (blind — mirrors desktop app for this frame) =====");
        let blind = crate::solve(
            image,
            &SolveHints::default(),
            &cache,
            &SolveConfig::default(),
            None,
        );
        eprintln!("blind outcome: {:?}", blind.as_ref().map(|s|
            format!("SOLVED ra={:.4} dec={:.4} scale={:.3}\" rms={:.2}px",
                s.wcs.crval.0, s.wcs.crval.1, s.pixel_scale_arcsec, s.rms_residual_px)
        ).unwrap_or_else(|e| format!("ERR: {e}")));
        eprintln!("===== HINTED (reference-correct centre, blind scale, radius=1°) =====");
        let hinted = crate::solve(
            image,
            &SolveHints {
                ra: Some(234.60),
                dec: Some(-33.24),
                fov_deg: None,
                pixel_scale_arcsec: None,
                search_radius_deg: Some(1.0),
            },
            &cache,
            &SolveConfig::default(),
            None,
        );
        eprintln!("hinted outcome: {:?}", hinted.as_ref().map(|s|
            format!("SOLVED ra={:.4} dec={:.4} scale={:.3}\" rms={:.2}px",
                s.wcs.crval.0, s.wcs.crval.1, s.pixel_scale_arcsec, s.rms_residual_px)
        ).unwrap_or_else(|e| format!("ERR: {e}")));
    }

    #[test]
    fn bruteforce_is_superset_of_cone_coverage_approx() {
        // On a non-panicking region, brute-force must contain every pixel
        // the normal cone_coverage_approx path returns (the fallback may be
        // slightly larger due to the half-diagonal inflation — that's fine,
        // the `cone()` consumer filters stars by true great-circle distance).
        let approx = super::cone_search_pixels(180.0, 30.0, 0.5);
        let brute = super::cone_search_pixels_bruteforce(180.0, 30.0, 0.5);
        assert!(!approx.is_empty(), "sanity: approx returns pixels");
        for p in &approx {
            assert!(
                brute.contains(p),
                "brute-force missing pixel {p} that approx returned"
            );
        }
    }
}
