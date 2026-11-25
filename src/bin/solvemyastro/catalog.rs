//! The catalog-building commands: Gaia files to bins, tiers, index packs and
//! the publish tree.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{ArgGroup, Args};

use solvemyastro::bins::RECORD_SIZE;
use solvemyastro::cache::{build_cache_from_legacy_dir, BuildProgress, StarRecord};
use solvemyastro::gaia::{self, IngestOptions, IngestProgress};
use solvemyastro::index::{self, BuildParams, Preset};
use solvemyastro::layout::STARS_FILE;
use solvemyastro::StarCache;

/// Julian year of Gaia DR3 positions.
const GAIA_DR3_EPOCH: f64 = 2016.0;

#[derive(Args)]
pub struct IngestArgs {
    /// Folder of Gaia DR3 `GaiaSource_*.csv.gz` files (and, if present, `_MD5SUM.txt`).
    #[arg(long, value_name = "DIR")]
    gaia: PathBuf,
    /// Folder the bins are written to; it must not hold bins already.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// A star at or past this G magnitude is dropped.
    #[arg(long, default_value_t = gaia::DEFAULT_MAG_LIMIT)]
    mag_limit: f32,
    /// Files read in parallel.
    #[arg(long, default_value_t = gaia::DEFAULT_JOBS)]
    jobs: usize,
    /// Check every file against `_MD5SUM.txt` before reading any. A missing list, or a file it does
    /// not name, fails the run.
    #[arg(long)]
    verify: bool,
}

pub fn ingest(a: IngestArgs) -> Result<()> {
    let opts = IngestOptions { mag_limit: a.mag_limit, jobs: a.jobs, verify_md5: a.verify };
    let summary = gaia::ingest(&a.gaia, &a.out, &opts, &|p| match p {
        IngestProgress::Started { total_files } => tracing::info!(total_files, "ingest started"),
        IngestProgress::Verified { name } => tracing::info!(file = %name, "checksum matches"),
        IngestProgress::FileDone { name, completed, total_files, stars_kept } => {
            tracing::info!(file = %name, completed, total_files, stars_kept, "file read")
        }
        IngestProgress::Finalizing => tracing::info!("writing bins"),
        IngestProgress::Complete { total_stars } => tracing::info!(total_stars, "bins written"),
    })?;
    match summary.coverage.complete_cells() {
        Some(cells) => tracing::info!(
            complete_cells = cells.len(),
            partial_cells = summary.coverage.partial_cells().map_or(0, |c| c.len()),
            "coverage by file range"
        ),
        None => tracing::warn!(
            files_without_range = summary.coverage.unknown.len(),
            "some files carry no range in their name, so which cells are whole is unknown"
        ),
    }
    println!("ingest: {} files, {} stars -> {}", summary.files, summary.stars, a.out.display());
    Ok(())
}

#[derive(Args)]
pub struct BuildTiersArgs {
    /// Folder of bins (`ingest`'s output).
    #[arg(long, value_name = "DIR")]
    bins: PathBuf,
    /// Folder the `tier_<density>/stars.smac` caches are written under.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Julian year of the catalog positions.
    #[arg(long, default_value_t = GAIA_DR3_EPOCH)]
    epoch: f64,
}

pub fn build_tiers(a: BuildTiersArgs) -> Result<()> {
    let started = Instant::now();
    let tiers = solvemyastro::tiers::build_tiers(&a.bins, &a.out, a.epoch)?;
    for (density, stars) in &tiers {
        println!("tier_{density}: {stars} stars");
    }
    println!("build-tiers: {} tiers -> {} ({:.1} s)", tiers.len(), a.out.display(), started.elapsed().as_secs_f64());
    Ok(())
}

#[derive(Args)]
#[command(group(ArgGroup::new("source").required(true).args(["bins", "from"])))]
pub struct BuildCacheArgs {
    /// Folder of bins to turn into one cache.
    #[arg(long, value_name = "DIR")]
    bins: Option<PathBuf>,
    /// The same, under its older name (a folder of `healpix_NNNNNN.bin` files).
    #[arg(long, value_name = "DIR")]
    from: Option<PathBuf>,
    /// Folder the `stars.smac` is written to.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Julian year of the catalog positions.
    #[arg(long, default_value_t = GAIA_DR3_EPOCH)]
    epoch: f64,
}

/// Every `healpix_*.bin` in `dir` can be statted, is a regular file and holds a
/// whole number of records. `build_cache_from_legacy_dir` reads each bin with
/// the error turned into an empty list, so a torn or unreadable bin must be
/// refused here, by name, before it is called. A read that fails after this
/// check has passed is not covered.
fn check_bins(dir: &Path) -> Result<()> {
    let entries = std::fs::read_dir(dir).with_context(|| format!("read bins folder {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("read bins folder {}", dir.display()))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !(name.starts_with("healpix_") && name.ends_with(".bin")) {
            continue;
        }
        let path = entry.path();
        let meta = std::fs::metadata(&path).with_context(|| format!("stat {}", path.display()))?;
        if !meta.is_file() {
            bail!("{} is not a file", path.display());
        }
        if meta.len() % RECORD_SIZE as u64 != 0 {
            bail!("{} is {} bytes, not a whole number of {RECORD_SIZE}-byte records", path.display(), meta.len());
        }
    }
    Ok(())
}

pub fn build_cache(a: BuildCacheArgs) -> Result<()> {
    let from = a.bins.or(a.from).context("--bins <dir> or --from <dir> is required")?;
    let (out, epoch) = (a.out, a.epoch);
    check_bins(&from).inspect_err(|e| tracing::error!(from = %from.display(), error = %format!("{e:#}"), "bins refused"))?;
    std::fs::create_dir_all(&out).with_context(|| format!("create --out {}", out.display()))?;
    tracing::info!(from = %from.display(), out = %out.display(), epoch, "build-cache");
    let total = build_cache_from_legacy_dir(&from, &out, epoch, |p| match p {
        BuildProgress::Ingesting { records } => tracing::info!(records, "ingesting"),
        BuildProgress::Finalizing { shards_done, shards_total } => {
            tracing::info!(shards_done, shards_total, "finalizing shards")
        }
        BuildProgress::Complete { records } => tracing::info!(records, "done"),
    })?;
    println!("build-cache: wrote {total} stars to {}", out.join(STARS_FILE).display());
    Ok(())
}

#[derive(Args)]
pub struct BuildBrightCacheArgs {
    /// The deep cache folder to derive from.
    #[arg(long, value_name = "DIR")]
    from: PathBuf,
    /// Folder the bright cache is written to.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Stars brighter than this G magnitude are all kept.
    #[arg(long, default_value_t = 14.0)]
    bright_floor: f64,
    /// Stars wanted per level-6 cell.
    #[arg(long, default_value_t = 100)]
    target_density: usize,
    /// hybrid: the floor, topped up to the density; floor-only; cap-only.
    #[arg(long, default_value = "hybrid", value_parser = ["hybrid", "floor-only", "cap-only"])]
    mode: String,
}

pub fn build_bright_cache(a: BuildBrightCacheArgs) -> Result<()> {
    // Derive a smaller "bright" star cache from an existing deep cache. Hybrid
    // algorithm: per HEALPix-6 pixel, include all stars with mag < bright_floor
    // (floor), then if the floor count is below target_density, top up with the
    // next-brightest stars (continuing past the floor) until reaching the
    // target or exhausting the pixel.
    let (from, out) = (a.from, a.out);
    let bright_floor = a.bright_floor as f32;
    let (target_density, mode) = (a.target_density, a.mode);
    std::fs::create_dir_all(&out).with_context(|| format!("create --out {}", out.display()))?;
    let src = StarCache::open(&from)?;
    let src_epoch = src.catalog_epoch();
    let src_count = src.star_count();
    tracing::info!(
        from = %from.display(),
        out = %out.display(),
        source_epoch = src_epoch,
        source_stars = src_count,
        mode = %mode,
        bright_floor,
        target_density,
        "build-bright-cache"
    );

    // Stream per-pixel selections into the existing build_cache.
    // Buffer per pixel is small (< 1000 stars typically); pre-sort
    // happens on disk (records are mag-sorted ascending per pixel).
    const N_PIXELS: u64 = 49_152; // depth=6
    let bright_floor_v = bright_floor;
    let target_density_v = target_density;
    let mode_v = mode.clone();
    // The first pixel that cannot be read stops the selection (later pixels
    // give nothing) and fails the build below, naming it.
    let first_error: RefCell<Option<anyhow::Error>> = RefCell::new(None);
    let first_error_ref = &first_error;
    let from_ref = &from;
    let pixel_iter = (0..N_PIXELS).flat_map(move |px| {
        let records = if first_error_ref.borrow().is_some() {
            Vec::new()
        } else {
            match src.iter_pixel_records(px) {
                Ok(r) => r,
                Err(e) => {
                    let e = e.context(format!("read pixel {px} of the cache in {}", from_ref.display()));
                    *first_error_ref.borrow_mut() = Some(e);
                    Vec::new()
                }
            }
        };
        let selected: Vec<StarRecord> = match mode_v.as_str() {
            "floor-only" => records.into_iter().take_while(|r| r.mag < bright_floor_v).collect(),
            "cap-only" => records.into_iter().take(target_density_v).collect(),
            _ /* hybrid */ => {
                let floor: Vec<StarRecord> =
                    records.iter().take_while(|r| r.mag < bright_floor_v).copied().collect();
                if floor.len() >= target_density_v {
                    floor
                } else {
                    let mut s = floor;
                    let need = target_density_v - s.len();
                    s.extend(records.iter().skip(s.len()).take(need).copied());
                    s
                }
            }
        };
        selected.into_iter()
    });

    let built = solvemyastro::cache::build_cache(pixel_iter, &out, src_epoch, |p| match p {
        BuildProgress::Ingesting { records } => {
            if records % 500_000 == 0 {
                tracing::info!(records, "ingesting");
            }
        }
        BuildProgress::Finalizing { shards_done, shards_total } => {
            if shards_done % 5000 == 0 || shards_done == shards_total {
                tracing::info!(shards_done, shards_total, "finalizing pixels");
            }
        }
        BuildProgress::Complete { records } => tracing::info!(records, "done"),
    });
    if let Some(e) = first_error.take() {
        tracing::warn!(error = %format!("{e:#}"), out = %out.display(), "a source pixel was unreadable; the bright cache is discarded");
        let smac = out.join(STARS_FILE);
        match std::fs::remove_file(&smac) {
            Ok(()) => {}
            Err(re) if re.kind() == std::io::ErrorKind::NotFound => {}
            Err(re) => tracing::error!(path = %smac.display(), error = %re, "could not remove the half-written cache"),
        }
        return Err(e);
    }
    let total = built?;

    // Sidecar metadata for debugging + future stale-cache detection.
    let meta = format!(
        "{{\n  \"source_smac_size\": {},\n  \"source_epoch\": {},\n  \
         \"bright_floor\": {},\n  \"target_density\": {},\n  \
         \"mode\": \"{}\",\n  \"output_star_count\": {}\n}}\n",
        src_count, src_epoch, bright_floor, target_density, mode, total,
    );
    std::fs::write(out.join("bright.meta.json"), &meta).with_context(|| format!("write bright.meta.json in {}", out.display()))?;
    println!("build-bright-cache: wrote {total} stars to {}", out.join(STARS_FILE).display());
    Ok(())
}

#[derive(Args)]
pub struct BuildIndexArgs {
    /// Folder holding the `tier_*` caches.
    #[arg(long, value_name = "DIR")]
    tiers: PathBuf,
    /// Folder the pack's `quads.smaq` is written to.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Build a published pack: `base` (tier_500, depth 2-7) or `narrow`
    /// (tier_500 + tier_2000, depth 8).
    #[arg(
        long,
        value_parser = ["base", "narrow"],
        conflicts_with_all = ["min_depth", "max_depth", "stars_per_cell", "quads_per_cell"]
    )]
    preset: Option<String>,
    /// Coarsest band depth (largest quads). [default: 2]
    #[arg(long)]
    min_depth: Option<u8>,
    /// Finest band depth (smallest quads). [default: 7]
    #[arg(long)]
    max_depth: Option<u8>,
    /// Brightest stars kept per cell. [default: 10]
    #[arg(long)]
    stars_per_cell: Option<usize>,
    /// Quads kept per cell. [default: 8]
    #[arg(long)]
    quads_per_cell: Option<usize>,
}

pub fn build_index(a: BuildIndexArgs) -> Result<()> {
    let (layers, params) = match a.preset.as_deref().map(|n| Preset::from_name(n).context("unknown preset")) {
        Some(preset) => {
            let preset = preset?;
            (index::open_preset_layers(&a.tiers, preset)?, preset.params())
        }
        None => {
            let d = BuildParams::default();
            let params = BuildParams {
                min_depth: a.min_depth.unwrap_or(d.min_depth),
                max_depth: a.max_depth.unwrap_or(d.max_depth),
                stars_per_cell: a.stars_per_cell.unwrap_or(d.stars_per_cell),
                quads_per_cell: a.quads_per_cell.unwrap_or(d.quads_per_cell),
            };
            // Tiers bright -> faint; the index only needs the bright ones.
            let mut layers = Vec::new();
            for t in ["tier_500", "tier_2000"] {
                let dir = a.tiers.join(t);
                if dir.is_dir() {
                    layers.push(StarCache::open(&dir)?);
                }
            }
            if layers.is_empty() {
                bail!("no tier_500 / tier_2000 under {}", a.tiers.display());
            }
            (layers, params)
        }
    };
    let refs: Vec<&StarCache> = layers.iter().collect();
    let started = Instant::now();
    let (n_stars, n_quads) = index::build_index(&refs, &a.out, &params, |m| tracing::info!(step = %m, "build-index"))?;
    println!(
        "build-index: {n_stars} stars, {n_quads} quads -> {} ({:.1} s)",
        a.out.join(index::INDEX_FILE).display(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// `<name>=<dir>`: a pack's name in the manifest and the folder that holds it.
fn named_pack(s: &str) -> Result<(String, PathBuf), String> {
    match s.split_once('=') {
        Some((name, dir)) if !name.is_empty() && !dir.is_empty() => Ok((name.to_string(), PathBuf::from(dir))),
        _ => Err(format!("expected <name>=<dir>, got {s:?}")),
    }
}

#[derive(Args)]
pub struct PackageArgs {
    /// Folder holding the `tier_*` caches to package.
    #[arg(long, value_name = "DIR")]
    tiers: Option<PathBuf>,
    /// An index pack to package, as `<name>=<dir>`; repeat for several.
    #[arg(long, value_name = "NAME=DIR", value_parser = named_pack)]
    index: Vec<(String, PathBuf)>,
    /// The publish folder.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
}

pub fn package(a: PackageArgs) -> Result<()> {
    if a.tiers.is_none() && a.index.is_empty() {
        bail!("nothing to package: give --tiers <dir> and/or --index <name>=<dir>");
    }
    let packs: Vec<(&str, &Path)> = a.index.iter().map(|(n, d)| (n.as_str(), d.as_path())).collect();
    // The index packs are checked before the tiers are written, so a refused
    // pack name leaves no half-written publish tree.
    if !packs.is_empty() {
        solvemyastro::package::check_index_packs(&packs)?;
    }
    if let Some(tiers) = &a.tiers {
        let m = solvemyastro::package::package_tiers(tiers, &a.out)?;
        println!("package: {} tiers (catalog epoch J{}) -> {}", m.tiers.len(), m.catalog_epoch, a.out.display());
    }
    if !packs.is_empty() {
        let m = solvemyastro::package::package_index(&packs, &a.out)?;
        println!("package: {} index packs (catalog epoch J{}) -> {}", m.packs.len(), m.catalog_epoch, a.out.display());
    }
    Ok(())
}
