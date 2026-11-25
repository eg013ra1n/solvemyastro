//! `info`: what a folder holds.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Args;

use solvemyastro::bins::RECORD_SIZE;
use solvemyastro::gaia::Coverage;
use solvemyastro::index::{QuadIndex, FORMAT_VERSION, INDEX_FILE};
use solvemyastro::layout::{discover_index_packs, discover_layers, STARS_FILE};
use solvemyastro::manifest::{IndexManifest, Manifest, INDEX_MANIFEST_FILE, MANIFEST_FILE};
use solvemyastro::StarCache;

#[derive(Args)]
pub struct InfoArgs {
    /// A star cache, a folder of tiers, an index pack or a folder of them, a folder of bins or a publish folder.
    dir: PathBuf,
}

pub fn run(a: InfoArgs) -> Result<()> {
    let dir = a.dir.as_path();
    if !dir.is_dir() {
        bail!("{} is not a folder", dir.display());
    }
    if dir.join(STARS_FILE).is_file() {
        return cache(dir);
    }
    if dir.join(INDEX_FILE).is_file() {
        return index_pack(dir);
    }
    if has_bins(dir)? {
        return bins(dir);
    }
    if !tier_dirs(dir).is_empty() {
        return tiers(dir);
    }
    let packs = discover_index_packs(dir);
    if !packs.is_empty() {
        return index_packs(dir, &packs);
    }
    if dir.join(MANIFEST_FILE).is_file() || dir.join(INDEX_MANIFEST_FILE).is_file() {
        return publish(dir);
    }
    bail!(
        "{} holds no {STARS_FILE}, {INDEX_FILE}, tier_* folder, index pack folder, healpix_*.bin file or manifest",
        dir.display()
    )
}

fn cache(dir: &Path) -> Result<()> {
    let cache = StarCache::open(dir)?;
    println!("cache-info: {}", dir.display());
    println!("  stars:         {}", cache.star_count());
    println!("  catalog epoch: J{}", cache.catalog_epoch());
    Ok(())
}

/// The `tier_<density>` folders that hold a star cache.
fn tier_dirs(dir: &Path) -> Vec<PathBuf> {
    // `discover_layers` also answers with the folder itself when it holds a
    // legacy cache, which `run` has already dealt with.
    discover_layers(dir).into_iter().filter(|p| p != dir).collect()
}

fn tiers(dir: &Path) -> Result<()> {
    println!("tiers: {}", dir.display());
    for tier in tier_dirs(dir) {
        let cache = StarCache::open(&tier)?;
        let name = tier.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        println!("  {name}: {} stars, catalog epoch J{}", cache.star_count(), cache.catalog_epoch());
    }
    Ok(())
}

fn index_pack(dir: &Path) -> Result<()> {
    println!("index pack: {}", dir.display());
    pack_lines(dir, "  ")
}

/// A folder of index packs, one subfolder each (the app's `smac_index`; its
/// cached `manifest.json` is an index manifest, which `info` does not need).
fn index_packs(dir: &Path, packs: &[PathBuf]) -> Result<()> {
    println!("index packs: {}", dir.display());
    for pack in packs {
        let name = pack.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        println!("  {name}:");
        pack_lines(pack, "    ")?;
    }
    Ok(())
}

fn pack_lines(dir: &Path, indent: &str) -> Result<()> {
    let ix = QuadIndex::open(dir)?;
    let (mut quads, mut stars) = (0usize, 0usize);
    for b in ix.bands() {
        quads += b.n_quads();
        stars += b.n_stars();
    }
    let (first, last) = (ix.bands().first().map(|b| b.depth), ix.bands().last().map(|b| b.depth));
    println!("{indent}format:        {FORMAT_VERSION}");
    println!("{indent}catalog epoch: J{}", ix.catalog_epoch());
    println!("{indent}min fov:       {:.2} deg", ix.min_fov_deg());
    println!("{indent}bands:         {} (depth {}..{})", ix.bands().len(), first.unwrap_or(0), last.unwrap_or(0));
    println!("{indent}quads:         {quads}");
    println!("{indent}stars:         {stars}");
    Ok(())
}

/// Whether the folder holds a `healpix_NNNNNN.bin` file.
fn has_bins(dir: &Path) -> Result<bool> {
    for e in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        if is_bin(&e.with_context(|| format!("read {}", dir.display()))?.file_name().to_string_lossy()) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn is_bin(name: &str) -> bool {
    name.starts_with("healpix_") && name.ends_with(".bin")
}

fn bins(dir: &Path) -> Result<()> {
    let (mut files, mut bytes, mut ragged) = (0usize, 0u64, 0usize);
    for e in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let e = e.with_context(|| format!("read {}", dir.display()))?;
        if !is_bin(&e.file_name().to_string_lossy()) {
            continue;
        }
        let len = e.metadata().with_context(|| format!("stat {}", e.path().display()))?.len();
        files += 1;
        bytes += len;
        if len % RECORD_SIZE as u64 != 0 {
            ragged += 1;
        }
    }
    println!("bins: {}", dir.display());
    println!("  files:   {files}");
    println!("  records: {}", bytes / RECORD_SIZE as u64);
    if ragged > 0 {
        println!("  {ragged} files are not a whole number of {RECORD_SIZE}-byte records");
    }
    match Coverage::read(dir)? {
        None => println!("  no coverage record: which cells are whole is unknown"),
        Some(cov) => {
            println!("  coverage: {} source files at level {}", cov.files.len(), cov.level);
            match (cov.complete_cells(), cov.partial_cells()) {
                (Some(complete), Some(partial)) => {
                    // Gaia splits its files by the HEALPix index in `source_id`
                    // and the bins use the DR3 ra/dec, so a star near a cell
                    // edge can sit in a neighbouring file.
                    println!(
                        "complete cells: {} (complete by file range: every level-8 pixel of the cell lies in the \
                         files' names; a star near a cell edge can sit in a neighbouring file)",
                        complete.len()
                    );
                    println!("partial cells: {} (reached by part of their files only)", partial.len());
                }
                _ => println!(
                    "complete cells: unknown ({} files carry no range in their name)",
                    cov.unknown.len()
                ),
            }
        }
    }
    Ok(())
}

fn publish(dir: &Path) -> Result<()> {
    println!("publish: {}", dir.display());
    let path = dir.join(MANIFEST_FILE);
    if path.is_file() {
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let m = Manifest::from_json_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        println!("  tiers (catalog epoch J{}):", m.catalog_epoch);
        for t in &m.tiers {
            println!("    {}: {} bytes, min fov {:.2} deg", t.zip, t.size_bytes, t.min_fov_deg);
        }
    }
    let path = dir.join(INDEX_MANIFEST_FILE);
    if path.is_file() {
        let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
        let m = IndexManifest::from_json_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        println!("  index packs (format {}, catalog epoch J{}):", m.format, m.catalog_epoch);
        for p in &m.packs {
            println!("    {}: {} bytes, min fov {:.2} deg", p.name, p.size_bytes, p.min_fov_deg);
        }
    }
    Ok(())
}
