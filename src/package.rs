//! The publish tree: per-tier and per-pack zips with their SHA-256 sidecars
//! and the two manifests ([`crate::manifest`]) a download path reads.
//!
//! ```text
//! <publish>/manifest.json            tier_<d>.zip   tier_<d>.zip.sha256
//! <publish>/index_manifest.json      index_<name>.zip   index_<name>.zip.sha256
//! ```
//!
//! A tier zip holds `tier_<d>/stars.smac`, a pack zip `<name>/quads.smaq`. The
//! manifests are written with `serde_json::to_vec_pretty` in the field order
//! of their structs, so the same inputs give the same bytes.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read};
use std::path::Path;

use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use zip::{write::SimpleFileOptions, CompressionMethod, ZipWriter};

use crate::cache::StarCache;
use crate::index::{QuadIndex, FORMAT_VERSION, INDEX_FILE};
use crate::layout::{tier_dir_name, STARS_FILE};
use crate::manifest::{
    IndexManifest, IndexManifestPack, Manifest, ManifestTier, INDEX_MANIFEST_FILE, MANIFEST_FILE,
};
use crate::tiers::TIER_DENSITIES;

fn min_fov_for(density: u32) -> f64 {
    match density {
        d if d <= 500 => 0.6,
        d if d <= 2000 => 0.3,
        d if d <= 5000 => 0.2,
        _ => 0.15,
    }
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    // digest 0.11 dropped `LowerHex` on `Output` (generic-array ->
    // hybrid-array); hex the bytes by hand — same lowercase digest string.
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Zip `src` as the single entry `entry_name` into `zip_path` (Stored —
/// dense binary; `large_file` for >4 GB files).
pub(crate) fn zip_file(src: &Path, entry_name: &str, zip_path: &Path) -> Result<()> {
    let zf = BufWriter::new(File::create(zip_path).with_context(|| format!("create {}", zip_path.display()))?);
    let mut zw = ZipWriter::new(zf);
    let opts = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        .large_file(true);
    zw.start_file(entry_name, opts).with_context(|| format!("start {entry_name} in {}", zip_path.display()))?;
    let mut f = BufReader::new(File::open(src).with_context(|| format!("open {}", src.display()))?);
    io::copy(&mut f, &mut zw).with_context(|| format!("zip {}", src.display()))?;
    zw.finish().with_context(|| format!("finish {}", zip_path.display()))?;
    Ok(())
}

/// Write the `"<hex>  <zip name>\n"` SHA-256 sidecar for `zip_path` next to it
/// (`<zip name>.sha256`); returns the sidecar's file name.
pub(crate) fn write_sha256_sidecar(pub_dir: &Path, zip_name: &str, zip_path: &Path) -> Result<String> {
    let sha_name = format!("{zip_name}.sha256");
    let digest = sha256_file(zip_path)?;
    let path = pub_dir.join(&sha_name);
    fs::write(&path, format!("{digest}  {zip_name}\n")).with_context(|| format!("write {}", path.display()))?;
    Ok(sha_name)
}

/// Package every `tier_<d>/stars.smac` under `tiers_dir` (the densities of
/// [`TIER_DENSITIES`] that are there) into `publish_dir` as `tier_<d>.zip`,
/// its sidecar and `manifest.json`. The manifest's catalog epoch is the first
/// tier's. No tier at all is an error.
pub fn package_tiers(tiers_dir: &Path, publish_dir: &Path) -> Result<Manifest> {
    fs::create_dir_all(publish_dir).with_context(|| format!("create {}", publish_dir.display()))?;

    let mut epoch = None;
    let mut tiers = Vec::new();
    for density in TIER_DENSITIES {
        let dir_name = tier_dir_name(density);
        let tier_dir = tiers_dir.join(&dir_name);
        let smac = tier_dir.join(STARS_FILE);
        if !smac.is_file() {
            tracing::info!(density, path = %smac.display(), "tier not present, not packaged");
            continue;
        }
        if epoch.is_none() {
            let cache = StarCache::open(&tier_dir).with_context(|| format!("open {}", tier_dir.display()))?;
            epoch = Some(cache.catalog_epoch());
        }
        let zip_name = format!("{dir_name}.zip");
        let zip_path = publish_dir.join(&zip_name);
        zip_file(&smac, &format!("{dir_name}/{STARS_FILE}"), &zip_path)?;
        let sha256 = write_sha256_sidecar(publish_dir, &zip_name, &zip_path)?;
        tiers.push(ManifestTier {
            density,
            zip: zip_name,
            sha256,
            dir: dir_name,
            size_bytes: fs::metadata(&zip_path).with_context(|| format!("stat {}", zip_path.display()))?.len(),
            min_fov_deg: min_fov_for(density),
        });
        tracing::info!(density, "tier packaged");
    }
    let Some(catalog_epoch) = epoch else {
        bail!("no tier found under {} (expected {}/{STARS_FILE})", tiers_dir.display(), tier_dir_name(TIER_DENSITIES[0]));
    };

    let manifest = Manifest { version: 1, catalog_epoch, tiers };
    write_manifest(&publish_dir.join(MANIFEST_FILE), &manifest)?;
    Ok(manifest)
}

/// Package the quad-index packs `(name, dir)` — `dir` holds `quads.smaq` —
/// into `publish_dir` as `index_<name>.zip` (entry `<name>/quads.smaq`), its
/// sidecar and `index_manifest.json`, in the order given. Every pack must open
/// and all must be built from the same catalog epoch, which the manifest
/// records; no pack at all is an error.
pub fn package_index(packs: &[(&str, &Path)], publish_dir: &Path) -> Result<IndexManifest> {
    if packs.is_empty() {
        bail!("no index pack to package");
    }
    fs::create_dir_all(publish_dir).with_context(|| format!("create {}", publish_dir.display()))?;

    let mut epoch: Option<(f64, &str)> = None;
    let mut entries = Vec::new();
    for &(name, dir) in packs {
        let file = dir.join(INDEX_FILE);
        let ix = QuadIndex::open(dir)
            .with_context(|| format!("the {name} pack at {} does not open", dir.display()))?;
        let (min_fov_deg, pack_epoch) = (ix.min_fov_deg(), ix.catalog_epoch());
        drop(ix);
        match epoch {
            None => epoch = Some((pack_epoch, name)),
            Some((first, first_name)) if first != pack_epoch => {
                tracing::warn!(pack = name, first_pack = first_name, pack_epoch, first_epoch = first, "index packs differ in catalog epoch");
                bail!(
                    "the {name} pack is built from catalog epoch {pack_epoch} but the {first_name} pack from {first}: \
                     one manifest cannot name both"
                );
            }
            Some(_) => {}
        }

        let zip_name = format!("index_{name}.zip");
        let zip_path = publish_dir.join(&zip_name);
        zip_file(&file, &format!("{name}/{INDEX_FILE}"), &zip_path)?;
        let sha256 = write_sha256_sidecar(publish_dir, &zip_name, &zip_path)?;
        entries.push(IndexManifestPack {
            name: name.to_string(),
            zip: zip_name,
            sha256,
            dir: name.to_string(),
            size_bytes: fs::metadata(&zip_path).with_context(|| format!("stat {}", zip_path.display()))?.len(),
            min_fov_deg,
        });
        tracing::info!(pack = name, "index pack packaged");
    }
    let (catalog_epoch, _) = epoch.expect("packs is not empty");

    let manifest = IndexManifest { version: 1, format: FORMAT_VERSION, catalog_epoch, packs: entries };
    write_manifest(&publish_dir.join(INDEX_MANIFEST_FILE), &manifest)?;
    Ok(manifest)
}

fn write_manifest<T: serde::Serialize>(path: &Path, manifest: &T) -> Result<()> {
    let json = serde_json::to_vec_pretty(manifest).with_context(|| format!("serialize {}", path.display()))?;
    fs::write(path, json).with_context(|| format!("write {}", path.display()))
}
