//! The catalog builder's second half: Gaia bins become density tiers, tiers
//! and index packs become the publish tree, and the index presets name which
//! tiers each pack is built from.

mod common;

use std::fs;
use std::io::{BufWriter, Read};
use std::path::Path;

use common::gaia_fixture::*;
use solvemyastro::bins::{self, BinRecord};
use solvemyastro::cache::build_cache;
use solvemyastro::gaia::{ingest, IngestOptions};
use solvemyastro::index::{build_index, open_preset_layers, BuildParams, Preset, FORMAT_VERSION, INDEX_FILE};
use solvemyastro::manifest::{IndexManifest, Manifest};
use solvemyastro::package::{package_index, package_tiers};
use solvemyastro::tiers::{build_tiers, cell_cum_counts, TIER_DENSITIES};
use solvemyastro::{QuadIndex, StarCache, StarRecord};

/// A tiny real tier cache (a synthetic full sky) in `dir`.
fn tiny_tier(dir: &Path, epoch: f64) {
    let mut s = 11u64;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let stars: Vec<StarRecord> = (0..8_000)
        .map(|_| StarRecord {
            ra: rnd() * 360.0,
            dec: (rnd() * 2.0 - 1.0).asin().to_degrees(),
            mag: (4.0 + rnd() * 10.0) as f32,
            pmra_mas_yr: 0.0,
            pmdec_mas_yr: 0.0,
        })
        .collect();
    build_cache(stars, dir, epoch, |_| {}).unwrap();
}

/// One pack built the way the publisher builds it, at a cheap depth.
fn tiny_pack(tiers: &Path, packs: &Path, name: &str) {
    let cache = StarCache::open(&tiers.join("tier_500")).unwrap();
    let params = BuildParams { min_depth: 2, max_depth: 2, stars_per_cell: 10, quads_per_cell: 8 };
    build_index(&[&cache], &packs.join(name), &params, |_| {}).unwrap();
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

/// The fixture's two files ingested into `<tmp>/bins`.
fn fixture_bins(tmp: &Path) -> std::path::PathBuf {
    let (gaia, bins) = (tmp.join("gaia"), tmp.join("bins"));
    fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    ingest(&gaia, &bins, &IngestOptions::default(), &|_| {}).unwrap();
    bins
}

#[test]
fn presets_pin_the_published_parameters() {
    assert_eq!(Preset::Base.name(), "base");
    assert_eq!(Preset::Narrow.name(), "narrow");
    assert_eq!(Preset::from_name("base"), Some(Preset::Base));
    assert_eq!(Preset::from_name("narrow"), Some(Preset::Narrow));
    assert_eq!(Preset::from_name("wide"), None);

    let (b, n) = (Preset::Base.params(), Preset::Narrow.params());
    let d = BuildParams::default();
    assert_eq!(
        (b.min_depth, b.max_depth, b.stars_per_cell, b.quads_per_cell),
        (d.min_depth, d.max_depth, d.stars_per_cell, d.quads_per_cell)
    );
    assert_eq!((b.min_depth, b.max_depth, b.stars_per_cell, b.quads_per_cell), (2, 7, 10, 8));
    assert_eq!((n.min_depth, n.max_depth, n.stars_per_cell, n.quads_per_cell), (8, 8, 10, 4));
    assert_eq!(Preset::Base.tier_dirs(), ["tier_500"]);
    assert_eq!(Preset::Narrow.tier_dirs(), ["tier_500", "tier_2000"]);
}

#[test]
fn narrow_preset_names_the_missing_tier() {
    let tmp = tempfile::tempdir().unwrap();
    tiny_tier(&tmp.path().join("tier_500"), 2016.0);

    let base = open_preset_layers(tmp.path(), Preset::Base).unwrap();
    assert_eq!(base.len(), 1);
    let err = open_preset_layers(tmp.path(), Preset::Narrow).err().expect("tier_2000 is missing");
    assert!(format!("{err:#}").contains("tier_2000"), "{err:#}");

    tiny_tier(&tmp.path().join("tier_2000"), 2016.0);
    assert_eq!(open_preset_layers(tmp.path(), Preset::Narrow).unwrap().len(), 2);
}

#[test]
fn package_lists_only_present_tiers() {
    let tmp = tempfile::tempdir().unwrap();
    let (tiers, publish) = (tmp.path().join("tiers"), tmp.path().join("publish"));
    tiny_tier(&tiers.join("tier_500"), 2016.0);
    tiny_tier(&tiers.join("tier_2000"), 2016.0);

    let m = package_tiers(&tiers, &publish).unwrap();
    let densities: Vec<u32> = m.tiers.iter().map(|t| t.density).collect();
    assert_eq!(densities, vec![500, 2000]);
    assert!(publish.join("tier_2000.zip").is_file());
    assert!(!publish.join("tier_5000.zip").exists());

    // Nothing to package is an error, not an empty manifest.
    let empty = tmp.path().join("empty");
    fs::create_dir(&empty).unwrap();
    assert!(package_tiers(&empty, &tmp.path().join("publish2")).is_err());
}

#[test]
fn package_writes_zip_sha_and_manifest_per_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let (tiers, publish) = (tmp.path().join("tiers"), tmp.path().join("publish"));
    let recs = vec![StarRecord { ra: 10.0, dec: 20.0, mag: 8.0, pmra_mas_yr: 0.0, pmdec_mas_yr: 0.0 }];
    build_cache(recs, &tiers.join("tier_500"), 2016.5, |_| {}).unwrap();

    let manifest = package_tiers(&tiers, &publish).unwrap();
    assert!(publish.join("tier_500.zip").is_file());
    assert!(publish.join("tier_500.zip.sha256").is_file());

    let m = json(&publish.join("manifest.json"));
    assert_eq!(m["version"], 1);
    assert_eq!(m["catalog_epoch"], 2016.5, "the first tier's header epoch");
    assert_eq!(m["tiers"][0]["density"], 500);
    assert_eq!(m["tiers"][0]["zip"], "tier_500.zip");
    assert_eq!(m["tiers"][0]["sha256"], "tier_500.zip.sha256");
    assert_eq!(m["tiers"][0]["dir"], "tier_500");
    assert_eq!(m["tiers"][0]["min_fov_deg"], 0.6);
    let zip_len = fs::metadata(publish.join("tier_500.zip")).unwrap().len();
    assert_eq!(m["tiers"][0]["size_bytes"].as_u64().unwrap(), zip_len);

    // The file is the pretty-printed manifest in struct field order; what was
    // returned is what was written.
    let text = fs::read_to_string(publish.join("manifest.json")).unwrap();
    assert!(text.starts_with("{\n  \"version\": 1,\n  \"catalog_epoch\": 2016.5,\n  \"tiers\": [\n    {\n      \"density\": 500,"));
    assert_eq!(text.as_bytes(), serde_json::to_vec_pretty(&manifest).unwrap());
    assert_eq!(Manifest::from_json_slice(text.as_bytes()).unwrap().tiers.len(), 1);

    // The sidecar is "<hex>  <zip name>\n" and the digest is the zip's.
    let sidecar = fs::read_to_string(publish.join("tier_500.zip.sha256")).unwrap();
    let digest = sidecar.split_whitespace().next().unwrap();
    assert_eq!(digest.len(), 64);
    assert!(sidecar.ends_with("  tier_500.zip\n"), "{sidecar:?}");
    assert_eq!(digest, sha256_hex(&publish.join("tier_500.zip")));

    // The zip holds exactly tier_500/stars.smac, byte for byte.
    let mut z = zip::ZipArchive::new(fs::File::open(publish.join("tier_500.zip")).unwrap()).unwrap();
    assert_eq!(z.len(), 1);
    let mut entry = z.by_index(0).unwrap();
    assert_eq!(entry.name(), "tier_500/stars.smac");
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, fs::read(tiers.join("tier_500").join("stars.smac")).unwrap());
}

fn sha256_hex(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(fs::read(path).unwrap()).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn package_index_writes_zips_sidecars_and_a_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let (tiers, packs) = (tmp.path().join("tiers"), tmp.path().join("packs"));
    tiny_tier(&tiers.join("tier_500"), 2016.0);
    tiny_pack(&tiers, &packs, "base");
    tiny_pack(&tiers, &packs, "narrow");
    let publish = tmp.path().join("publish");

    let (base, narrow) = (packs.join("base"), packs.join("narrow"));
    let manifest = package_index(&[("base", base.as_path()), ("narrow", narrow.as_path())], &publish).unwrap();
    assert_eq!(manifest.packs.len(), 2);

    for name in ["base", "narrow"] {
        assert!(publish.join(format!("index_{name}.zip")).is_file());
        let sidecar = fs::read_to_string(publish.join(format!("index_{name}.zip.sha256"))).unwrap();
        assert!(sidecar.ends_with(&format!("  index_{name}.zip\n")), "{sidecar:?}");
        assert_eq!(sidecar.split_whitespace().next().unwrap(), sha256_hex(&publish.join(format!("index_{name}.zip"))));
    }
    let m = json(&publish.join("index_manifest.json"));
    assert_eq!(m["version"], 1);
    assert_eq!(m["format"], FORMAT_VERSION);
    assert_eq!(m["catalog_epoch"], 2016.0);
    assert_eq!(m["packs"][1]["name"], "narrow");
    assert_eq!(m["packs"][0]["zip"], "index_base.zip");
    assert_eq!(m["packs"][0]["sha256"], "index_base.zip.sha256");
    assert_eq!(m["packs"][0]["dir"], "base");
    let want_fov = QuadIndex::open(&base).unwrap().min_fov_deg();
    assert_eq!(m["packs"][0]["min_fov_deg"].as_f64().unwrap(), want_fov);
    let zip_len = fs::metadata(publish.join("index_base.zip")).unwrap().len();
    assert_eq!(m["packs"][0]["size_bytes"].as_u64().unwrap(), zip_len);

    let text = fs::read_to_string(publish.join("index_manifest.json")).unwrap();
    assert!(text.starts_with("{\n  \"version\": 1,\n  \"format\": 3,\n  \"catalog_epoch\": 2016.0,\n  \"packs\": [\n"));
    assert_eq!(text.as_bytes(), serde_json::to_vec_pretty(&manifest).unwrap());
    assert_eq!(IndexManifest::from_json_slice(text.as_bytes()).unwrap().packs.len(), 2);

    // The zip holds exactly base/quads.smaq, byte for byte.
    let mut z = zip::ZipArchive::new(fs::File::open(publish.join("index_base.zip")).unwrap()).unwrap();
    assert_eq!(z.len(), 1);
    let mut entry = z.by_index(0).unwrap();
    assert_eq!(entry.name(), "base/quads.smaq");
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).unwrap();
    assert_eq!(bytes, fs::read(base.join(INDEX_FILE)).unwrap());
}

#[test]
fn package_index_takes_one_pack_and_refuses_none_a_missing_one_and_mixed_epochs() {
    let tmp = tempfile::tempdir().unwrap();
    let (tiers, packs) = (tmp.path().join("tiers"), tmp.path().join("packs"));
    tiny_tier(&tiers.join("tier_500"), 2016.0);
    tiny_pack(&tiers, &packs, "base");
    let base = packs.join("base");

    let m = package_index(&[("base", base.as_path())], &tmp.path().join("one")).unwrap();
    assert_eq!(m.packs.len(), 1);
    assert!(!tmp.path().join("one").join("index_narrow.zip").exists());

    assert!(package_index(&[], &tmp.path().join("none")).is_err());
    let missing = packs.join("narrow");
    let err = package_index(&[("narrow", missing.as_path())], &tmp.path().join("gone")).unwrap_err();
    assert!(format!("{err:#}").contains("narrow"), "{err:#}");

    // A pack built from a catalog of another epoch cannot share a manifest.
    let other = tmp.path().join("other");
    tiny_tier(&other.join("tier_500"), 2020.0);
    tiny_pack(&other, &packs, "narrow");
    let narrow = packs.join("narrow");
    let err = package_index(&[("base", base.as_path()), ("narrow", narrow.as_path())], &tmp.path().join("mixed"))
        .unwrap_err();
    assert!(format!("{err:#}").contains("epoch"), "{err:#}");
}

#[test]
fn quad_index_reports_the_epoch_it_was_built_from() {
    let tmp = tempfile::tempdir().unwrap();
    tiny_tier(&tmp.path().join("tier_500"), 2017.25);
    tiny_pack(tmp.path(), &tmp.path().join("packs"), "base");
    let ix = QuadIndex::open(&tmp.path().join("packs").join("base")).unwrap();
    assert_eq!(ix.catalog_epoch(), 2017.25);
}

#[test]
fn builds_four_disjoint_tier_caches() {
    // One dense cell (pixel 0) with 7000 mag-ascending stars, so every band is
    // used and the dense cap (6714) bites.
    let tmp = tempfile::tempdir().unwrap();
    let bins_dir = tmp.path().join("gaia_dr3");
    fs::create_dir_all(&bins_dir).unwrap();
    let mut recs: Vec<BinRecord> =
        (0..7000).map(|i| BinRecord::from_values(10.0, 20.0, 5.0 + i as f32 * 0.001, 0.0, 0.0)).collect();
    let f = fs::File::create(bins_dir.join(bins::bin_file_name(0))).unwrap();
    bins::write_records(&mut BufWriter::new(f), &mut recs).unwrap();
    let out = tmp.path().join("out");

    // No coverage record: the build warns and goes on.
    let counts = build_tiers(&bins_dir, &out, 2016.0).unwrap();
    assert_eq!(counts, vec![(500, 420), (2000, 1259), (5000, 2517), (8000, 2518)]);
    let total: usize = counts.iter().map(|(_, n)| n).sum();
    assert_eq!(total, 6714, "the dense cap, not all 7000");
    for (d, n) in counts {
        let c = StarCache::open(&out.join(format!("tier_{d}"))).unwrap();
        assert_eq!(c.star_count() as usize, n);
    }
}

#[test]
fn tiers_from_fixture_bins_are_deterministic() {
    let tmp = tempfile::tempdir().unwrap();
    let bins_dir = fixture_bins(tmp.path());
    let (one, two) = (tmp.path().join("one"), tmp.path().join("two"));

    let counts = build_tiers(&bins_dir, &one, 2016.0).unwrap();
    assert_eq!(build_tiers(&bins_dir, &two, 2016.0).unwrap(), counts);
    // Cell 0: 420 / 1 259 / 321 / 0 by rank; cell 1's 53 stars all lie in the
    // first band.
    assert_eq!(counts, vec![(500, 420 + CELL1_KEPT), (2000, 1259), (5000, 321), (8000, 0)]);
    for d in TIER_DENSITIES {
        let name = format!("tier_{d}/stars.smac");
        assert_eq!(fs::read(one.join(&name)).unwrap(), fs::read(two.join(&name)).unwrap(), "{name}");
    }

    // Each tier holds its own rank band of the bin, in the bin's order.
    let bounds = cell_cum_counts();
    let cell0 = read_bin(&bins_dir, 0);
    for (k, d) in TIER_DENSITIES.iter().enumerate() {
        let cache = StarCache::open(&one.join(format!("tier_{d}"))).unwrap();
        let got: Vec<f32> = cache.iter_pixel_records(0).unwrap().iter().map(|r| r.mag).collect();
        let hi = bounds[k + 1].min(cell0.len());
        let want: Vec<f32> = cell0[bounds[k].min(hi)..hi].iter().map(|r| r.mag()).collect();
        assert_eq!(got, want, "tier_{d}, cell 0");
        let n1 = cache.iter_pixel_records(1).unwrap().len();
        assert_eq!(n1, if k == 0 { CELL1_KEPT } else { 0 }, "tier_{d}, cell 1");
    }
}
