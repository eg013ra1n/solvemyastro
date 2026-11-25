//! The `solvemyastro` command, run as a process: the catalog chain from Gaia
//! files to a publish tree, exit codes, and the stage commands' flags.

mod common;

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use common::gaia_fixture::write_fixture;
use solvemyastro::index::{build_index, BuildParams, INDEX_FILE};
use solvemyastro::manifest::{INDEX_MANIFEST_FILE, MANIFEST_FILE};
use solvemyastro::cache::build_cache;
use solvemyastro::{StarCache, StarRecord};

/// Run the binary with `args`; the log filter is left to the command.
fn run(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_solvemyastro"))
        .args(args)
        .env_remove("RUST_LOG")
        .output()
        .expect("the binary starts")
}

fn ok(args: &[&std::ffi::OsStr]) -> Output {
    let out = run(args);
    assert!(
        out.status.success(),
        "{args:?} exited {:?}\nstderr: {}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    out
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// `&[&OsStr]` from literals and paths.
macro_rules! args {
    ($($a:expr),* $(,)?) => { &[$(std::ffi::OsStr::new(&$a)),*] };
}

/// The fixture's files ingested and cut into tiers under `root`.
fn fixture_tiers(root: &Path) -> std::path::PathBuf {
    let (gaia, bins, tiers) = (root.join("gaia"), root.join("bins"), root.join("tiers"));
    fs::create_dir_all(&gaia).unwrap();
    write_fixture(&gaia);
    ok(args!["ingest", "--gaia", gaia, "--out", bins]);
    ok(args!["build-tiers", "--bins", bins, "--out", tiers]);
    tiers
}

#[test]
fn cli_chain_runs_in_a_folder_with_a_space() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("a b");
    let (gaia, bins, tiers) = (root.join("gaia"), root.join("bins"), root.join("tiers"));
    let (packs, publish) = (root.join("packs"), root.join("publish"));
    fs::create_dir_all(&gaia).unwrap();
    write_fixture(&gaia);

    ok(args!["ingest", "--gaia", gaia, "--out", bins]);
    ok(args!["build-tiers", "--bins", bins, "--out", tiers]);
    let base = packs.join("base");
    ok(args!["build-index", "--tiers", tiers, "--out", base, "--min-depth", "2", "--max-depth", "3"]);
    let index_arg = format!("base={}", base.display());
    ok(args!["package", "--tiers", tiers, "--index", index_arg, "--out", publish]);

    let info = text(&ok(args!["info", bins]).stdout);
    assert!(info.lines().any(|l| l.starts_with("complete cells: 1")), "{info}");
    assert!(info.contains("complete by file range"), "{info}");
    let info = text(&ok(args!["info", tiers]).stdout);
    for tier in ["tier_500", "tier_2000", "tier_5000", "tier_8000"] {
        assert!(info.contains(tier), "{tier} missing from:\n{info}");
    }
    let info = text(&ok(args!["info", base]).stdout);
    assert!(info.contains("quads"), "{info}");
    assert!(info.contains("min fov"), "{info}");
    ok(args!["info", tiers.join("tier_500")]);
    ok(args!["cache-info", tiers.join("tier_500")]);
    let info = text(&ok(args!["info", publish]).stdout);
    assert!(info.contains("base"), "{info}");

    assert!(publish.join(MANIFEST_FILE).is_file());
    assert!(publish.join(INDEX_MANIFEST_FILE).is_file());
    assert!(publish.join("tier_500.zip").is_file());
    assert!(publish.join("index_base.zip").is_file());
}

#[test]
fn ingest_into_a_filled_folder_exits_1() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, bins) = (tmp.path().join("gaia"), tmp.path().join("bins"));
    fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    ok(args!["ingest", "--gaia", gaia, "--out", bins]);

    let out = run(args!["ingest", "--gaia", gaia, "--out", bins]);
    assert_eq!(out.status.code(), Some(1));
    let err = text(&out.stderr);
    assert!(err.contains("already holds bins"), "{err}");
    assert!(err.starts_with("error: ") || err.contains("\nerror: "), "{err}");
}

#[test]
fn bad_arguments_exit_2() {
    let out = run(args!["build-index", "--preset", "base", "--min-depth", "3", "--tiers", "t", "--out", "o"]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out.stderr));
    let out = run(args!["no-such-command"]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(args!["build-cache", "--out", "o"]);
    assert_eq!(out.status.code(), Some(2), "a source folder is required");
    let out = run(args!["build-cache", "--bins", "a", "--from", "b", "--out", "o"]);
    assert_eq!(out.status.code(), Some(2), "the two sources exclude each other");
}

#[test]
fn solve_without_a_catalog_exits_1() {
    let tmp = tempfile::tempdir().unwrap();
    let image = tmp.path().join("frame.fits");
    fs::write(&image, b"not a frame").unwrap();
    let out = run(args!["solve", image]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    let err = text(&out.stderr);
    assert!(err.contains("--catalog") && err.contains("--tiers"), "{err}");
}

#[test]
fn package_without_an_input_exits_1_and_writes_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let publish = tmp.path().join("publish");
    let out = run(args!["package", "--out", publish]);
    assert_eq!(out.status.code(), Some(1));
    assert!(!publish.exists());
    let out = run(args!["package", "--index", "no-equals-sign", "--out", publish]);
    assert_eq!(out.status.code(), Some(2), "{}", text(&out.stderr));
}

/// A full-sky synthetic cache in `dir`: the fixture's tiers only fill one cell,
/// which the index of a second layer cannot be told from the first by.
fn sky_tier(dir: &Path) {
    let mut s = 7u64;
    let mut rnd = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let stars: Vec<StarRecord> = (0..20_000)
        .map(|_| StarRecord {
            ra: rnd() * 360.0,
            dec: (rnd() * 2.0 - 1.0).asin().to_degrees(),
            mag: (14.0 + rnd() * 2.0) as f32,
            pmra_mas_yr: 0.0,
            pmdec_mas_yr: 0.0,
        })
        .collect();
    build_cache(stars, dir, 2016.0, |_| {}).unwrap();
}

#[test]
fn build_index_without_preset_keeps_todays_layers() {
    let tmp = tempfile::tempdir().unwrap();
    let tiers = fixture_tiers(tmp.path());
    // The fixture's tier_500 and a tier_2000 that covers the whole sky.
    fs::remove_dir_all(tiers.join("tier_2000")).unwrap();
    sky_tier(&tiers.join("tier_2000"));
    let out_dir = tmp.path().join("cli-pack");
    ok(args![
        "build-index", "--tiers", tiers, "--out", out_dir, "--min-depth", "2", "--max-depth", "2",
        "--stars-per-cell", "12", "--quads-per-cell", "5"
    ]);

    let (l500, l2000) = (
        StarCache::open(&tiers.join("tier_500")).unwrap(),
        StarCache::open(&tiers.join("tier_2000")).unwrap(),
    );
    let params = BuildParams { min_depth: 2, max_depth: 2, stars_per_cell: 12, quads_per_cell: 5 };
    let direct = tmp.path().join("direct-pack");
    build_index(&[&l500, &l2000], &direct, &params, |_| {}).unwrap();
    let built = fs::read(out_dir.join(INDEX_FILE)).unwrap();
    assert_eq!(built, fs::read(direct.join(INDEX_FILE)).unwrap());

    // The second layer is part of the pack: the test can tell it from the base layer alone.
    let alone = tmp.path().join("alone-pack");
    build_index(&[&l500], &alone, &params, |_| {}).unwrap();
    assert!(built != fs::read(alone.join(INDEX_FILE)).unwrap(), "tier_2000 changes nothing here");
}

#[test]
fn build_index_preset_names_the_missing_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let tiers = fixture_tiers(tmp.path());
    fs::remove_dir_all(tiers.join("tier_2000")).unwrap();
    let out = run(args!["build-index", "--preset", "narrow", "--tiers", tiers, "--out", tmp.path().join("p")]);
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("tier_2000"), "{}", text(&out.stderr));
}

#[test]
fn log_file_gets_json_lines_and_quiet_silences_progress() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, bins, log) = (tmp.path().join("gaia"), tmp.path().join("bins"), tmp.path().join("run.jsonl"));
    fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    let out = ok(args!["--log", log, "-q", "ingest", "--gaia", gaia, "--out", bins]);
    assert!(!text(&out.stderr).contains("INFO"), "{}", text(&out.stderr));

    let lines = fs::read_to_string(&log).unwrap();
    assert!(!lines.trim().is_empty(), "the log file is empty");
    for line in lines.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
        assert!(v.get("fields").is_some(), "{line}");
    }
    assert!(lines.contains("gaia ingest complete"), "{lines}");
}

#[test]
fn build_cache_takes_bins_or_from() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, bins) = (tmp.path().join("gaia"), tmp.path().join("bins"));
    fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    ok(args!["ingest", "--gaia", gaia, "--out", bins]);
    let (a, b) = (tmp.path().join("a"), tmp.path().join("b"));
    ok(args!["build-cache", "--bins", bins, "--out", a]);
    ok(args!["build-cache", "--from", bins, "--out", b]);
    assert_eq!(fs::read(a.join("stars.smac")).unwrap(), fs::read(b.join("stars.smac")).unwrap());
    let info = text(&ok(args!["info", a]).stdout);
    assert!(info.contains("stars"), "{info}");
}

#[test]
fn build_bright_cache_writes_a_sidecar() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, bins, deep, bright) =
        (tmp.path().join("gaia"), tmp.path().join("bins"), tmp.path().join("deep"), tmp.path().join("bright"));
    fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    ok(args!["ingest", "--gaia", gaia, "--out", bins]);
    ok(args!["build-cache", "--bins", bins, "--out", deep]);
    ok(args!["build-bright-cache", "--from", deep, "--out", bright, "--target-density", "50", "--mode", "cap-only"]);
    let meta = fs::read_to_string(bright.join("bright.meta.json")).unwrap();
    assert!(meta.contains("\"mode\": \"cap-only\""), "{meta}");
    let out = run(args!["build-bright-cache", "--from", deep, "--out", bright, "--mode", "sideways"]);
    assert_eq!(out.status.code(), Some(2));
}
