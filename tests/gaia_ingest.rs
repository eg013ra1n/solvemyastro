//! Gaia ingest: a folder of `GaiaSource_*.csv.gz` files becomes bins, with a
//! coverage record, whatever the file order or thread count.

mod common;

use std::path::Path;
use std::sync::Mutex;

use common::gaia_fixture::*;
use md5::{Digest, Md5};
use solvemyastro::bins::BinRecord;
use solvemyastro::gaia::{ingest, Coverage, IngestOptions, IngestProgress, COVERAGE_FILE, MD5_LIST_FILE};

fn run(gaia: &Path, out: &Path, opts: &IngestOptions) -> anyhow::Result<solvemyastro::gaia::IngestSummary> {
    ingest(gaia, out, opts, &|_| {})
}

fn md5_hex(path: &Path) -> String {
    let digest = Md5::digest(std::fs::read(path).unwrap());
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn sort_key(r: &BinRecord) -> (u16, u32, u32, i16, i16) {
    (r.mag_raw, r.ra.to_bits(), r.dec.to_bits(), r.pmra_raw, r.pmdec_raw)
}

fn has_star(records: &[BinRecord], cell: u64, k: u32) -> Option<&BinRecord> {
    let (ra, dec) = bin_position(cell, k);
    records.iter().find(|r| r.ra == ra && r.dec == dec)
}

#[test]
fn ingest_keeps_filters_and_sorts() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, out) = (tmp.path().join("gaia"), tmp.path().join("out"));
    std::fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);

    let events = Mutex::new(Vec::new());
    // One job: the files are read in name order, so the pair order below
    // is the sorter's work and not a thread race.
    let opts = IngestOptions { jobs: 1, ..Default::default() };
    let summary = ingest(&gaia, &out, &opts, &|e| events.lock().unwrap().push(e)).unwrap();

    assert_eq!(summary.files, 2);
    assert_eq!(summary.stars, CELL0_STARS as usize + CELL1_KEPT);
    let names: Vec<_> = bin_bytes(&out).into_keys().collect();
    assert_eq!(names, vec!["healpix_000000.bin", "healpix_000001.bin"]);

    let cell0 = read_bin(&out, 0);
    let cell1 = read_bin(&out, 1);
    assert_eq!(cell0.len(), CELL0_STARS as usize);
    assert_eq!(cell1.len(), CELL1_KEPT);

    // The filters: G at the limit, RUWE at the cap and a malformed ra are gone;
    // an empty RUWE, a RUWE under the cap and a null proper motion stay.
    assert!(has_star(&cell1, 1, K_G_AT_LIMIT).is_none(), "G 21.0 at the default limit");
    assert!(has_star(&cell1, 1, K_RUWE_AT_CAP).is_none(), "RUWE 1.4");
    assert!(cell1.iter().all(|r| r.dec != bin_position(1, K_BAD_RA).1), "malformed ra");
    assert!(has_star(&cell1, 1, K_EMPTY_RUWE).is_some(), "empty RUWE");
    assert!(has_star(&cell1, 1, K_RUWE_UNDER_CAP).is_some(), "RUWE 1.39");
    let null_pm = has_star(&cell1, 1, K_NULL_PM).expect("null proper motion kept");
    assert_eq!((null_pm.pmra_raw, null_pm.pmdec_raw), (0, 0));
    let plain = has_star(&cell1, 1, 0).unwrap();
    assert_eq!((plain.pmra_raw, plain.pmdec_raw), (150, -250));

    // Every bin is sorted by the SA-6 key.
    for recs in [&cell0, &cell1] {
        assert!(recs.windows(2).all(|w| sort_key(&w[0]) <= sort_key(&w[1])));
    }
    // The equal-G pairs came from different files and still order by position.
    let pair_g = 9000u16;
    let pair: Vec<_> = cell0.iter().filter(|r| r.mag_raw == pair_g).collect();
    assert_eq!(pair.len(), 2);
    assert!(pair[0].ra.to_bits() < pair[1].ra.to_bits());
    // Star 1 is from file B and has the smaller ra: file order would put star 0
    // (file A) first, so only the sort puts it second.
    assert_eq!((pair[0].ra, pair[0].dec), bin_position(0, 1));
    assert_eq!((pair[1].ra, pair[1].dec), bin_position(0, 0));

    // Progress: started, one FileDone per file, finalizing, complete.
    let events = events.into_inner().unwrap();
    assert!(matches!(events.first(), Some(IngestProgress::Started { total_files: 2 })));
    assert_eq!(events.iter().filter(|e| matches!(e, IngestProgress::FileDone { .. })).count(), 2);
    assert!(matches!(events[events.len() - 2], IngestProgress::Finalizing));
    assert!(matches!(events.last(), Some(IngestProgress::Complete { total_stars }) if *total_stars == summary.stars));
    assert!(!events.iter().any(|e| matches!(e, IngestProgress::Verified { .. })), "no verify by default");
    assert!(!out.join(".scratch").exists(), "scratch removed at the end");
}

#[test]
fn ingest_is_the_same_for_any_jobs_and_order() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, gaia_listed) = (tmp.path().join("gaia"), tmp.path().join("gaia_listed"));
    for d in [&gaia, &gaia_listed] {
        std::fs::create_dir(d).unwrap();
        write_fixture(d);
    }
    // B before A: the other file order.
    std::fs::write(
        gaia_listed.join(MD5_LIST_FILE),
        format!(
            "{}  {FILE_B}\n{}  {FILE_A}\n",
            md5_hex(&gaia_listed.join(FILE_B)),
            md5_hex(&gaia_listed.join(FILE_A))
        ),
    )
    .unwrap();

    let one = tmp.path().join("one");
    let four = tmp.path().join("four");
    let listed = tmp.path().join("listed");
    run(&gaia, &one, &IngestOptions { jobs: 1, ..Default::default() }).unwrap();
    run(&gaia, &four, &IngestOptions { jobs: 4, ..Default::default() }).unwrap();
    run(&gaia_listed, &listed, &IngestOptions { jobs: 1, ..Default::default() }).unwrap();

    let reference = bin_bytes(&one);
    assert_eq!(reference.len(), 2);
    assert_eq!(reference, bin_bytes(&four), "jobs 1 vs 4");
    assert_eq!(reference, bin_bytes(&listed), "reverse list order");
    let cov = std::fs::read(one.join(COVERAGE_FILE)).unwrap();
    assert_eq!(cov, std::fs::read(four.join(COVERAGE_FILE)).unwrap());
    assert_eq!(cov, std::fs::read(listed.join(COVERAGE_FILE)).unwrap());
}

#[test]
fn coverage_reports_complete_and_partial_cells() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, out) = (tmp.path().join("gaia"), tmp.path().join("out"));
    std::fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    let summary = run(&gaia, &out, &IngestOptions::default()).unwrap();

    for cov in [summary.coverage.clone(), Coverage::read(&out).unwrap().expect("record on disk")] {
        assert_eq!(cov.level, 8);
        assert!(cov.unknown.is_empty());
        assert_eq!(cov.complete_cells(), Some(vec![0]));
        assert_eq!(cov.partial_cells(), Some(vec![1]));
    }
}

#[test]
fn a_file_without_a_range_makes_coverage_unknown() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, out) = (tmp.path().join("gaia"), tmp.path().join("out"));
    std::fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    let mine = [Row::star(star_in_cell(5, 0).0, star_in_cell(5, 0).1, "14.0")];
    write_gaia_file(&gaia, "mine.csv.gz", &mine);

    let summary = run(&gaia, &out, &IngestOptions::default()).unwrap();
    assert_eq!(summary.files, 3);
    assert_eq!(read_bin(&out, 5).len(), 1, "its star is ingested");
    let cov = Coverage::read(&out).unwrap().unwrap();
    assert_eq!(cov.unknown, vec!["mine.csv.gz".to_string()]);
    assert_eq!(cov.complete_cells(), None);
    assert_eq!(cov.partial_cells(), None);
}

#[test]
fn stale_scratch_is_cleared_and_filled_output_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, out) = (tmp.path().join("gaia"), tmp.path().join("out"));
    std::fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);

    // A leftover of a crashed run: one record that no file holds.
    std::fs::create_dir_all(out.join(".scratch")).unwrap();
    let (ra, dec) = bin_position(0, 2100);
    let stale = BinRecord::from_values(ra, dec, 5.0, 0.0, 0.0);
    let mut f = std::fs::File::create(out.join(".scratch").join("p_000000.raw")).unwrap();
    stale.write_to(&mut f).unwrap();
    drop(f);

    run(&gaia, &out, &IngestOptions::default()).unwrap();
    let cell0 = read_bin(&out, 0);
    assert_eq!(cell0.len(), CELL0_STARS as usize);
    assert!(cell0.iter().all(|r| r.mag_raw != 5000), "the stale record is gone");
    assert!(!out.join(".scratch").exists());

    let before = bin_bytes(&out);
    let err = run(&gaia, &out, &IngestOptions::default()).unwrap_err();
    let text = format!("{err:#}");
    assert!(text.contains("already holds bins"), "{text}");
    assert!(text.contains("choose an empty folder"), "{text}");
    assert_eq!(before, bin_bytes(&out), "a refused run changes nothing");
}

#[test]
fn a_truncated_gzip_fails_with_its_name_and_no_coverage() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, out) = (tmp.path().join("gaia"), tmp.path().join("out"));
    std::fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    let bytes = std::fs::read(gaia.join(FILE_B)).unwrap();
    std::fs::write(gaia.join(FILE_B), &bytes[..bytes.len() / 2]).unwrap();

    let err = run(&gaia, &out, &IngestOptions { jobs: 2, ..Default::default() }).unwrap_err();
    assert!(err.to_string().contains(FILE_B), "{err:#}");
    assert!(!out.join(COVERAGE_FILE).exists(), "no coverage after a failed run");
    assert!(!out.join(".scratch").exists(), "scratch cleaned after a failed run");
}

#[test]
fn md5_verify_catches_a_changed_file_and_parses_crlf() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, out) = (tmp.path().join("gaia"), tmp.path().join("out"));
    std::fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    let list = format!(
        "{}  {FILE_A}\r\n{}  {FILE_B}\r\n",
        md5_hex(&gaia.join(FILE_A)),
        md5_hex(&gaia.join(FILE_B))
    );
    std::fs::write(gaia.join(MD5_LIST_FILE), list).unwrap();
    let verify = IngestOptions { verify_md5: true, jobs: 2, ..Default::default() };

    // Whole files, CRLF list: every file is verified, then ingested.
    let events = Mutex::new(Vec::new());
    ingest(&gaia, &out, &verify, &|e| events.lock().unwrap().push(e)).unwrap();
    let events = events.into_inner().unwrap();
    assert_eq!(events.iter().filter(|e| matches!(e, IngestProgress::Verified { .. })).count(), 2);
    let first_file_done = events.iter().position(|e| matches!(e, IngestProgress::FileDone { .. })).unwrap();
    let last_verified = events.iter().rposition(|e| matches!(e, IngestProgress::Verified { .. })).unwrap();
    assert!(last_verified < first_file_done, "all files are checked before the first is read");

    // One file changed after it was listed.
    let out2 = tmp.path().join("out2");
    let mut changed = std::fs::read(gaia.join(FILE_B)).unwrap();
    changed.push(0);
    std::fs::write(gaia.join(FILE_B), changed).unwrap();
    let err = run(&gaia, &out2, &verify).unwrap_err();
    assert!(err.to_string().contains(FILE_B), "{err:#}");
    assert!(bin_bytes(&out2).is_empty(), "nothing is binned after a failed check");
    assert!(!out2.join(COVERAGE_FILE).exists());
    // Without the check the same folder is read (the change sits after the gzip stream).
    run(&gaia, &tmp.path().join("out3"), &IngestOptions::default()).unwrap();

    // Verification asked for, and no list.
    let bare = tmp.path().join("bare");
    std::fs::create_dir(&bare).unwrap();
    write_fixture(&bare);
    let err = run(&bare, &tmp.path().join("out4"), &verify).unwrap_err();
    assert!(err.to_string().contains(MD5_LIST_FILE), "{err:#}");
}

#[test]
fn a_folder_without_data_files_is_an_error() {
    let tmp = tempfile::tempdir().unwrap();
    let gaia = tmp.path().join("gaia");
    std::fs::create_dir(&gaia).unwrap();
    std::fs::write(gaia.join("notes.txt"), "x").unwrap();
    let err = run(&gaia, &tmp.path().join("out"), &IngestOptions::default()).unwrap_err();
    assert!(err.to_string().contains("no .csv.gz files"), "{err:#}");
}

#[test]
fn the_magnitude_limit_is_an_option() {
    let tmp = tempfile::tempdir().unwrap();
    let (gaia, out) = (tmp.path().join("gaia"), tmp.path().join("out"));
    std::fs::create_dir(&gaia).unwrap();
    write_fixture(&gaia);
    run(&gaia, &out, &IngestOptions { mag_limit: 13.0, ..Default::default() }).unwrap();
    for pixel in [0, 1] {
        let recs = read_bin(&out, pixel);
        assert!(!recs.is_empty());
        assert!(recs.iter().all(|r| r.mag() < 13.0));
    }
}
