//! A tiny Gaia DR3 download: two `GaiaSource_*.csv.gz` files whose ranges make
//! level-6 cell 0 complete and cell 1 partial, with generated stars in both
//! cells and one row for every filter the ingest applies. File A's range lies
//! wholly in cell 0, so every cell 1 row is in file B.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use flate2::write::GzEncoder;
use flate2::Compression;
use solvemyastro::bins::BinRecord;
use solvemyastro::healpix;

/// Level-8 pixels 0..=7 (half of cell 0).
pub const FILE_A: &str = "GaiaSource_000000-000007.csv.gz";
/// Level-8 pixels 8..=20 (the rest of cell 0 and the first five of cell 1).
pub const FILE_B: &str = "GaiaSource_000008-000020.csv.gz";

/// Generated stars in cell 0: the tiers split them 420 / 1 259 / 321 / 0.
pub const CELL0_STARS: u32 = 2000;
/// Generated stars in cell 1.
pub const CELL1_STARS: u32 = 50;
/// What cell 1 keeps: its generated stars and the three filter rows that pass
/// (empty RUWE, null proper motion, RUWE just under the cap).
pub const CELL1_KEPT: usize = 53;
/// Pairs of cell 0 stars with equal G, one of each pair in each file.
pub const EQUAL_G_PAIRS: u32 = 20;

/// `k` of the filter rows in cell 1.
pub const K_G_AT_LIMIT: u32 = 100;
pub const K_RUWE_AT_CAP: u32 = 101;
pub const K_EMPTY_RUWE: u32 = 102;
pub const K_NULL_PM: u32 = 103;
pub const K_BAD_RA: u32 = 104;
pub const K_RUWE_UNDER_CAP: u32 = 105;

/// One CSV row, every column as the text the file holds.
#[derive(Clone, Debug)]
pub struct Row {
    pub ra: String,
    pub dec: String,
    pub parallax: String,
    pub g: String,
    pub pmra: String,
    pub pmdec: String,
    pub ruwe: String,
}

impl Row {
    /// A good star: proper motion 1.5 / -2.5 mas/yr, RUWE 1.0.
    pub fn star(ra: f64, dec: f64, g: &str) -> Self {
        Self {
            ra: format!("{ra:.7}"),
            dec: format!("{dec:.7}"),
            parallax: "0.5".into(),
            g: g.into(),
            pmra: "1.5".into(),
            pmdec: "-2.5".into(),
            ruwe: "1.0".into(),
        }
    }
}

/// Write a gzip with a `#` preamble and a header naming the columns in a
/// different order than the ingest reads them, so the column lookup is used.
pub fn write_gaia_file(dir: &Path, name: &str, rows: &[Row]) {
    let f = File::create(dir.join(name)).unwrap();
    let mut gz = GzEncoder::new(f, Compression::fast());
    writeln!(gz, "# Gaia DR3 test data").unwrap();
    writeln!(gz, "# not the real archive").unwrap();
    writeln!(gz, "source_id,ra,dec,parallax,phot_g_mean_mag,pmra,pmdec,ruwe").unwrap();
    for (i, r) in rows.iter().enumerate() {
        writeln!(gz, "{},{},{},{},{},{},{},{}", 1000 + i, r.ra, r.dec, r.parallax, r.g, r.pmra, r.pmdec, r.ruwe)
            .unwrap();
    }
    gz.finish().unwrap();
}

/// A position inside level-6 cell `cell`, distinct for each `k` below 2 200: a
/// sunflower spiral of up to 0.25 degrees around the cell's centre.
pub fn star_in_cell(cell: u64, k: u32) -> (f64, f64) {
    let (ra0, dec0) = healpix::pixel_center(6, cell);
    let r = 0.25 * ((k as f64 + 0.5) / 2200.0).sqrt();
    let a = k as f64 * 2.399_963_229_728_653;
    let dec = dec0 + r * a.sin();
    let ra = ra0 + r * a.cos() / dec.to_radians().cos();
    // Seven decimals, as the row text holds them: the file and the test agree.
    let (ra, dec) = ((ra * 1e7).round() / 1e7, (dec * 1e7).round() / 1e7);
    // The ingest bins the f32 values.
    assert_eq!(healpix::sky_to_pixel(ra as f32 as f64, dec as f32 as f64), cell, "star {k} left cell {cell}");
    (ra, dec)
}

/// The f32 position of `star_in_cell`, as a bin holds it.
pub fn bin_position(cell: u64, k: u32) -> (f32, f32) {
    let (ra, dec) = star_in_cell(cell, k);
    (ra as f32, dec as f32)
}

/// G of cell 0's star `k`: pairs `(2i, 2i + 1)` for `i < EQUAL_G_PAIRS` share
/// theirs, and `2i` goes to file A, `2i + 1` to file B.
fn cell0_g(k: u32) -> String {
    if k < 2 * EQUAL_G_PAIRS {
        format!("{:.3}", 9.0 + (k / 2) as f64 * 0.01)
    } else {
        format!("{:.3}", 10.0 + k as f64 * 0.004)
    }
}

/// The rows of the two files, `(file A, file B)`.
pub fn fixture_rows() -> (Vec<Row>, Vec<Row>) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for k in 0..CELL0_STARS {
        let (ra, dec) = star_in_cell(0, k);
        let row = Row::star(ra, dec, &cell0_g(k));
        if k % 2 == 0 { a.push(row) } else { b.push(row) }
    }
    for k in 0..CELL1_STARS {
        let (ra, dec) = star_in_cell(1, k);
        b.push(Row::star(ra, dec, &format!("{:.3}", 12.0 + k as f64 * 0.1)));
    }
    let case = |k: u32, g: &str| {
        let (ra, dec) = star_in_cell(1, k);
        Row::star(ra, dec, g)
    };
    let mut at_limit = case(K_G_AT_LIMIT, "21.0");
    at_limit.pmra = "3".into();
    let mut ruwe_cap = case(K_RUWE_AT_CAP, "15.4");
    ruwe_cap.ruwe = "1.4".into();
    let mut empty_ruwe = case(K_EMPTY_RUWE, "15.5");
    empty_ruwe.ruwe = String::new();
    let mut null_pm = case(K_NULL_PM, "15.6");
    null_pm.pmra = "null".into();
    null_pm.pmdec = String::new();
    let mut bad_ra = case(K_BAD_RA, "15.7");
    bad_ra.ra = "not-a-number".into();
    let mut ruwe_under = case(K_RUWE_UNDER_CAP, "15.8");
    ruwe_under.ruwe = "1.39".into();
    b.extend([at_limit, ruwe_cap, empty_ruwe, null_pm, bad_ra, ruwe_under]);
    (a, b)
}

/// Write both files into `dir`.
pub fn write_fixture(dir: &Path) {
    let (a, b) = fixture_rows();
    write_gaia_file(dir, FILE_A, &a);
    write_gaia_file(dir, FILE_B, &b);
}

/// Every `healpix_NNNNNN.bin` of a folder, by file name.
pub fn bin_bytes(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut out = BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        if name.starts_with("healpix_") && name.ends_with(".bin") {
            out.insert(name, std::fs::read(&p).unwrap());
        }
    }
    out
}

/// The records of one bin, in file order.
pub fn read_bin(dir: &Path, pixel: u64) -> Vec<BinRecord> {
    let bytes = std::fs::read(dir.join(solvemyastro::bins::bin_file_name(pixel))).unwrap();
    solvemyastro::bins::read_records_until_mag(&bytes, 100.0)
}
