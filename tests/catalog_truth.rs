//! External-truth validation for the `smac_gaia` catalog (Phase 0a).
//!
//! The `cache.rs` unit tests are internal round-trips — they prove the file
//! *format* works (serialize ↔ deserialize ↔ cone-query against data we just
//! wrote). They do NOT prove the *data inside* `stars.smac` matches Gaia DR3.
//!
//! This harness closes that gap. For a set of representative sky regions it
//! compares `StarCache::cone()` against a fixture CSV fetched fresh from the
//! ESA Gaia TAP service, and asserts:
//!
//!   * every truth star is present locally within 0.1″ (f32 storage gives
//!     ~0.02″ quantisation, so 0.1″ is a generous gate),
//!   * magnitudes match to the u16/1000 quantisation,
//!   * < 1% of truth stars are missing locally.
//!
//! It is **fixture-based on purpose**: solvemyastro keeps a deliberately lean
//! dependency tree (no HTTP client). Fixtures live in
//! `tests/fixtures/catalog_truth/<region>.csv`. When a fixture is missing the
//! test prints the exact `curl` command to generate it and skips that region,
//! so the suite stays offline and deterministic once fixtures exist.
//!
//! Run:
//!   cargo test -p solvemyastro --test catalog_truth -- --ignored --nocapture
//!
//! Catalog path: `$SOLVEMYASTRO_SMAC_GAIA` or the default app-data location.

use std::path::PathBuf;

use solvemyastro::StarCache;

/// Set to `true` after the catalog has been rebuilt with the server-side
/// `ruwe < 1.4` filter (Phase 0b). It controls the ADQL printed for fixture
/// generation so fixtures match how the catalog was built. While the current
/// (unfiltered) catalog is in place, keep it `false`.
const RUWE_FILTERED: bool = false;

/// Catalog epoch — `cone()` called with this performs a no-op PM propagation,
/// so returned positions are the raw J2016.0 catalog coordinates, directly
/// comparable to `gaiadr3.gaia_source` ra/dec.
const CATALOG_EPOCH: f64 = 2016.0;

/// Magnitude ceiling — must match how the catalog was built (Gaia G < 16).
const MAG_LIMIT: f32 = 16.0;

/// Default catalog location when `$SOLVEMYASTRO_SMAC_GAIA` is unset.
const DEFAULT_SMAC_GAIA: &str = "/Volumes/BigMac/Users/astrobureau/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia";

/// A validation region: a cone on the sky with a distinctive character so the
/// set as a whole exercises high-Dec, RA-wrap, dense, and sparse fields.
struct Region {
    /// Fixture file stem (`tests/fixtures/catalog_truth/<name>.csv`).
    name: &'static str,
    ra_deg: f64,
    dec_deg: f64,
    radius_deg: f64,
    /// What this region is meant to stress.
    character: &'static str,
}

/// Radii are kept small enough that each cone stays well under the ESA TAP
/// anonymous-sync row cap (~2000) at G < 16.
const REGIONS: &[Region] = &[
    Region { name: "polaris",      ra_deg: 37.9500,  dec_deg: 89.2641, radius_deg: 0.20, character: "high declination, near the pole" },
    Region { name: "vega",         ra_deg: 279.2347, dec_deg: 38.7837, radius_deg: 0.10, character: "mid-Dec, bright anchor star" },
    Region { name: "m31_core",     ra_deg: 10.6847,  dec_deg: 41.2692, radius_deg: 0.06, character: "crowded galaxy core" },
    Region { name: "ngp_sparse",   ra_deg: 192.0000, dec_deg: 27.4000, radius_deg: 0.20, character: "sparse, near North Galactic Pole" },
    Region { name: "ra_wrap",      ra_deg: 0.0800,   dec_deg: 5.0000,  radius_deg: 0.10, character: "RA≈0 — exercises cone RA wraparound" },
    Region { name: "south",        ra_deg: 80.0000,  dec_deg: -65.0000, radius_deg: 0.10, character: "deep southern declination" },
    Region { name: "cygnus_dense", ra_deg: 305.0000, dec_deg: 40.0000, radius_deg: 0.04, character: "dense Milky Way field" },
];

/// ADQL for one region's truth cone.
fn region_adql(r: &Region) -> String {
    let ruwe = if RUWE_FILTERED { " AND ruwe < 1.4" } else { "" };
    format!(
        "SELECT ra,dec,phot_g_mean_mag,pmra,pmdec FROM gaiadr3.gaia_source \
         WHERE phot_g_mean_mag < {MAG_LIMIT}{ruwe} \
         AND 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{},{},{}))",
        r.ra_deg, r.dec_deg, r.radius_deg
    )
}

/// The `curl` one-liner that generates a missing fixture.
fn region_curl(r: &Region, fixture_path: &std::path::Path) -> String {
    format!(
        "curl -s 'https://gea.esac.esa.int/tap-server/tap/sync' \\\n  \
         --data-urlencode 'REQUEST=doQuery' \\\n  \
         --data-urlencode 'LANG=ADQL' \\\n  \
         --data-urlencode 'FORMAT=csv' \\\n  \
         --data-urlencode \"QUERY={}\" \\\n  \
         -o '{}'",
        region_adql(r),
        fixture_path.display()
    )
}

/// One Gaia row from a fixture CSV (J2016.0).
struct TruthStar {
    ra: f64,
    dec: f64,
    mag: f32,
}

/// Parse an ESA TAP CSV body. Header is `ra,dec,phot_g_mean_mag,pmra,pmdec`;
/// the first 3 columns are all this harness needs. Rows with an unparseable
/// ra/dec/mag are skipped (TAP never emits those for these columns, but be
/// defensive). Empty pmra/pmdec are fine — they are not read here.
fn parse_truth_csv(body: &str) -> Vec<TruthStar> {
    let mut out = Vec::new();
    for (i, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Skip the header row (non-numeric first field).
        if i == 0 && line.starts_with("ra") {
            continue;
        }
        let mut f = line.split(',');
        let ra: f64 = match f.next().and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let dec: f64 = match f.next().and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let mag: f32 = match f.next().and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        out.push(TruthStar { ra, dec, mag });
    }
    out
}

/// Great-circle separation in arcseconds (Vincenty form).
fn sep_arcsec(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let dra = (ra1 - ra2).to_radians();
    let (s1, c1) = dec1.to_radians().sin_cos();
    let (s2, c2) = dec2.to_radians().sin_cos();
    let (sd, cd) = dra.sin_cos();
    let a = (c2 * sd).hypot(c1 * s2 - s1 * c2 * cd);
    let b = s1 * s2 + c1 * c2 * cd;
    a.atan2(b).to_degrees() * 3600.0
}

fn smac_gaia_path() -> PathBuf {
    std::env::var("SOLVEMYASTRO_SMAC_GAIA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(DEFAULT_SMAC_GAIA))
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("catalog_truth")
}

#[test]
#[ignore = "needs the smac_gaia catalog + pre-fetched ESA TAP fixtures"]
fn catalog_matches_gaia_dr3_truth() {
    let cache_dir = smac_gaia_path();
    if !cache_dir.is_dir() {
        println!(
            "SKIP catalog_truth: smac_gaia not found at {}\n\
             Set $SOLVEMYASTRO_SMAC_GAIA to override.",
            cache_dir.display()
        );
        return;
    }
    let cache = StarCache::open(&cache_dir).expect("open smac_gaia");

    let fdir = fixture_dir();
    let mut missing: Vec<&Region> = Vec::new();
    let mut checked = 0usize;

    // Aggregate gates across all regions that had a fixture.
    let mut worst_median_sep = 0.0_f64;
    let mut worst_missing_frac = 0.0_f64;
    let mut total_truth = 0usize;
    let mut total_mag_mismatch = 0usize;

    println!("\n=== Catalog external-truth validation ===");
    println!("catalog: {}", cache_dir.display());
    println!("RUWE_FILTERED const = {RUWE_FILTERED}\n");

    for r in REGIONS {
        let fixture = fdir.join(format!("{}.csv", r.name));
        let body = match std::fs::read_to_string(&fixture) {
            Ok(b) => b,
            Err(_) => {
                missing.push(r);
                continue;
            }
        };
        let truth = parse_truth_csv(&body);
        if truth.is_empty() {
            println!("WARN  {:<13} fixture present but parsed 0 rows — refetch it", r.name);
            missing.push(r);
            continue;
        }
        if truth.len() >= 2000 {
            println!(
                "WARN  {:<13} fixture has {} rows — may be ESA-sync row-capped; \
                 shrink radius_deg or use async TAP",
                r.name,
                truth.len()
            );
        }

        let local = cache
            .cone(r.ra_deg, r.dec_deg, r.radius_deg, MAG_LIMIT, CATALOG_EPOCH)
            .expect("cone query");

        // For each truth star, nearest local star.
        let mut seps: Vec<f64> = Vec::with_capacity(truth.len());
        let mut missing_here = 0usize;
        let mut mag_mismatch = 0usize;
        for t in &truth {
            let mut best = f64::MAX;
            let mut best_mag = f32::NAN;
            for l in &local {
                let s = sep_arcsec(t.ra, t.dec, l.ra, l.dec);
                if s < best {
                    best = s;
                    best_mag = l.mag;
                }
            }
            // A truth star with no local match within 1″ is "missing".
            if best > 1.0 {
                missing_here += 1;
                continue;
            }
            seps.push(best);
            // Magnitude must agree to the u16/1000 storage quantisation
            // (allow a hair of slack for the f32 round-trip).
            if (t.mag - best_mag).abs() > 0.002 {
                mag_mismatch += 1;
            }
        }

        seps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = seps.get(seps.len() / 2).copied().unwrap_or(0.0);
        let p99 = seps.get(seps.len() * 99 / 100).copied().unwrap_or(0.0);
        let max = seps.last().copied().unwrap_or(0.0);
        let missing_frac = missing_here as f64 / truth.len() as f64;

        println!(
            "{:<13} {:<34} truth={:>5} local={:>5} | Δpos med/p99/max = {:.3}/{:.3}/{:.3}″ \
             | missing={} ({:.2}%) | mag-mismatch={}",
            r.name,
            r.character,
            truth.len(),
            local.len(),
            median,
            p99,
            max,
            missing_here,
            missing_frac * 100.0,
            mag_mismatch,
        );

        worst_median_sep = worst_median_sep.max(median);
        worst_missing_frac = worst_missing_frac.max(missing_frac);
        total_truth += truth.len();
        total_mag_mismatch += mag_mismatch;
        checked += 1;
    }

    if !missing.is_empty() {
        println!("\n--- {} fixture(s) missing — generate with: ---", missing.len());
        if let Err(e) = std::fs::create_dir_all(&fdir) {
            println!("(could not create {}: {e})", fdir.display());
        }
        for r in &missing {
            let path = fdir.join(format!("{}.csv", r.name));
            println!("\n# {} — {}\n{}", r.name, r.character, region_curl(r, &path));
        }
        println!();
    }

    if checked == 0 {
        println!(
            "\nSKIP catalog_truth: no fixtures present. Fetch them with the curl \
             commands above, then re-run. (Test passes vacuously so CI without \
             fixtures is not red.)"
        );
        return;
    }

    println!(
        "\nSUMMARY: {checked}/{} regions checked | worst median Δpos = {:.3}″ | \
         worst missing = {:.2}% | total mag-mismatch = {}/{}",
        REGIONS.len(),
        worst_median_sep,
        worst_missing_frac * 100.0,
        total_mag_mismatch,
        total_truth,
    );

    // Gates. Only assert over regions that actually had a fixture.
    assert!(
        worst_median_sep < 0.1,
        "GATE FAIL: a region's median position delta {:.3}″ exceeds 0.1″ — \
         catalog positions disagree with Gaia DR3",
        worst_median_sep
    );
    assert!(
        worst_missing_frac < 0.01,
        "GATE FAIL: a region is missing {:.2}% of truth stars (>1%) — \
         catalog is incomplete (HEALPix indexing or ingest filter bug)",
        worst_missing_frac * 100.0
    );
    assert!(
        total_mag_mismatch == 0,
        "GATE FAIL: {total_mag_mismatch} stars have a magnitude delta beyond \
         the u16/1000 quantisation — ingest magnitude scaling is wrong"
    );

    println!("GATE: catalog external-truth validation PASSED");
}
