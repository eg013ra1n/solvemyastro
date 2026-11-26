//! Corpus acceptance gate for solvemyastro.
//!
//! Run with:
//!   cargo test -p solvemyastro --release --test corpus_bench -- --ignored --nocapture
//!
//! Data paths default to a layout under `$HOME`; override with
//! `SOLVEMYASTRO_BENCH_DIR` (default `$HOME/Pictures/Astro/Platesolve Bench`),
//! `SOLVEMYASTRO_SMAC_GAIA` (default `$HOME/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia`)
//! and `SOLVEMYASTRO_SMAC_GAIA_BRIGHT` (same with `smac_gaia_bright`); `SOLVEMYASTRO_TIER_DIR`
//! sets the tier-stack tests' catalog. The frames are published at
//! <https://artfrom.space/builds/platesolve_bench/platesolve_bench.zip> and
//! the tier catalog at <https://artfrom.space/catalogs/>;
//! `scripts/fetch_bench_data.sh <dest>` downloads and unpacks both.
//!
//! GATE (asserted):
//!   - Every non-skipped frame with a reference-truth entry is solved here.
//!   - M78 (`2024-01-13_20-27-32_L_-10.00_120.00s_0081.fits`) solves within tolerance.
//!   - M78's solution carries SIP (`sip_forward.is_some()`).
//!   - 0 false positives (wrong-position "solve").
//!   - 0 panics.
//!   - Skipped entirely (not solved, not counted, not gated) — see
//!     `SKIP_FRAMES`: `_VSH8902` (24 mm, ~85° FOV, outside the FOV ladder).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use solvemyastro::{solve, SolveConfig, SolveHints, StarCache};

// ---------------------------------------------------------------------------
// Corpus directory + cached reference-truth table
// ---------------------------------------------------------------------------

/// Default locations, relative to `$HOME`.
const BENCH_DIR: &str = "Pictures/Astro/Platesolve Bench";
const SMAC_GAIA: &str =
    "Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia";
/// Optional bright sub-catalog path. If present, the bench uses
/// `Caches::tiered(&deep, &bright)`; if absent, it falls back to
/// `Caches::deep_only(&deep)` (byte-identical to the pre-bright bench).
const SMAC_GAIA_BRIGHT: &str =
    "Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia_bright";

/// Corpus directory: `$SOLVEMYASTRO_BENCH_DIR`, else [`BENCH_DIR`].
fn bench_dir() -> PathBuf {
    env_path_or("SOLVEMYASTRO_BENCH_DIR", BENCH_DIR)
}

/// Deep catalog: `$SOLVEMYASTRO_SMAC_GAIA`, else [`SMAC_GAIA`].
fn smac_gaia_dir() -> PathBuf {
    env_path_or("SOLVEMYASTRO_SMAC_GAIA", SMAC_GAIA)
}

/// Bright sub-catalog: `$SOLVEMYASTRO_SMAC_GAIA_BRIGHT`, else [`SMAC_GAIA_BRIGHT`].
fn smac_gaia_bright_dir() -> PathBuf {
    env_path_or("SOLVEMYASTRO_SMAC_GAIA_BRIGHT", SMAC_GAIA_BRIGHT)
}

fn env_path_or(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(default)
    })
}

/// Key: distinctive filename substring; value: (ra_deg, dec_deg, scale_arcsec_per_px)
fn cached_truth() -> &'static [(&'static str, f64, f64, f64)] {
    &[
        ("2024-01-13_20-27-32_L_-10.00_120.00s_0081", 86.682, 0.042, 0.96),
        ("2024-02-24_19-42-32_H_-20.60_300.00s_0074", 114.184, 65.610, 0.48),
        ("M 82_H_-20.60_300.00s_0254",                148.949, 69.679, 0.48),
        ("M 51_H_-20.60_300.00s_0060",                202.483, 47.194, 0.48),
        ("NGC 2903_H_-19.80_300.00s_0004",            143.034, 21.507, 0.48),
        ("2024-09-08_04-30-13_L_-9.90_180.00s_0008",  85.391, -2.441, 0.39),
        ("2025-08-23_01-38-32_B_-10.00_180.00s_0012", 326.954, 47.242, 2.22),
        ("2025-10-05_00-50-11_R_-0.00_120.00s_0000",   56.655, 24.068, 1.73),
        ("2025-10-05_04-30-45_R_0.00_30.00s_0003",    151.538, 41.938, 1.73),
        ("2025-10-26_01-35-09_L_-10.00_300.00s_0034",  83.780, -5.330, 1.73),
        ("Light_Gamma Scuti_180.0s_Bin1_20251018-201649_0010", 276.506, -13.584, 2.88),
        ("Light_Pane 4_300.0s_Bin1_O_gain111_20211011-024956", 36.702,  60.880, 0.88),
        ("Light_Pane 7_300.0s_Bin1_O_gain111_20211014-045033", 36.113,  61.488, 0.88),
        ("Light_veil_300.0s_Bin1_0015",               312.045, 30.958, 1.89),
        // M31 frames solved blind (no usable hints: _DSC2522's FOCALLEN says
        // 50 mm, the solution is ~1000 mm; _DSC5767 carries no header at
        // all). Confirmed independently: projecting M31, M32 and M110
        // through each solution lands on the galaxies in the image.
        ("_DSC2522",                                   10.763, 41.257, 1.004),
        ("_DSC5767",                                   10.727, 41.706, 2.770),
    ]
}

/// Frames excluded from the corpus gate entirely — not solved, not counted,
/// not gated:
///   - `_VSH8902`: 24 mm on a full-frame body, ~85° FOV — far outside the
///     FOV ladder (14°); no known truth.
const SKIP_FRAMES: &[&str] = &["_VSH8902"];

// ---------------------------------------------------------------------------
// Minimal FITS header reader (no external dep — hand-rolled 2880-byte blocks)
// ---------------------------------------------------------------------------

/// Parse a FITS file and return a flat map of header keyword→value strings.
/// Reads only the primary HDU header.
fn read_fits_header(path: &Path) -> HashMap<String, String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return HashMap::new(),
    };
    let mut map = HashMap::new();
    let mut pos = 0;
    let block = 2880;
    let card = 80;
    let mut done = false;
    while pos + block <= bytes.len() && !done {
        for i in 0..(block / card) {
            let start = pos + i * card;
            let raw = &bytes[start..start + card];
            // Convert to ASCII-safe string.
            let s: String = raw.iter().map(|&b| if b >= 0x20 && b < 0x7f { b as char } else { ' ' }).collect();
            let key = s[..8].trim().to_uppercase();
            if key == "END" {
                done = true;
                break;
            }
            if s.len() > 10 && s.as_bytes()[8] == b'=' {
                let val_comment = s[10..].trim();
                // Strip trailing comment (after /).
                let val = if let Some(slash) = val_comment.find('/') {
                    let before = val_comment[..slash].trim();
                    // If a string value, preserve internal slashes.
                    if before.starts_with('\'') {
                        let end = before[1..].find('\'').map(|p| p + 2).unwrap_or(before.len());
                        before[..end].trim_matches('\'').trim().to_string()
                    } else {
                        before.to_string()
                    }
                } else {
                    val_comment.trim_matches('\'').trim().to_string()
                };
                if !key.is_empty() {
                    map.insert(key, val);
                }
            }
        }
        pos += block;
    }
    map
}

/// Parse an XISF file's embedded FITSKeyword tags and return a keyword→value map.
///
/// XISF 1.0 embeds FITS keywords as XML attributes:
///   `<FITSKeyword name="OBJCTRA" value="'2 12 04.043'" comment="..." />`
///
/// We extract the `name` and `value` attributes with a simple byte-scan
/// (no XML parser needed for this well-defined sub-format).
fn read_xisf_header(path: &Path) -> HashMap<String, String> {
    use std::io::Read;
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return HashMap::new(),
    };

    // XISF signature: "XISF0100" (8 bytes) + little-endian u32 header_length + u32 reserved.
    let mut sig = [0u8; 16];
    if file.read_exact(&mut sig).is_err() {
        return HashMap::new();
    }
    if &sig[..8] != b"XISF0100" {
        return HashMap::new();
    }
    let hdr_len = u32::from_le_bytes([sig[8], sig[9], sig[10], sig[11]]) as usize;
    if hdr_len == 0 || hdr_len > 2 * 1024 * 1024 {
        return HashMap::new(); // sanity cap: 2 MB XML
    }

    let mut xml_bytes = vec![0u8; hdr_len];
    if file.read_exact(&mut xml_bytes).is_err() {
        return HashMap::new();
    }
    let xml = String::from_utf8_lossy(&xml_bytes);

    // Extract all `<FITSKeyword name="KEY" value="VALUE" .../>` occurrences.
    let mut map = HashMap::new();
    let tag = "FITSKeyword ";
    let mut search = xml.as_ref();
    while let Some(idx) = search.find(tag) {
        let rest = &search[idx + tag.len()..];

        // Pull name="..."
        let name_opt = extract_xml_attr(rest, "name");
        // Pull value="..."
        let val_opt = extract_xml_attr(rest, "value");

        if let (Some(name), Some(raw_val)) = (name_opt, val_opt) {
            // FITS string values may be wrapped in single quotes: strip them.
            let val = raw_val.trim().trim_matches('\'').trim().to_string();
            map.insert(name.to_uppercase(), val);
        }

        // Advance past this tag.
        search = &search[idx + tag.len()..];
    }
    map
}

/// Extract the value of an XML attribute: `name="value"` → `value`.
/// Handles both single and double quote delimiters.
fn extract_xml_attr<'a>(s: &'a str, attr: &str) -> Option<&'a str> {
    let needle = format!("{attr}=");
    let idx = s.find(needle.as_str())?;
    let after = &s[idx + needle.len()..];
    let (quote, rest) = if after.starts_with('"') {
        ('"', &after[1..])
    } else if after.starts_with('\'') {
        ('\'', &after[1..])
    } else {
        return None;
    };
    let end = rest.find(quote)?;
    Some(&rest[..end])
}

/// Parse an HMS string like "05 43 00.000" or "05:43:00.0" → decimal degrees.
fn parse_ra_hms(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let parts = if parts.len() < 3 {
        let colon: Vec<&str> = s.split(':').collect();
        if colon.len() >= 3 { colon } else { return None }
    } else {
        parts
    };
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    Some((h + m / 60.0 + sec / 3600.0) * 15.0)
}

/// Parse a DMS string like "+00 02 33.6" or "+00:02:33.6" → decimal degrees.
fn parse_dec_dms(s: &str) -> Option<f64> {
    let s = s.trim();
    let neg = s.starts_with('-');
    let s2 = s.trim_start_matches(['+', '-']);
    let parts: Vec<&str> = s2.split_whitespace().collect();
    let parts = if parts.len() < 3 {
        let colon: Vec<&str> = s2.split(':').collect();
        if colon.len() >= 2 { colon } else {
            // Maybe just degrees.
            let v: f64 = s.parse().ok()?;
            return Some(v);
        }
    } else {
        parts
    };
    let d: f64 = parts[0].parse().ok()?;
    let m: f64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let sec: f64 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let v = d + m / 60.0 + sec / 3600.0;
    Some(if neg { -v } else { v })
}

/// Parse the observation epoch (Julian year) from DATE-OBS for proper-motion
/// propagation. Month/day precision is ample for PM (mas/yr over years).
fn parse_obs_epoch(hdr: &HashMap<String, String>) -> Option<f64> {
    let date = hdr.get("DATE-OBS").or_else(|| hdr.get("DATE_OBS"))?;
    let date_part = date.split('T').next()?; // "YYYY-MM-DD..."
    let mut it = date_part.split('-');
    let y: f64 = it.next()?.trim().parse().ok()?;
    let mo: f64 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(1.0);
    let d: f64 = it.next().and_then(|s| s.trim().parse().ok()).unwrap_or(1.0);
    if !(1900.0..3000.0).contains(&y) {
        return None;
    }
    Some(y + ((mo - 1.0) * 30.4 + (d - 1.0)) / 365.25)
}

/// Extract solve hints from a FITS header map.
/// Follows the same precedence as athenaeum's hints.rs.
fn hints_from_fits(hdr: &HashMap<String, String>, long_px: u32) -> SolveHints {
    // Pixel scale check: images with an existing WCS (plate-solved stacks) have
    // unreliable FOCALLEN/CDELT1 for the drizzled pixel scale. Detect this early.
    let has_existing_wcs = hdr.contains_key("CRVAL1") && hdr.contains_key("CRVAL2")
        && hdr.contains_key("CRPIX1");

    // RA: for images with existing WCS, prefer CRVAL1 (accurate plate-solve center)
    // over OBJCTRA (telescope pointing at capture time, may be off-center or stale).
    // For fresh sub-frames without WCS, OBJCTRA is the best available hint.
    let ra = if has_existing_wcs {
        hdr.get("CRVAL1").and_then(|s| {
            let v: f64 = s.parse().ok()?;
            if v.is_finite() && v >= 0.0 && v < 360.0 { Some(v) } else { None }
        })
        .or_else(|| hdr.get("OBJCTRA").and_then(|s| parse_ra_hms(s)))
    } else {
        hdr.get("OBJCTRA")
            .and_then(|s| parse_ra_hms(s))
            .or_else(|| hdr.get("RA").and_then(|s| {
                let v: f64 = s.parse().ok()?;
                if v >= 0.0 && v < 24.0 { Some(v * 15.0) } else { Some(v) }
            }))
    };

    // Dec: same logic — CRVAL2 for stacked images, OBJCTDEC for fresh frames.
    let dec = if has_existing_wcs {
        hdr.get("CRVAL2").and_then(|s| {
            let v: f64 = s.parse().ok()?;
            if v.is_finite() && v.abs() <= 90.0 { Some(v) } else { None }
        })
        .or_else(|| hdr.get("OBJCTDEC").and_then(|s| parse_dec_dms(s)))
    } else {
        hdr.get("OBJCTDEC")
            .and_then(|s| parse_dec_dms(s))
            .or_else(|| hdr.get("DEC").and_then(|s| {
                parse_dec_dms(s).or_else(|| s.parse::<f64>().ok())
            }))
    };

    // Pixel scale: use FOCALLEN + XPIXSZ from the native camera.
    // Exception: images with an existing plate-solved WCS (CRVAL1 present) may have
    // been drizzled/resampled in a processing program, where the FOCALLEN/CDELT1 metadata reflects
    // the original sub-frame camera scale rather than the actual drizzled pixel scale.
    // For those, suppress the scale hint and let the solver do a full blind scale search
    // (it already has an accurate position hint from CRVAL1/CRVAL2, so only the scale
    // rungs need to be explored, which is fast with the hinted search centre).
    let pixel_scale_arcsec = if has_existing_wcs {
        // Already plate-solved: FOCALLEN/CDELT1 may not match actual drizzled pixel scale.
        // Blank scale hint → solver tries all 9 blind rungs with the CRVAL position anchor.
        None
    } else {
        let focallen = hdr.get("FOCALLEN").and_then(|s| s.parse::<f64>().ok());
        let xpixsz = hdr.get("XPIXSZ").and_then(|s| s.parse::<f64>().ok());
        let binning = hdr.get("XBINNING").and_then(|s| s.parse::<f64>().ok()).unwrap_or(1.0);
        match (focallen, xpixsz) {
            (Some(fl), Some(sz)) if fl > 0.0 && sz > 0.0 => {
                let px_mm = sz / 1000.0 * binning;
                let scale_tan = px_mm / fl;
                let s = scale_tan.atan().to_degrees() * 3600.0;
                if s > 0.01 && s < 50.0 { Some(s) } else { None }
            }
            _ => None,
        }
    };

    // FOV from scale + image size (for the FOV-ladder hint).
    let fov_deg = pixel_scale_arcsec.map(|s| s * long_px as f64 / 3600.0);

    SolveHints {
        ra,
        dec,
        fov_deg,
        pixel_scale_arcsec,
        search_radius_deg: Some(15.0), // generous radius around the prior
        epoch: parse_obs_epoch(hdr),
    }
}

// ---------------------------------------------------------------------------
// Great-circle separation
// ---------------------------------------------------------------------------

fn sep_arcsec(r1: f64, d1: f64, r2: f64, d2: f64) -> f64 {
    let (a, b) = (d1.to_radians(), d2.to_radians());
    let v = a.sin() * b.sin() + a.cos() * b.cos() * (r1 - r2).to_radians().cos();
    v.clamp(-1.0, 1.0).acos().to_degrees() * 3600.0
}

// ---------------------------------------------------------------------------
// Corpus bench test
// ---------------------------------------------------------------------------

#[test]
#[ignore = "needs smac_gaia cache + corpus; run with --release --nocapture"]
fn corpus_bench() {
    let bench_dir = bench_dir();
    let bench_dir = bench_dir.as_path();
    let smac_path = smac_gaia_dir();
    let smac_path = smac_path.as_path();

    if !bench_dir.is_dir() {
        println!("SKIP corpus_bench: bench dir not found at {}", bench_dir.display());
        return;
    }
    if !smac_path.is_dir() {
        println!("SKIP corpus_bench: smac_gaia cache not found at {}", smac_path.display());
        return;
    }

    let cache = StarCache::open(smac_path).expect("open smac_gaia cache");

    // Optional bright sub-catalog. Present → use tiered caches (bright
    // for quad matching, deep for verify, auto-fallback). Absent → byte-
    // identical to the pre-bright path. Either way the gate is the same.
    let bright_path = smac_gaia_bright_dir();
    let bright_path = bright_path.as_path();
    let bright_cache: Option<StarCache> = if bright_path.is_dir() {
        match StarCache::open(bright_path) {
            Ok(c) => {
                println!("corpus_bench: using BRIGHT sub-catalog at {} ({} stars)",
                    bright_path.display(), c.star_count());
                Some(c)
            }
            Err(e) => {
                println!("corpus_bench: failed to open bright cache ({e}), using deep-only");
                None
            }
        }
    } else {
        println!("corpus_bench: no bright sub-catalog at {} — using deep-only", bright_path.display());
        None
    };

    // SOLVEMYASTRO_PRECISE=1 — opt-in "precise" profile for Phase-1 measurement.
    // Enables PSF-refined centroids (Moffat LM via rustafits) in the detection
    // stage. Results are printed per-frame (RMS, FWHM) but NOT gated — the hard
    // gate always runs on the default fast profile. Run as:
    //   SOLVEMYASTRO_PRECISE=1 cargo test -p solvemyastro --release \
    //     --test corpus_bench -- --ignored --nocapture
    let precise_mode = std::env::var("SOLVEMYASTRO_PRECISE").ok().as_deref() == Some("1");
    if precise_mode {
        println!("\n[corpus_bench] Running in PRECISE mode (PSF centroid refinement ON)");
    }

    let cfg = SolveConfig {
        // Phase 1 instrumentation: print per-stage timing for each frame so we
        // can validate where wall-clock actually goes before committing to the
        // Phase 3 speed work. Strip this back to default() once the data is
        // captured.
        print_timing: true,
        centroid_refinement: if precise_mode {
            solvemyastro::CentroidRefinement::Auto
        } else {
            solvemyastro::CentroidRefinement::Off
        },
        ..SolveConfig::default()
    };

    // Enumerate corpus files (FITS/FIT/XISF), sorted.
    let mut files: Vec<PathBuf> = std::fs::read_dir(bench_dir)
        .expect("read bench dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
                Some("fits" | "fit" | "xisf")
            )
        })
        .collect();
    files.sort();

    // Table header
    println!(
        "\n{:52} {:5} {:5} {:>8} {:>8} {:>7} {:>7}  {}",
        "File", "Truth", "SMac", "ΔPos\"", "ΔScl%", "InlRt", "ms", "Notes"
    );
    println!("{}", "-".repeat(130));

    let truth = cached_truth();
    let mut n_truth_solved = 0usize;
    let mut n_solvemyastro_solved = 0usize;
    let mut n_correct = 0usize;
    let mut n_wrong = 0usize; // false positives

    // Per-frame results for gate assertions.
    #[allow(dead_code)]
    struct FrameResult {
        name: String,
        truth_solved: bool,
        truth_ra: f64,
        truth_dec: f64,
        truth_scale: f64,
        smc_solved: bool,
        smc_ra: f64,
        smc_dec: f64,
        smc_scale: f64,
        dpos_arcsec: f64,
        dscale_pct: f64,
        correct: bool,
        has_sip: bool,
        notes: String,
        // Phase-0 instrumentation: precision + speed metrics for the
        // machine-readable baseline (corpus_run.json / corpus_baseline.json).
        rms_px: f64,        // sol.rms_residual_px; -1.0 if unsolved
        inlier_ratio: f64,  // -1.0 if unsolved
        matched_stars: usize,
        total_detected: usize,
        sip_order: i32,     // 0 if no SIP
        elapsed_ms: u64,
        // Phase-1 PSF refinement metrics (precise-mode only).
        psf_fwhm_px: Option<f64>,
    }

    let mut results: Vec<FrameResult> = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        // Skip user-excluded frames entirely (not solved, not counted, not gated).
        if let Some(tag) = SKIP_FRAMES.iter().find(|s| name.contains(**s)) {
            println!("{:52} SKIPPED ({tag})", &name[..name.len().min(52)]);
            continue;
        }

        // Lookup reference truth.
        let truth_entry = truth.iter().find(|&&(key, ..)| name.contains(key));
        let truth_solved = truth_entry.is_some();
        let (truth_ra, truth_dec, truth_scale) = truth_entry
            .map(|&(_, ra, dec, sc)| (ra, dec, sc))
            .unwrap_or((0.0, 0.0, 0.0));

        if truth_solved {
            n_truth_solved += 1;
        }

        // Read header for hints.
        let hdr = if name.ends_with(".fits") || name.ends_with(".fit") {
            read_fits_header(path)
        } else if name.ends_with(".xisf") {
            read_xisf_header(path)
        } else {
            HashMap::new()
        };

        // Determine image long-axis from NAXIS1/NAXIS2.
        let naxis1 = hdr.get("NAXIS1").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        let naxis2 = hdr.get("NAXIS2").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        let long_px = naxis1.max(naxis2);

        let hints = hints_from_fits(&hdr, long_px);

        let t0 = Instant::now();

        // Run solver with panic protection.
        let solve_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let caches = match &bright_cache {
                Some(b) => solvemyastro::Caches::tiered(&cache, b),
                None    => solvemyastro::Caches::deep_only(&cache),
            };
            solve(path, &hints, &caches, &cfg, None)
        }));

        let elapsed_ms = t0.elapsed().as_millis() as u64;

        // Phase-0 metric capture (set in the solved arm; defaults = unsolved).
        let mut m_rms = -1.0_f64;
        let mut m_inlier = -1.0_f64;
        let mut m_matched = 0usize;
        let mut m_total = 0usize;
        let mut m_sip_order = 0i32;

        let mut m_psf_fwhm: Option<f64> = None;

        let (smc_solved, smc_ra, smc_dec, smc_scale, has_sip, notes) = match solve_result {
            Ok(Ok(sol)) => {
                let ra = sol.wcs.crval.0;
                let dec = sol.wcs.crval.1;
                let scale = sol.pixel_scale_arcsec;
                let sip = sol.wcs.sip_forward.is_some();
                m_rms = sol.rms_residual_px;
                m_inlier = sol.inlier_ratio;
                m_matched = sol.matched_stars;
                m_total = sol.total_detected;
                m_sip_order = sol
                    .wcs
                    .sip_forward
                    .as_ref()
                    .map(|(a, _)| a.order as i32)
                    .unwrap_or(0);
                m_psf_fwhm = sol.psf_fwhm_px;
                (true, ra, dec, scale, sip, String::new())
            }
            Ok(Err(e)) => {
                let msg = e.to_string().lines().next().unwrap_or("").chars().take(50).collect();
                (false, 0.0, 0.0, 0.0, false, msg)
            }
            Err(p) => {
                let msg = p.downcast_ref::<&str>().map(|s| s.to_string())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "PANIC".to_string());
                (false, 0.0, 0.0, 0.0, false, format!("PANIC: {}", &msg.chars().take(45).collect::<String>()))
            }
        };

        let (dpos, dscale) = if truth_solved && smc_solved {
            let dp = sep_arcsec(truth_ra, truth_dec, smc_ra, smc_dec);
            let ds = if truth_scale > 0.0 {
                (smc_scale - truth_scale).abs() / truth_scale * 100.0
            } else {
                -1.0
            };
            (dp, ds)
        } else {
            (-1.0, -1.0)
        };

        let correct = truth_solved && smc_solved && dpos >= 0.0 && dpos < 30.0 && dscale < 2.0;
        let wrong = smc_solved && (!truth_solved || !correct);

        if smc_solved { n_solvemyastro_solved += 1; }
        if correct { n_correct += 1; }
        if wrong && truth_solved && !correct { n_wrong += 1; }

        let truth_tag = if truth_solved { "OK   " } else { "FAIL " };
        let smc_tag = if smc_solved { "OK   " } else { "FAIL " };
        let dpos_str = if dpos >= 0.0 { format!("{:8.1}", dpos) } else { "      --".to_string() };
        let dscale_str = if dscale >= 0.0 { format!("{:8.2}", dscale) } else { "      --".to_string() };
        let inlier_str = "--   ";
        let sip_note = if has_sip { " SIP" } else { "" };
        let flag = if truth_solved && !correct && smc_solved { "  <-- WRONG" }
                   else if truth_solved && !smc_solved { "  <-- MISSED" }
                   else { "" };
        // Precise-mode suffix: show per-frame RMS and PSF FWHM when available.
        let precise_suffix = if precise_mode {
            let rms_s = if m_rms >= 0.0 { format!(" rms={:.3}px", m_rms) } else { String::new() };
            let fwhm_s = match m_psf_fwhm {
                Some(f) => format!(" fwhm={:.2}px", f),
                None    => " fwhm=--".to_string(),
            };
            format!("{rms_s}{fwhm_s}")
        } else {
            String::new()
        };

        println!(
            "{:52} {} {} {} {} {} {:>7}  {}{}{}{}",
            &name[..name.len().min(52)],
            truth_tag, smc_tag,
            dpos_str, dscale_str,
            inlier_str,
            elapsed_ms,
            notes,
            sip_note,
            flag,
            precise_suffix
        );

        results.push(FrameResult {
            name: name.clone(),
            truth_solved,
            truth_ra,
            truth_dec,
            truth_scale,
            smc_solved,
            smc_ra,
            smc_dec,
            smc_scale,
            dpos_arcsec: dpos,
            dscale_pct: dscale,
            correct,
            has_sip,
            notes: format!("{notes}{sip_note}"),
            rms_px: m_rms,
            inlier_ratio: m_inlier,
            matched_stars: m_matched,
            total_detected: m_total,
            sip_order: m_sip_order,
            elapsed_ms,
            psf_fwhm_px: m_psf_fwhm,
        });
    }

    // -----------------------------------------------------------------------
    // Phase-0 instrumentation: machine-readable per-frame run + regression gate
    // -----------------------------------------------------------------------
    // Always write tests/corpus_run.json (precision + speed, diffable). If a
    // schema-2 tests/corpus_baseline.json exists, gate per-frame rms_px and
    // elapsed_ms against it (no-worse). Freeze a new baseline after a clean run
    // with: cp tests/corpus_run.json tests/corpus_baseline.json
    let f64_or_null = |v: f64| -> serde_json::Value {
        if v >= 0.0 { serde_json::Value::from(v) } else { serde_json::Value::Null }
    };
    let run_rows: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "frame": r.name,
                "solved": r.smc_solved,
                "dpos_arcsec": f64_or_null(r.dpos_arcsec),
                "dscale_pct": f64_or_null(r.dscale_pct),
                "rms_px": f64_or_null(r.rms_px),
                "inlier_ratio": f64_or_null(r.inlier_ratio),
                "matched_stars": r.matched_stars,
                "total_detected": r.total_detected,
                "sip_order": r.sip_order,
                "has_sip": r.has_sip,
                "elapsed_ms": r.elapsed_ms,
            })
        })
        .collect();
    let run_doc = serde_json::json!({ "schema_version": 2, "frames": run_rows });
    let run_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus_run.json");
    std::fs::write(&run_path, serde_json::to_string_pretty(&run_doc).unwrap())
        .expect("write corpus_run.json");
    println!("\nPhase-0: wrote machine-readable run to {}", run_path.display());

    // Summary
    println!("{}", "-".repeat(130));
    let total = results.len();
    println!(
        "SUMMARY: {} frames (skipped frames excluded) | Truth={} | solvemyastro={} | correct={} | wrong={}",
        total, n_truth_solved, n_solvemyastro_solved, n_correct, n_wrong
    );

    // M78 result
    let m78 = results.iter().find(|r| r.name.contains("2024-01-13_20-27-32_L_-10.00_120.00s_0081"));
    match m78 {
        Some(r) => {
            println!(
                "\nM78: solved={} dpos={:.1}\" dscale={:.2}% sip={}",
                r.smc_solved,
                if r.dpos_arcsec >= 0.0 { r.dpos_arcsec } else { -1.0 },
                if r.dscale_pct >= 0.0 { r.dscale_pct } else { -1.0 },
                r.has_sip
            );
        }
        None => println!("\nM78: not found in results"),
    }

    // Identify regressions (every truth frame solvemyastro did not solve).
    let missed: Vec<&FrameResult> = results.iter()
        .filter(|r| r.truth_solved && !r.smc_solved)
        .collect();
    let wrong_pos: Vec<&FrameResult> = results.iter()
        .filter(|r| r.smc_solved && r.truth_solved && !r.correct)
        .collect();

    if !missed.is_empty() {
        println!("\nMISSED (truth available, solvemyastro did not solve):");
        for r in &missed {
            println!("  {}", r.name);
        }
    }
    if !wrong_pos.is_empty() {
        println!("\nWRONG POSITION (false positives):");
        for r in &wrong_pos {
            println!("  {} dpos={:.1}\" dscale={:.2}%", r.name, r.dpos_arcsec, r.dscale_pct);
        }
    }

    // -----------------------------------------------------------------------
    // GATE ASSERTIONS
    // -----------------------------------------------------------------------

    // Gate 1: M78 must solve.
    let m78 = results.iter().find(|r| r.name.contains("2024-01-13_20-27-32_L_-10.00_120.00s_0081"))
        .expect("M78 must be in the corpus");
    assert!(
        m78.smc_solved,
        "GATE FAIL: M78 did not solve (the primary gate frame)"
    );
    assert!(
        m78.dpos_arcsec < 30.0,
        "GATE FAIL: M78 solved but position error {:.1}\" exceeds 30\"",
        m78.dpos_arcsec
    );
    assert!(
        m78.dscale_pct < 2.0,
        "GATE FAIL: M78 solved but scale error {:.2}% exceeds 2%",
        m78.dscale_pct
    );
    assert!(
        m78.has_sip,
        "GATE FAIL: M78 solved but SIP is not present (sip_forward must be Some)"
    );

    // Gate 2: No false positives.
    assert!(
        wrong_pos.is_empty(),
        "GATE FAIL: {} false positive(s) detected: {:?}",
        wrong_pos.len(),
        wrong_pos.iter().map(|r| &r.name).collect::<Vec<_>>()
    );

    // Gate 3: Every non-skipped frame with a truth entry must be solved here.
    let mandatory_missed: Vec<&FrameResult> = results.iter()
        .filter(|r| r.truth_solved && !r.smc_solved)
        .collect();
    assert!(
        mandatory_missed.is_empty(),
        "GATE FAIL: {} frame(s) with reference truth were not solved by solvemyastro: {:?}",
        mandatory_missed.len(),
        mandatory_missed.iter().map(|r| &r.name).collect::<Vec<_>>()
    );

    // Gate 4: No panics (if we got here, none occurred).
    let panics: Vec<&FrameResult> = results.iter()
        .filter(|r| r.notes.starts_with("PANIC"))
        .collect();
    assert!(
        panics.is_empty(),
        "GATE FAIL: {} panic(s): {:?}",
        panics.len(),
        panics.iter().map(|r| &r.name).collect::<Vec<_>>()
    );

    // Regression gate vs a frozen schema-2 baseline (skipped if absent/old).
    let baseline_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/corpus_baseline.json");
    match std::fs::read_to_string(&baseline_path)
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    {
        Some(base) if base.get("schema_version").and_then(|v| v.as_u64()) == Some(2) => {
            let base_frames: Vec<serde_json::Value> =
                base["frames"].as_array().cloned().unwrap_or_default();
            // Precision is gated per-frame and strictly (rms is deterministic).
            // Speed is gated on the NET wall-clock total, not per-frame: per-frame
            // wall-clock varies ~2x under machine load, so a strict per-frame gate
            // trips on noise. The deterministic per-frame search-breadth signal is
            // `cone_calls` in the `[solvemyastro timing]` stderr lines.
            let mut rms_regressions: Vec<String> = Vec::new();
            let mut base_ms_total = 0u64;
            let mut new_ms_total = 0u64;

            // Collect baseline rms_px values for median computation.
            let mut baseline_rms_vals: Vec<f64> = Vec::new();

            for r in &results {
                let Some(b) = base_frames
                    .iter()
                    .find(|b| b["frame"].as_str() == Some(r.name.as_str()))
                else {
                    continue;
                };
                if r.rms_px >= 0.0 {
                    if let Some(brms) = b["rms_px"].as_f64() {
                        if r.rms_px > brms * 1.15 + 0.05 {
                            rms_regressions.push(format!(
                                "{}: rms {:.3}px > 1.15x baseline {:.3}px",
                                r.name, r.rms_px, brms
                            ));
                        }
                        baseline_rms_vals.push(brms);
                    }
                }
                if let Some(bms) = b["elapsed_ms"].as_u64() {
                    base_ms_total += bms;
                    new_ms_total += r.elapsed_ms;
                }
            }
            let speed_ratio = new_ms_total as f64 / base_ms_total.max(1) as f64;
            println!(
                "\nGATE vs baseline — precision: {} rms regression(s); speed (net wall): \
                 {}ms vs {}ms ({:.2}x)",
                rms_regressions.len(),
                new_ms_total,
                base_ms_total,
                speed_ratio
            );
            for x in &rms_regressions {
                println!("  rms-regression: {x}");
            }
            assert!(
                rms_regressions.is_empty(),
                "GATE FAIL: {} precision regression(s) vs baseline",
                rms_regressions.len()
            );
            assert!(
                speed_ratio <= 1.30,
                "GATE FAIL: net wall {:.2}x baseline (>1.30x) — investigate (re-run on a quiet \
                 machine; wall-clock is noise-sensitive, cone_calls is the deterministic signal)",
                speed_ratio
            );

            // Phase-4 gate: median per-frame rms_px ≤ 1.05× baseline median.
            // This ensures the precise profile never degrades the *typical* frame's
            // accuracy relative to the fast baseline, even if individual frames shift
            // within the per-frame ±15% tolerance. Applied to the fast (default) run
            // which matches the baseline profile byte-for-byte.
            if !baseline_rms_vals.is_empty() {
                baseline_rms_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let baseline_median = if baseline_rms_vals.len() % 2 == 0 {
                    (baseline_rms_vals[baseline_rms_vals.len() / 2 - 1]
                        + baseline_rms_vals[baseline_rms_vals.len() / 2])
                        * 0.5
                } else {
                    baseline_rms_vals[baseline_rms_vals.len() / 2]
                };

                let mut run_rms_vals: Vec<f64> = results
                    .iter()
                    .filter_map(|r| if r.rms_px >= 0.0 { Some(r.rms_px) } else { None })
                    .collect();
                run_rms_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let run_median = if !run_rms_vals.is_empty() {
                    if run_rms_vals.len() % 2 == 0 {
                        (run_rms_vals[run_rms_vals.len() / 2 - 1]
                            + run_rms_vals[run_rms_vals.len() / 2])
                            * 0.5
                    } else {
                        run_rms_vals[run_rms_vals.len() / 2]
                    }
                } else {
                    0.0
                };

                println!(
                    "GATE median rms_px: run={:.3}px baseline={:.3}px ratio={:.3}x",
                    run_median,
                    baseline_median,
                    if baseline_median > 0.0 { run_median / baseline_median } else { 1.0 }
                );
                assert!(
                    run_median <= baseline_median * 1.05,
                    "GATE FAIL: median rms {:.3}px > 1.05x baseline median {:.3}px (ratio={:.3}x)",
                    run_median,
                    baseline_median,
                    run_median / baseline_median.max(1e-9)
                );
            }
        }
        _ => {
            println!(
                "\nNote: no schema-2 corpus_baseline.json — freeze this run with:\n  \
                 cp solvemyastro/tests/corpus_run.json solvemyastro/tests/corpus_baseline.json"
            );
        }
    }

    println!("\nGATE: ALL assertions PASSED");

    // -----------------------------------------------------------------------
    // Phase-1 precise-mode summary (printed after the gate so it doesn't
    // clutter normal runs). Shows per-frame RMS for the two key targets.
    // -----------------------------------------------------------------------
    if precise_mode {
        println!("\n=== PRECISE-MODE SUMMARY (PSF centroid refinement) ===");
        println!("{:52} {:>10} {:>10}", "Frame", "RMS (px)", "FWHM (px)");
        println!("{}", "-".repeat(75));
        for r in &results {
            if r.rms_px >= 0.0 {
                let fwhm_s = match r.psf_fwhm_px {
                    Some(f) => format!("{:10.3}", f),
                    None    => "        --".to_string(),
                };
                println!("{:52} {:10.3} {}", &r.name[..r.name.len().min(52)], r.rms_px, fwhm_s);
            }
        }

        // Focus targets.
        let frame_0008 = results.iter().find(|r| r.name.contains("_0008"));
        let frame_m78  = results.iter().find(|r| r.name.contains("2024-01-13_20-27-32_L_-10.00_120.00s_0081"));

        println!("\n--- Phase-1 targets ---");
        if let Some(r) = frame_0008 {
            println!("  _0008: rms={:.3}px fwhm={}",
                r.rms_px,
                r.psf_fwhm_px.map(|f| format!("{:.2}px", f)).unwrap_or("--".to_string()));
        } else {
            println!("  _0008: not in results");
        }
        if let Some(r) = frame_m78 {
            println!("  M78 (_0081): rms={:.3}px fwhm={}",
                r.rms_px,
                r.psf_fwhm_px.map(|f| format!("{:.2}px", f)).unwrap_or("--".to_string()));
        } else {
            println!("  M78: not in results");
        }
        println!("=======================================================");
    }
}

/// Real-data Plan 2 guard: the new `Caches::Layered` path, with a SINGLE layer,
/// must solve a frame byte-identically to the legacy `Caches::deep_only` path on
/// the same cache. Uses the brightest tier (`tier_500`) as a usable standalone
/// catalog and an easy wide-field frame.
///
/// Run with:
///   SOLVEMYASTRO_TIER_DIR=/path/to/catalog_out \
///     cargo test -p solvemyastro --release --test corpus_bench \
///     -- --ignored layered_single_layer_solve_equals_legacy --nocapture
#[test]
#[ignore = "needs a tier cache (SOLVEMYASTRO_TIER_DIR=<dir with tier_*>) + corpus frame"]
fn layered_single_layer_solve_equals_legacy() {
    let tier_dir = match std::env::var("SOLVEMYASTRO_TIER_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            println!("SKIP: set SOLVEMYASTRO_TIER_DIR to a dir containing tier_*");
            return;
        }
    };
    let bench_dir = bench_dir();
    let bench_dir = bench_dir.as_path();
    if !bench_dir.is_dir() {
        println!("SKIP: bench dir absent at {}", bench_dir.display());
        return;
    }
    let layer_dir = tier_dir.join("tier_500");
    let cache = match StarCache::open(&layer_dir) {
        Ok(c) => c,
        Err(e) => {
            println!("SKIP: open {} failed: {e}", layer_dir.display());
            return;
        }
    };
    let frame = bench_dir.join("Light_veil_300.0s_Bin1_0015.fit");
    if !frame.is_file() {
        println!("SKIP: veil frame absent at {}", frame.display());
        return;
    }

    let hdr = read_fits_header(&frame);
    let naxis1 = hdr.get("NAXIS1").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
    let naxis2 = hdr.get("NAXIS2").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
    let hints = hints_from_fits(&hdr, naxis1.max(naxis2));
    let cfg = SolveConfig { print_timing: false, ..SolveConfig::default() };

    let legacy = solve(&frame, &hints, &solvemyastro::Caches::deep_only(&cache), &cfg, None)
        .expect("legacy deep_only solve");
    let layers = [&cache];
    let layered = solve(&frame, &hints, &solvemyastro::Caches::layered(&layers), &cfg, None)
        .expect("layered 1-layer solve");

    assert!(
        (legacy.wcs.crval.0 - layered.wcs.crval.0).abs() < 1e-9,
        "crval RA differs: {} vs {}",
        legacy.wcs.crval.0,
        layered.wcs.crval.0
    );
    assert!(
        (legacy.wcs.crval.1 - layered.wcs.crval.1).abs() < 1e-9,
        "crval Dec differs: {} vs {}",
        legacy.wcs.crval.1,
        layered.wcs.crval.1
    );
    assert_eq!(
        legacy.matched_stars, layered.matched_stars,
        "matched_stars differ"
    );
    assert!(
        (legacy.rms_residual_px - layered.rms_residual_px).abs() < 1e-9,
        "rms differs: {} vs {}",
        legacy.rms_residual_px,
        layered.rms_residual_px
    );
    assert_eq!(
        legacy.wcs.sip_forward.is_some(),
        layered.wcs.sip_forward.is_some(),
        "SIP presence differs"
    );
    println!(
        "OK: 1-layer Layered solve == deep_only solve (crval=({:.6},{:.6}), matched={}, rms={:.4}px)",
        layered.wcs.crval.0, layered.wcs.crval.1, layered.matched_stars, layered.rms_residual_px
    );
}

// ---------------------------------------------------------------------------
// Recorded WCS of every bench frame, held to a frozen baseline
// ---------------------------------------------------------------------------

/// The truth frames the blind index pass solves through the base pack
/// (name substrings, picked as the ones that solve blind through the base pack
/// at 0.3.0; scale in arcsec/px). Two at <= 0.5, two at 1-2, two at 2.2-2.9;
/// none is above 3 in the corpus. (The 0.39 frame and the 0.88 pair do not
/// solve blind through the base pack.)
const INDEX_PASS_FRAMES: &[&str] = &[
    "2024-02-24_19-42-32_H", // 0.48
    "M 51_H",                // 0.48
    "_DSC2522",              // 1.004
    "2025-10-26_01-35-09",   // 1.73
    "2025-08-23_01-38-32",   // 2.22
    "Gamma Scuti",           // 2.88
];

/// The WCS gate's run file: every solved frame's WCS, written by
/// `corpus_layered_tiers` (not committed; regenerated on each run).
const LAYERED_RUN_FILE: &str = "tests/corpus_layered_run.json";
/// The frozen baseline the run is compared against (frozen at 0.3.0).
const LAYERED_BASELINE_FILE: &str = "tests/corpus_layered_baseline.json";

fn sip_table_json(c: &solvemyastro::SipCoefficients) -> serde_json::Value {
    serde_json::json!(c.coeffs)
}

/// One frame's record: whether it solved and, when it did, the WCS (`crval`,
/// `crpix`, `cd`, SIP) with the run's own numbers (`matched_stars`,
/// `inlier_ratio`, `log10_odds`, `reference_stars`, `hypotheses`, `rms_px`,
/// `elapsed_ms`). Only the WCS keys are compared.
fn frame_json(name: &str, res: &anyhow::Result<solvemyastro::SolveSolution>) -> serde_json::Value {
    use serde_json::{json, Value};
    let Ok(s) = res else {
        return json!({ "frame": name, "solved": false });
    };
    let w = &s.wcs;
    let sip = match &w.sip_forward {
        None => Value::Null,
        Some((a, b)) => {
            let (ap, bp) = match &w.sip_reverse {
                Some((ap, bp)) => (Some(ap), Some(bp)),
                None => (None, None),
            };
            json!({
                "a_order": a.order,
                "a": sip_table_json(a),
                "b_order": b.order,
                "b": sip_table_json(b),
                "ap_order": ap.map(|c| c.order),
                "ap": ap.map(sip_table_json),
                "bp_order": bp.map(|c| c.order),
                "bp": bp.map(sip_table_json),
            })
        }
    };
    // The decision's numbers (0.4.0): every solved frame cleared the bar, its
    // ratio is the one assignment's count over the reference list, and at least
    // one hypothesis was verified.
    assert!(
        s.log10_odds >= solvemyastro::verify::ACCEPT_LOG10_ODDS,
        "{name}: solved with log10_odds {} below the bar",
        s.log10_odds
    );
    assert!(s.reference_stars_in_frame > 0, "{name}: solved with no reference star in the frame");
    let ratio = s.matched_stars as f64 / s.reference_stars_in_frame as f64;
    assert!(
        (s.inlier_ratio - ratio).abs() < 1e-12,
        "{name}: inlier_ratio {} != matched {} / reference_stars {}",
        s.inlier_ratio,
        s.matched_stars,
        s.reference_stars_in_frame
    );
    assert!(s.hypotheses_verified >= 1, "{name}: solved with no hypothesis counted");
    json!({
        "frame": name,
        "solved": true,
        "crval": [w.crval.0, w.crval.1],
        "crpix": [w.crpix.0, w.crpix.1],
        "cd": [w.cd[0][0], w.cd[0][1], w.cd[1][0], w.cd[1][1]],
        "sip": sip,
        "matched_stars": s.matched_stars,
        "inlier_ratio": s.inlier_ratio,
        "log10_odds": s.log10_odds,
        "reference_stars": s.reference_stars_in_frame,
        "hypotheses": s.hypotheses_verified,
        "rms_px": s.rms_residual_px,
        "elapsed_ms": s.solve_time_ms,
    })
}

/// The own-set keys of an accepted candidate, from `solve_traced`'s last
/// trace: `path`, `recentered`, `hypotheses`, `own` (image stars that chose
/// the seed), `image_pts`, `linear_inliers` (the seed pre-filter's inliers),
/// `inliers_outside_own` (of those, the ones not in the own set) and
/// `final_inliers`, `reference_mag_limit` (the cone level the reference list came from). Recorded, not compared with the baseline (`wcs_moved`
/// reads only the WCS keys).
fn own_set_json(traces: &[solvemyastro::diag::VerifyTrace]) -> serde_json::Value {
    let t = traces.last().expect("a solved frame has a verify trace");
    assert!(t.accepted, "the last verify trace of a solved frame is the accepted candidate's");
    serde_json::json!({
        "path": t.path.as_str(),
        "recentered": t.recentered,
        "hypotheses": t.hypotheses_so_far,
        "own": t.own_star_count,
        "image_pts": t.image_star_count,
        "linear_inliers": t.seed_inliers,
        "inliers_outside_own": t.inliers_outside_own,
        "final_inliers": t.final_inliers,
        "reference_mag_limit": t.reference_mag_limit,
    })
}

/// `frame_json` plus `own_set` for a solved frame.
fn traced_frame_json(
    name: &str,
    res: &anyhow::Result<solvemyastro::SolveSolution>,
    traces: &[solvemyastro::diag::VerifyTrace],
) -> serde_json::Value {
    let mut rec = frame_json(name, res);
    if res.is_ok() {
        rec["own_set"] = own_set_json(traces);
    }
    rec
}

/// The first key path at which two values differ, or `None`. Numbers are equal
/// when `|x - y| <= 1e-9 * max(|x|, |y|)` (exactly equal when both are 0);
/// arrays must have equal length.
fn json_moved(path: &str, run: &serde_json::Value, base: &serde_json::Value) -> Option<String> {
    use serde_json::Value;
    match (run, base) {
        (Value::Number(a), Value::Number(b)) => {
            let (x, y) = (a.as_f64().unwrap_or(f64::NAN), b.as_f64().unwrap_or(f64::NAN));
            let same = x == y || (x - y).abs() <= 1e-9 * x.abs().max(y.abs());
            (!same).then(|| path.to_string())
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(path.to_string());
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(i, (x, y))| json_moved(&format!("{path}[{i}]"), x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            let mut keys: Vec<&String> = a.keys().chain(b.keys()).collect();
            keys.sort();
            keys.dedup();
            keys.into_iter().find_map(|k| {
                json_moved(
                    &format!("{path}.{k}"),
                    a.get(k).unwrap_or(&Value::Null),
                    b.get(k).unwrap_or(&Value::Null),
                )
            })
        }
        (x, y) => (x != y).then(|| path.to_string()),
    }
}

/// The first moved WCS key of a frame against its baseline record (`crval`,
/// `crpix`, `cd`, `sip` only), as a path such as `"sip.a[2][1]"`.
fn wcs_moved(run: &serde_json::Value, base: &serde_json::Value) -> Option<String> {
    ["crval", "crpix", "cd", "sip"].iter().find_map(|k| {
        json_moved(
            k,
            run.get(*k).unwrap_or(&serde_json::Value::Null),
            base.get(*k).unwrap_or(&serde_json::Value::Null),
        )
    })
}

/// The baseline text, or `None` when the file does not exist. Any other read
/// error is a failure of the gate, never "no baseline".
fn read_baseline(res: std::io::Result<String>) -> Option<String> {
    match res {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => panic!("read {LAYERED_BASELINE_FILE}: {e}"),
    }
}

/// The names of the solved frames in a list of frame records.
fn solved_names(frames: &serde_json::Value) -> std::collections::BTreeSet<String> {
    frames
        .as_array()
        .into_iter()
        .flatten()
        .filter(|f| f["solved"] == true)
        .filter_map(|f| f["frame"].as_str().map(str::to_string))
        .collect()
}

/// Compare one list of the run against the baseline's: the solved sets must be
/// equal and no solved frame may have a moved WCS key. Returns the number of
/// solved frames compared and a problem line per difference.
fn compare_to_baseline(
    list: &str,
    run: &serde_json::Value,
    base: &serde_json::Value,
) -> (usize, Vec<String>) {
    let mut problems = Vec::new();
    let (rs, bs) = (solved_names(run), solved_names(base));
    for n in rs.difference(&bs) {
        problems.push(format!("{list}: {n} solved now, not in the baseline"));
    }
    for n in bs.difference(&rs) {
        problems.push(format!("{list}: {n} solved in the baseline, not now"));
    }
    let mut n = 0usize;
    for f in run.as_array().into_iter().flatten().filter(|f| f["solved"] == true) {
        let name = f["frame"].as_str().unwrap_or("?");
        let Some(b) = base
            .as_array()
            .into_iter()
            .flatten()
            .find(|b| b["frame"].as_str() == Some(name) && b["solved"] == true)
        else {
            continue;
        };
        n += 1;
        if let Some(key) = wcs_moved(f, b) {
            problems.push(format!("{list}: {name}: WCS key {key} moved"));
        }
    }
    (n, problems)
}


/// Plan 2 real-data gate: the full additive density-tier union must solve the
/// corpus. Opens `tier_500..tier_8000` from `SOLVEMYASTRO_TIER_DIR`, builds
/// `Caches::layered`, and solves every corpus frame. Gates on CORRECTNESS only
/// (M78 + SIP, 0 false positives, all truth frames solved, 0 panics) — not on
/// the frozen `corpus_baseline.json`, because the tier catalog differs from the
/// deep cache the baseline was frozen against.
///
/// It also records every frame's WCS (`frame_json`) in
/// `tests/corpus_layered_run.json`, solves [`INDEX_PASS_FRAMES`] blind through
/// the base pack in `SOLVEMYASTRO_INDEX_DIR` (`solve_with_index`; the variable
/// absent prints `SKIP`), and holds both lists to
/// `tests/corpus_layered_baseline.json` when that file exists: equal solved
/// sets and no moved `crval`, `crpix`, `cd` or SIP coefficient (`wcs_moved`).
///
/// Run with:
///   SOLVEMYASTRO_TIER_DIR=/path/to/catalog_out \
///   SOLVEMYASTRO_INDEX_DIR=/path/to/base_pack \
///     cargo test -p solvemyastro --release --test corpus_bench \
///     -- --ignored corpus_layered_tiers --nocapture
#[test]
#[ignore = "needs tiers (SOLVEMYASTRO_TIER_DIR=<dir with tier_*>) + corpus; run --release --nocapture"]
fn corpus_layered_tiers() {
    let tier_dir = match std::env::var("SOLVEMYASTRO_TIER_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            println!("SKIP: set SOLVEMYASTRO_TIER_DIR to a dir containing tier_*");
            return;
        }
    };
    let bench_dir = bench_dir();
    let bench_dir = bench_dir.as_path();
    if !bench_dir.is_dir() {
        println!("SKIP: bench dir absent at {}", bench_dir.display());
        return;
    }

    // Open the additive tier stack (base → deepest).
    let tier_names = ["tier_500", "tier_2000", "tier_5000", "tier_8000"];
    let mut caches_vec: Vec<StarCache> = Vec::new();
    for t in tier_names {
        let dir = tier_dir.join(t);
        match StarCache::open(&dir) {
            Ok(c) => {
                println!("opened {t} ({} stars)", c.star_count());
                caches_vec.push(c);
            }
            Err(e) => {
                println!("SKIP: open {} failed: {e}", dir.display());
                return;
            }
        }
    }
    let layer_refs: Vec<&StarCache> = caches_vec.iter().collect();
    let caches = solvemyastro::Caches::layered(&layer_refs);

    let cfg = SolveConfig {
        quad_tolerance: 0.007,
        catalog_mag_limit: 19.0,
        fit_sip: true,
        print_timing: false,
        ..SolveConfig::default()
    };

    let truth = cached_truth();
    let mut files: Vec<PathBuf> = std::fs::read_dir(bench_dir)
        .expect("read bench dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
                Some("fits" | "fit" | "xisf")
            )
        })
        .collect();
    files.sort();

    let mut n_truth = 0usize;
    let mut n_solved = 0usize;
    let mut n_correct = 0usize;
    let mut missed: Vec<String> = Vec::new();
    let mut wrong: Vec<String> = Vec::new();
    let mut panicked: Vec<String> = Vec::new();
    let mut m78_correct = false;
    let mut m78_sip = false;
    let mut tier_frames: Vec<serde_json::Value> = Vec::new();
    let mut bounded_pick: Option<(PathBuf, SolveHints, f64, usize, u128)> = None;

    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if SKIP_FRAMES.iter().any(|s| name.contains(*s)) {
            continue;
        }
        let truth_entry = truth.iter().find(|&&(k, ..)| name.contains(k));
        let (truth_solved, tra, tdec, tscale) = match truth_entry {
            Some(&(_, ra, dec, sc)) => (true, ra, dec, sc),
            None => (false, 0.0, 0.0, 0.0),
        };
        if truth_solved {
            n_truth += 1;
        }

        let hdr = if name.ends_with(".xisf") {
            read_xisf_header(path)
        } else {
            read_fits_header(path)
        };
        let n1 = hdr.get("NAXIS1").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        let n2 = hdr.get("NAXIS2").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
        let hints = hints_from_fits(&hdr, n1.max(n2));

        let t0 = Instant::now();
        let traced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solvemyastro::solve_traced(path, &hints, &caches, &[], &cfg, None)
        }));
        let elapsed_ms = t0.elapsed().as_millis();
        let (res, traces) = match traced {
            Ok((r, t)) => (Ok(r), t),
            Err(e) => (Err(e), Vec::new()),
        };
        tier_frames.push(match &res {
            Ok(r) => traced_frame_json(&name, r, &traces),
            Err(_) => frame_json(&name, &Err(anyhow::anyhow!("panicked"))),
        });
        // The quickest correct truth frame feeds the `solve_bounded` check.
        if let (Ok(Ok(s)), true) = (&res, truth_solved) {
            let near = sep_arcsec(tra, tdec, s.wcs.crval.0, s.wcs.crval.1) < 30.0
                && (s.pixel_scale_arcsec - tscale).abs() / tscale * 100.0 < 2.0;
            if near && bounded_pick.as_ref().is_none_or(|b| elapsed_ms < b.4) {
                bounded_pick =
                    Some((path.clone(), hints, tscale, tier_frames.len() - 1, elapsed_ms));
            }
        }
        let (solved, ra, dec, scale, has_sip) = match res {
            Ok(Ok(s)) => (
                true,
                s.wcs.crval.0,
                s.wcs.crval.1,
                s.pixel_scale_arcsec,
                s.wcs.sip_forward.is_some(),
            ),
            Ok(Err(_)) => (false, 0.0, 0.0, 0.0, false),
            Err(_) => {
                panicked.push(name.clone());
                (false, 0.0, 0.0, 0.0, false)
            }
        };
        if solved {
            n_solved += 1;
        }

        let dpos = if truth_solved && solved {
            sep_arcsec(tra, tdec, ra, dec)
        } else {
            -1.0
        };
        let dscale = if truth_solved && solved && tscale > 0.0 {
            (scale - tscale).abs() / tscale * 100.0
        } else {
            -1.0
        };
        let correct = truth_solved
            && solved
            && dpos >= 0.0
            && dpos < 30.0
            && (tscale <= 0.0 || dscale < 2.0);
        if correct {
            n_correct += 1;
        }
        if solved && truth_solved && !correct {
            wrong.push(name.clone());
        }
        if !solved && truth_solved {
            missed.push(name.clone());
        }
        if name.contains("2024-01-13_20-27-32_L_-10.00_120.00s_0081") {
            m78_correct = correct;
            m78_sip = has_sip;
        }

        let tag = if solved {
            if !truth_solved {
                "solved"
            } else if correct {
                "OK"
            } else {
                "WRONG"
            }
        } else if truth_solved {
            "MISSED"
        } else {
            "unsolved"
        };
        let detail = if solved {
            format!(
                "ra={ra:8.3} dec={dec:+8.3} {scale:6.3}\"/px {sip} dpos={dpos_s} {elapsed_ms:>5}ms",
                sip = if has_sip { "SIP" } else { "   " },
                dpos_s = if dpos >= 0.0 {
                    format!("{dpos:6.1}\"")
                } else {
                    "    -- ".to_string()
                },
            )
        } else {
            String::new()
        };
        println!("{:50} {tag:>8}  {detail}", &name[..name.len().min(50)]);
    }

    println!(
        "\nLAYERED TIERS: truth={n_truth} solved={n_solved} correct={n_correct} \
         wrong={} missed={} panicked={}",
        wrong.len(),
        missed.len(),
        panicked.len()
    );

    // Blind index pass: the base pack, no hints beyond the epoch.
    let mut index_frames: Vec<serde_json::Value> = Vec::new();
    let mut index_correct = 0usize;
    let mut index_wrong: Vec<String> = Vec::new();
    let mut index_panicked: Vec<String> = Vec::new();
    let mut index_ran = false;
    match std::env::var("SOLVEMYASTRO_INDEX_DIR") {
        Err(_) => println!("SKIP: SOLVEMYASTRO_INDEX_DIR not set"),
        Ok(d) => {
            let index = solvemyastro::QuadIndex::open(Path::new(&d))
                .unwrap_or_else(|e| panic!("open quad index {d}: {e}"));
            index_ran = true;
            for path in &files {
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                if !INDEX_PASS_FRAMES.iter().any(|k| name.contains(k)) {
                    continue;
                }
                let &(_, tra, tdec, tscale) = truth
                    .iter()
                    .find(|&&(k, ..)| name.contains(k))
                    .unwrap_or_else(|| panic!("index pass frame {name} has no truth entry"));
                let hdr = if name.ends_with(".xisf") {
                    read_xisf_header(path)
                } else {
                    read_fits_header(path)
                };
                let n1 = hdr.get("NAXIS1").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                let n2 = hdr.get("NAXIS2").and_then(|s| s.parse::<u32>().ok()).unwrap_or(0);
                let hints = SolveHints {
                    epoch: hints_from_fits(&hdr, n1.max(n2)).epoch,
                    ..SolveHints::default()
                };
                let traced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    solvemyastro::solve_traced(path, &hints, &caches, &[&index], &cfg, None)
                }));
                let did_panic = traced.is_err();
                let (res, traces) = traced
                    .unwrap_or_else(|_| (Err(anyhow::anyhow!("panicked")), Vec::new()));
                // A solve outside the truth's 30" / 2 % is WRONG: reported with
                // its position and scale, kept under "wrong" (a key `wcs_moved`
                // ignores), never recorded as a solve.
                let mut wrong_solve: Option<(f64, f64, f64)> = None;
                let correct = match &res {
                    Ok(s) => {
                        let ok = sep_arcsec(tra, tdec, s.wcs.crval.0, s.wcs.crval.1) < 30.0
                            && (s.pixel_scale_arcsec - tscale).abs() / tscale * 100.0 < 2.0;
                        if !ok {
                            wrong_solve =
                                Some((s.wcs.crval.0, s.wcs.crval.1, s.pixel_scale_arcsec));
                        }
                        ok
                    }
                    Err(_) => false,
                };
                if correct {
                    index_correct += 1;
                }
                let tag = if correct {
                    "OK".to_string()
                } else if did_panic {
                    index_panicked.push(name.clone());
                    "PANICKED".to_string()
                } else if let Some((ra, dec, sc)) = wrong_solve {
                    index_wrong.push(name.clone());
                    format!("WRONG ra={ra:.3} dec={dec:+.3} {sc:.3}\"/px")
                } else {
                    "MISSED".to_string()
                };
                println!("index pass: {:50} {tag}", &name[..name.len().min(50)]);
                let mut rec = if correct {
                    traced_frame_json(&name, &res, &traces)
                } else {
                    frame_json(&name, &Err(anyhow::anyhow!("not a truth solve")))
                };
                if let Some((ra, dec, sc)) = wrong_solve {
                    rec["wrong"] = serde_json::json!({
                        "crval": [ra, dec],
                        "scale_arcsec_px": sc,
                        "wcs": frame_json(&name, &res),
                    });
                }
                index_frames.push(rec);
            }
        }
    }
    println!("index pass: {index_correct} solved of {}", INDEX_PASS_FRAMES.len());
    panicked.extend(index_panicked);

    // `solve_bounded`: a window around a frame's true scale solves the frame
    // to the same field as `solve`. The bounded ladder is B2's, a different
    // ladder from the hinted one, so it may seed in another cell and end on a
    // slightly different WCS: the check is the same field (10" in position,
    // 1 % in scale), not the same bits.
    let (bpath, bhints, bscale, bidx, _) =
        bounded_pick.expect("a correct truth frame to check solve_bounded on");
    let bname = bpath.file_name().unwrap().to_string_lossy().to_string();
    let bounded = solvemyastro::orchestrate::solve_bounded(
        &bpath,
        &bhints,
        &caches,
        &[],
        &cfg,
        None,
        (bscale * 0.9, bscale * 1.1),
    );
    let bounded_problem: Option<String> = match &bounded {
        Ok(b) => {
            let r = &tier_frames[bidx];
            let sep = sep_arcsec(
                b.wcs.crval.0,
                b.wcs.crval.1,
                r["crval"][0].as_f64().unwrap(),
                r["crval"][1].as_f64().unwrap(),
            );
            let dscale = (b.pixel_scale_arcsec / bscale - 1.0).abs() * 100.0;
            println!(
                "solve_bounded: {bname} {sep:.2} arcsec from solve's centre, scale {dscale:.3} % \
                 from the truth's, matched {} vs {}",
                b.matched_stars, r["matched_stars"]
            );
            (sep >= 10.0 || dscale >= 1.0)
                .then(|| format!("{bname}: {sep:.2} arcsec, scale {dscale:.3} % off"))
        }
        Err(e) => Some(format!("{bname}: solve_bounded failed: {e}")),
    };

    let run = serde_json::json!({
        "schema_version": 1,
        "frames": tier_frames,
        "index_frames": index_frames,
    });
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::fs::write(
        manifest.join(LAYERED_RUN_FILE),
        serde_json::to_string_pretty(&run).expect("serialize run"),
    )
    .unwrap_or_else(|e| panic!("write {LAYERED_RUN_FILE}: {e}"));

    // The WCS gate: both lists against the frozen baseline. Every line is
    // printed before any assert, so one run shows every difference.
    let freezing = std::env::var("SOLVEMYASTRO_FREEZE").is_ok_and(|v| v == "1");
    let mut problems: Vec<String> = Vec::new();
    match read_baseline(std::fs::read_to_string(manifest.join(LAYERED_BASELINE_FILE))) {
        None if freezing => println!("WCS gate: no baseline, wrote the run"),
        None => {
            println!(
                "WCS gate: no baseline at {LAYERED_BASELINE_FILE} \
                 (SOLVEMYASTRO_FREEZE=1 freezes the run)"
            );
            problems.push(format!("no baseline at {LAYERED_BASELINE_FILE}"));
        }
        Some(text) => {
            let base: serde_json::Value = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("parse {LAYERED_BASELINE_FILE}: {e}"));
            let (n_t, more_t) = compare_to_baseline("frames", &run["frames"], &base["frames"]);
            let (n_i, more_i) =
                compare_to_baseline("index_frames", &run["index_frames"], &base["index_frames"]);
            problems.extend(more_t);
            problems.extend(more_i);
            let moved = problems.iter().filter(|p| p.contains("moved")).count();
            println!("WCS gate: {} solved frames, {moved} moved", n_t + n_i);
            for p in &problems {
                println!("WCS gate: {p}");
            }
        }
    }

    assert!(panicked.is_empty(), "GATE FAIL: panic(s): {panicked:?}");
    assert!(m78_correct, "GATE FAIL: M78 did not solve correctly on the tier stack");
    assert!(m78_sip, "GATE FAIL: M78 solved without SIP");
    assert!(wrong.is_empty(), "GATE FAIL: false positive(s): {wrong:?}");
    assert!(missed.is_empty(), "GATE FAIL: truth frame(s) not solved: {missed:?}");
    assert!(index_ran, "GATE FAIL: the blind index pass did not run (SOLVEMYASTRO_INDEX_DIR)");
    assert!(bounded_problem.is_none(), "GATE FAIL: solve_bounded: {bounded_problem:?}");
    assert!(index_wrong.is_empty(), "GATE FAIL: wrong blind index solve(s): {index_wrong:?}");
    assert_eq!(
        index_correct,
        INDEX_PASS_FRAMES.len(),
        "GATE FAIL: blind index pass solved {index_correct} of {}",
        INDEX_PASS_FRAMES.len()
    );
    assert!(problems.is_empty(), "GATE FAIL: WCS differs from the baseline: {problems:#?}");
    println!("\nGATE: layered-tier corpus PASSED");
}

#[test]
fn wcs_compare_flags_a_moved_coefficient() {
    use serde_json::json;
    let frame = || {
        json!({
            "frame": "f",
            "solved": true,
            "crval": [10.5, 41.25],
            "crpix": [1000.0, 750.0],
            "cd": [-1.0e-4, 2.0e-6, 3.0e-6, 1.0e-4],
            "sip": {
                "a_order": 3,
                "a": [[0.0, 0.0, 1.5e-7], [0.0, 2.0e-8], [3.0e-9]],
                "b_order": 3,
                "b": [[0.0, 0.0, -1.5e-7], [0.0, 4.0e-8], [5.0e-9]],
                "ap_order": 3,
                "ap": [[0.0, 0.0, 1.0e-7], [0.0, 2.0e-8], [3.0e-9]],
                "bp_order": 3,
                "bp": [[0.0, 0.0, -1.0e-7], [0.0, 4.0e-8], [5.0e-9]],
            },
            "matched_stars": 40,
            "inlier_ratio": 0.3,
            "rms_px": 0.2,
            "elapsed_ms": 1234,
        })
    };
    let base = frame();
    assert_eq!(wcs_moved(&frame(), &base), None);

    // A run's timing and counts are not WCS keys.
    let mut other = frame();
    other["elapsed_ms"] = json!(99999);
    other["matched_stars"] = json!(41);
    assert_eq!(wcs_moved(&other, &base), None);

    let mut moved = frame();
    let c = moved["sip"]["a"][1][1].as_f64().unwrap();
    moved["sip"]["a"][1][1] = json!(c * (1.0 + 1e-6));
    assert_eq!(wcs_moved(&moved, &base).as_deref(), Some("sip.a[1][1]"));

    let mut same = frame();
    same["sip"]["a"][1][1] = json!(c * (1.0 + 1e-12));
    assert_eq!(wcs_moved(&same, &base), None);

    let mut shifted = frame();
    shifted["crpix"][0] = json!(1000.001);
    assert_eq!(wcs_moved(&shifted, &base).as_deref(), Some("crpix[0]"));

    // SIP present on one side only, and a zero that became non-zero.
    let mut no_sip = frame();
    no_sip["sip"] = serde_json::Value::Null;
    assert_eq!(wcs_moved(&no_sip, &base).as_deref(), Some("sip"));
    let mut zero = frame();
    zero["sip"]["a"][0][0] = json!(1e-30);
    assert_eq!(wcs_moved(&zero, &base).as_deref(), Some("sip.a[0][0]"));
}

#[test]
fn only_a_missing_baseline_reads_as_no_baseline() {
    use std::io::{Error, ErrorKind};
    assert_eq!(read_baseline(Ok("{}".to_string())).as_deref(), Some("{}"));
    assert_eq!(read_baseline(Err(Error::from(ErrorKind::NotFound))), None);
    let denied = std::panic::catch_unwind(|| {
        read_baseline(Err(Error::from(ErrorKind::PermissionDenied)))
    });
    assert!(denied.is_err(), "a permission error must not read as no baseline");
}
