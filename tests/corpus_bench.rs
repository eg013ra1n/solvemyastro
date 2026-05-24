//! Corpus acceptance gate for solvemyastro.
//!
//! Run with:
//!   cargo test -p solvemyastro --release --test corpus_bench -- --ignored --nocapture
//!
//! GATE (asserted):
//!   - Every non-skipped frame with a reference-truth entry is solved here.
//!   - M78 (`2024-01-13_20-27-32_L_-10.00_120.00s_0081.fits`) solves within tolerance.
//!   - M78's solution carries SIP (`sip_forward.is_some()`).
//!   - 0 false positives (wrong-position "solve").
//!   - 0 panics.
//!   - Skipped entirely (not solved, not counted, not gated) — see
//!     `SKIP_FRAMES`: `_VSH8902` (no known truth, unsolvable on the bench
//!     catalog), `_DSC5767` and `_DSC2522` (DSLR wide-field, out of
//!     target corpus scope).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use solvemyastro::{solve, SolveConfig, SolveHints, StarCache};

// ---------------------------------------------------------------------------
// Corpus directory + cached reference-truth table
// ---------------------------------------------------------------------------

const BENCH_DIR: &str =
    "/Volumes/BigMac/Users/astrobureau/Pictures/Astro/Platesolve Bench";
const SMAC_GAIA: &str = "/Volumes/BigMac/Users/astrobureau/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia";

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
    ]
}

/// Frames excluded from the corpus gate entirely — not solved, not counted,
/// not gated. User-curated scope decision:
///   - `_VSH8902`: no known truth on the bench catalog (unsolvable here).
///   - `_DSC5767`: documented allowed-fail.
///   - `_DSC2522`: DSLR wide-field, out of target corpus scope.
const SKIP_FRAMES: &[&str] = &["_VSH8902", "_DSC5767", "_DSC2522"];

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
    // been drizzled/resampled in PixInsight, where the FOCALLEN/CDELT1 metadata reflects
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
    let bench_dir = Path::new(BENCH_DIR);
    let smac_path = Path::new(SMAC_GAIA);

    if !bench_dir.is_dir() {
        println!("SKIP corpus_bench: bench dir not found at {}", bench_dir.display());
        return;
    }
    if !smac_path.is_dir() {
        println!("SKIP corpus_bench: smac_gaia cache not found at {}", smac_path.display());
        return;
    }

    let cache = StarCache::open(smac_path).expect("open smac_gaia cache");
    let cfg = SolveConfig {
        // Phase 1 instrumentation: print per-stage timing for each frame so we
        // can validate where wall-clock actually goes before committing to the
        // Phase 3 speed work. Strip this back to default() once the data is
        // captured.
        print_timing: true,
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
            solve(path, &hints, &cache, &cfg, None)
        }));

        let elapsed_ms = t0.elapsed().as_millis() as u64;

        let (smc_solved, smc_ra, smc_dec, smc_scale, has_sip, notes) = match solve_result {
            Ok(Ok(sol)) => {
                let ra = sol.wcs.crval.0;
                let dec = sol.wcs.crval.1;
                let scale = sol.pixel_scale_arcsec;
                let sip = sol.wcs.sip_forward.is_some();
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

        println!(
            "{:52} {} {} {} {} {} {:>7}  {}{}{}",
            &name[..name.len().min(52)],
            truth_tag, smc_tag,
            dpos_str, dscale_str,
            inlier_str,
            elapsed_ms,
            notes,
            sip_note,
            flag
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
        });
    }

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

    println!("\nGATE: ALL assertions PASSED");
}
