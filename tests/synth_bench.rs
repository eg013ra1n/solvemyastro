//! Solve the synthetic star fields written by `examples/synth_fields.rs` and
//! score each against the true solution in its `SYN*` header keywords.
//!
//! Run with:
//!   SOLVEMYASTRO_SYNTH_DIR=<frames> SOLVEMYASTRO_TIER_DIR=<dir with tier_*> \
//!     cargo test --release --test synth_bench -- --ignored --nocapture
//!
//! Hints are built the way a capture program's headers would give them:
//! OBJCTRA/OBJCTDEC (deliberately off by up to 1°), FOCALLEN, XPIXSZ and
//! XBINNING for the scale. A frame is correct when its centre is within
//! max(30″, 5 px) of the truth and its scale within 1 %; imprecise when it
//! found the right field (centre within 5 % of the FOV, scale within 2 %) but
//! missed that; wrong otherwise.
//!
//! `SYNTH_MODE` picks which hints are passed (hinted / position / scale /
//! blind); `SYNTH_FRAMES` keeps only files whose name contains it.
//!
//! GATE (asserted): no wrong solutions and no panics; imprecise ones are
//! reported. Frames the solver
//! cannot solve are reported per focal length, not failed, because wide
//! lenses sit outside the solver's FOV ladder by design.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use solvemyastro::{solve, Caches, SolveConfig, SolveHints, StarCache};

/// Primary-header keyword → value (quotes stripped).
fn read_header(path: &Path) -> HashMap<String, String> {
    use std::io::Read;
    let mut buf = vec![0u8; 2880 * 4];
    let n = std::fs::File::open(path).and_then(|mut f| f.read(&mut buf)).unwrap_or(0);
    let mut map = HashMap::new();
    for card in buf[..n].chunks(80) {
        let s = String::from_utf8_lossy(card);
        if s.starts_with("END ") {
            break;
        }
        if s.len() >= 10 && &s[8..10] == "= " {
            let v = s[10..].split(" /").next().unwrap_or("").trim().trim_matches('\'').trim();
            map.insert(s[..8].trim().to_string(), v.to_string());
        }
    }
    map
}

fn sexagesimal(s: &str) -> Option<f64> {
    let neg = s.trim_start().starts_with('-');
    let p: Vec<f64> = s
        .trim()
        .trim_start_matches(['+', '-'])
        .split([' ', ':'])
        .filter(|t| !t.is_empty())
        .map(|t| t.parse().ok())
        .collect::<Option<_>>()?;
    let v = p.first()? + p.get(1).unwrap_or(&0.0) / 60.0 + p.get(2).unwrap_or(&0.0) / 3600.0;
    Some(if neg { -v } else { v })
}

/// Which header hints the solver gets (`SYNTH_MODE`): `hinted` (position and
/// scale, the default), `position` (no scale), `scale` (no position) or
/// `blind` (neither).
fn hints_for(h: &HashMap<String, String>, mode: &str) -> SolveHints {
    let num = |k: &str| h.get(k).and_then(|v| v.parse::<f64>().ok());
    let scale = match (num("FOCALLEN"), num("XPIXSZ")) {
        (Some(f), Some(p)) if f > 0.0 && p > 0.0 => {
            Some(206.264_806 * p * num("XBINNING").unwrap_or(1.0) / f)
        }
        _ => None,
    };
    let epoch = h.get("DATE-OBS").and_then(|d| d.get(..4)?.parse::<f64>().ok()).map(|y| y + 0.5);
    let position = matches!(mode, "hinted" | "position");
    let with_scale = matches!(mode, "hinted" | "scale");
    SolveHints {
        ra: h.get("OBJCTRA").and_then(|s| sexagesimal(s)).map(|v| v * 15.0).filter(|_| position),
        dec: h.get("OBJCTDEC").and_then(|s| sexagesimal(s)).filter(|_| position),
        fov_deg: None,
        pixel_scale_arcsec: scale.filter(|_| with_scale),
        search_radius_deg: Some(15.0),
        epoch,
    }
}

fn sep_arcsec(r1: f64, d1: f64, r2: f64, d2: f64) -> f64 {
    let (r1, d1, r2, d2) = (r1.to_radians(), d1.to_radians(), r2.to_radians(), d2.to_radians());
    let a = (d2.cos() * (r2 - r1).sin()).hypot(d1.cos() * d2.sin() - d1.sin() * d2.cos() * (r2 - r1).cos());
    let b = d1.sin() * d2.sin() + d1.cos() * d2.cos() * (r2 - r1).cos();
    a.atan2(b).to_degrees() * 3600.0
}

#[test]
#[ignore = "needs SOLVEMYASTRO_SYNTH_DIR (synth_fields output) + SOLVEMYASTRO_TIER_DIR"]
fn synth_bench() {
    let (Some(dir), Some(tiers)) = (
        std::env::var_os("SOLVEMYASTRO_SYNTH_DIR").map(PathBuf::from),
        std::env::var_os("SOLVEMYASTRO_TIER_DIR").map(PathBuf::from),
    ) else {
        println!("SKIP: set SOLVEMYASTRO_SYNTH_DIR and SOLVEMYASTRO_TIER_DIR");
        return;
    };
    let caches: Vec<StarCache> = ["tier_500", "tier_2000", "tier_5000", "tier_8000"]
        .iter()
        .map(|t| StarCache::open(&tiers.join(t)).expect("open tier"))
        .collect();
    let refs: Vec<&StarCache> = caches.iter().collect();
    let caches = Caches::layered(&refs);
    let cfg = SolveConfig::default();
    let mode = std::env::var("SYNTH_MODE").unwrap_or_else(|_| "hinted".into());
    assert!(
        matches!(mode.as_str(), "hinted" | "position" | "scale" | "blind"),
        "SYNTH_MODE must be hinted, position, scale or blind"
    );
    let only = std::env::var("SYNTH_FRAMES").ok();

    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read synth dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "fits"))
        .filter(|p| only.as_ref().is_none_or(|o| p.to_string_lossy().contains(o.as_str())))
        .collect();
    files.sort();

    // focal → (frames, correct, imprecise, total ms, wrong)
    let mut by_focal: BTreeMap<u32, (usize, usize, usize, u128, usize)> = BTreeMap::new();
    let mut wrong = Vec::new();
    let mut panicked = Vec::new();
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let h = read_header(path);
        let num = |k: &str| h.get(k).and_then(|v| v.parse::<f64>().ok()).unwrap_or(f64::NAN);
        let (tra, tdec, tscale, focal) = (num("SYNRA"), num("SYNDEC"), num("SYNSCALE"), num("FOCALLEN"));
        let hints = hints_for(&h, &mode);

        let t0 = Instant::now();
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            solve(path, &hints, &caches, &cfg, None)
        }));
        let ms = t0.elapsed().as_millis();
        let row = by_focal.entry(focal as u32).or_default();
        row.0 += 1;
        row.3 += ms;
        let line = match res {
            Ok(Ok(s)) => {
                let dpos = sep_arcsec(tra, tdec, s.wcs.crval.0, s.wcs.crval.1);
                let dscale = (s.pixel_scale_arcsec - tscale).abs() / tscale * 100.0;
                let fov_arcsec = tscale * num("NAXIS1").max(num("NAXIS2"));
                let tag = if dpos < (5.0 * tscale).max(30.0) && dscale < 1.0 {
                    row.1 += 1;
                    "OK"
                } else if dpos < 0.05 * fov_arcsec && dscale < 2.0 {
                    row.2 += 1;
                    "IMPRECISE"
                } else {
                    row.4 += 1;
                    wrong.push(name.clone());
                    "WRONG"
                };
                format!(
                    "{tag:>9} dpos={dpos:7.1}\" dscale={dscale:5.2}% m={:4} rms={:4.2}px",
                    s.matched_stars,
                    s.rms_residual_px
                )
            }
            Ok(Err(e)) => format!(" MISSED {}", e.to_string().lines().next().unwrap_or("")),
            Err(_) => {
                panicked.push(name.clone());
                " PANIC".to_string()
            }
        };
        println!("{name:26} {ms:>6}ms {line}");
    }

    println!("\nmode={mode}\n focal   frames  correct  imprecise  wrong   avg ms");
    for (f, (n, ok, rough, ms, bad)) in &by_focal {
        println!("{f:>5}mm {n:>7} {ok:>8} {rough:>10} {bad:>6} {:>8}", ms / (*n as u128).max(1));
    }
    assert!(panicked.is_empty(), "GATE FAIL: panic(s): {panicked:?}");
    assert!(wrong.is_empty(), "GATE FAIL: wrong solution(s): {wrong:?}");
}
