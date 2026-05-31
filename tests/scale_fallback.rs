//! Scale-recovery fallback: a real frame solved with a deliberately wrong
//! (aperture-derived) scale hint, using the app-faithful TIERED catalog
//! (deep + bright). Under the wrong hint the hinted attempt fails verification,
//! so the scale-recovery fallback (B1 harvest / B2 sweep) must engage and solve
//! at the known position — not the all-sky degrade. Local-only (needs both
//! caches + the bench frame); skips cleanly when any are absent.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use solvemyastro::{solve, Caches, SolveConfig, SolveHints, StarCache};

fn smac_dir() -> PathBuf {
    if let Ok(p) = std::env::var("ATHENAEUM_SMAC_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home)
        .join("Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia")
}

fn bright_dir() -> PathBuf {
    if let Ok(p) = std::env::var("ATHENAEUM_SMAC_BRIGHT_DIR") {
        return PathBuf::from(p);
    }
    // Convention: sibling of the deep cache directory.
    smac_dir()
        .parent()
        .map(|p| p.join("smac_gaia_bright"))
        .unwrap_or_default()
}

fn frame() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(
        "Pictures/Astro/Platesolve Bench/\
         Light_Pane 4_300.0s_Bin1_O_gain111_20211011-024956_-10.0C_0002_c_lps_r.xisf",
    )
}

#[test]
fn wrong_scale_hint_recovers_at_known_position() {
    let (dir, bdir, f) = (smac_dir(), bright_dir(), frame());
    if !dir.join("stars.smac").exists()
        || !bdir.join("stars.smac").exists()
        || !f.exists()
    {
        eprintln!(
            "SKIP wrong_scale_hint_recovers_at_known_position: cache(s) or frame absent"
        );
        return;
    }
    let deep = StarCache::open(&dir).expect("open deep cache");
    let bright = StarCache::open(&bdir).expect("open bright cache");
    // App-faithful tiered catalog: under the wrong hint the hinted attempt
    // fails verification, so the scale-recovery fallback must engage.
    let caches = Caches::tiered(&deep, &bright);

    // The real-world bug: aperture (140 mm) recorded as focal length -> 3.52"/px
    // (true is 0.879"/px). Position is correct.
    let wrong = SolveHints {
        ra: Some(36.7017),
        dec: Some(60.8800),
        fov_deg: None,
        pixel_scale_arcsec: Some(3.52),
        search_radius_deg: None,
        epoch: Some(2021.7737),
    };
    let cfg = SolveConfig { print_timing: true, ..SolveConfig::default() };

    let cancel = AtomicBool::new(false);
    let sol = solve(&f, &wrong, &caches, &cfg, Some(&cancel))
        .expect("scale-recovery fallback must solve the wrong-scale frame");

    // Correct sky + scale (truth from a clean solve).
    assert!((sol.wcs.crval.0 - 36.70).abs() < 0.2, "RA {:.3}", sol.wcs.crval.0);
    assert!((sol.wcs.crval.1 - 60.88).abs() < 0.2, "Dec {:.3}", sol.wcs.crval.1);
    assert!((sol.pixel_scale_arcsec - 0.879).abs() < 0.05,
            "scale {:.3}", sol.pixel_scale_arcsec);
    assert!(sol.matched_stars >= 10, "inliers {}", sol.matched_stars);

    // Cheap path: must NOT be the all-sky degrade (~160 s). The position-
    // constrained fallback rungs are single-digit seconds.
    assert!(sol.solve_time_ms < 30_000,
            "fallback too slow ({} ms) — likely fell through to all-sky", sol.solve_time_ms);
}
