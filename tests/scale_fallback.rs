//! Scale-recovery fallback: a real frame solved with a deliberately wrong
//! (aperture-derived) scale hint must still solve via B1/B2 at the known
//! position, cheaply — not via the all-sky degrade. Local-only (needs the deep
//! cache + the bench frame); skips cleanly when they are absent.

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

fn frame() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(
        "Pictures/Astro/Platesolve Bench/\
         Light_Pane 4_300.0s_Bin1_O_gain111_20211011-024956_-10.0C_0002_c_lps_r.xisf",
    )
}

#[test]
fn wrong_scale_hint_recovers_at_known_position() {
    let (dir, f) = (smac_dir(), frame());
    if !dir.join("stars.smac").exists() || !f.exists() {
        eprintln!("SKIP wrong_scale_hint_recovers_at_known_position: cache or frame absent");
        return;
    }
    let deep = StarCache::open(&dir).expect("open deep cache");
    let caches = Caches::deep_only(&deep);

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
        .expect("fallback must solve the wrong-scale frame");

    // Correct sky + scale (truth from a clean solve).
    assert!((sol.wcs.crval.0 - 36.70).abs() < 0.2, "RA {:.3}", sol.wcs.crval.0);
    assert!((sol.wcs.crval.1 - 60.88).abs() < 0.2, "Dec {:.3}", sol.wcs.crval.1);
    assert!((sol.pixel_scale_arcsec - 0.879).abs() < 0.05,
            "scale {:.3}", sol.pixel_scale_arcsec);
    assert!(sol.matched_stars >= 10, "inliers {}", sol.matched_stars);

    // Cheap path: must NOT be the all-sky degrade. The all-sky run took
    // ~160 s / ~146k cone calls; B1/B2 are position-constrained (<10 s).
    assert!(sol.solve_time_ms < 30_000,
            "fallback too slow ({} ms) — likely fell through to all-sky", sol.solve_time_ms);
}
