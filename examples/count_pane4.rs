//! Solves the "Pane 4" bench frame with the deep catalog only.
//!
//! The data paths come from `$SOLVEMYASTRO_BENCH_DIR` (default `$HOME/Pictures/Astro/Platesolve Bench`)
//! and `$SOLVEMYASTRO_SMAC_GAIA` (default
//! `$HOME/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia`).

fn env_dir(var: &str, under_home: &str) -> std::path::PathBuf {
    std::env::var_os(var).map(std::path::PathBuf::from).unwrap_or_else(|| {
        std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(under_home)
    })
}

fn main() {
    let cache = solvemyastro::StarCache::open(
        &env_dir(
            "SOLVEMYASTRO_SMAC_GAIA",
            "Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia",
        ),
    ).unwrap();

    println!("=== Test 1: CRVAL position hint + blind scale ===");
    let hints = solvemyastro::SolveHints {
        ra: Some(37.232),   // CRVAL1 from XISF header
        dec: Some(60.829),  // CRVAL2 from XISF header
        fov_deg: None,
        pixel_scale_arcsec: None,
        search_radius_deg: Some(15.0),
        epoch: None,
    };
    let cfg = solvemyastro::SolveConfig::default();
    let path = env_dir("SOLVEMYASTRO_BENCH_DIR", "Pictures/Astro/Platesolve Bench")
        .join("Light_Pane 4_300.0s_Bin1_O_gain111_20211011-024956_-10.0C_0002_c_lps_r.xisf");
    let path = path.as_path();

    match solvemyastro::solve(path, &hints, &solvemyastro::Caches::deep_only(&cache), &cfg, None) {
        Ok(sol) => println!("SOLVED: RA={:.4} Dec={:.4} scale={:.3}\"/px inliers={}", 
            sol.wcs.crval.0, sol.wcs.crval.1, sol.pixel_scale_arcsec, sol.matched_stars),
        Err(e) => println!("FAILED: {e}"),
    }
}
