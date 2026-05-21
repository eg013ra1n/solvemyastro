fn main() {
    let cache = solvemyastro::StarCache::open(
        std::path::Path::new("/Volumes/BigMac/Users/astrobureau/Library/Application Support/com.vsharifov.athenaeum/catalogs/smac_gaia")
    ).unwrap();

    println!("=== Test 1: CRVAL position hint + blind scale ===");
    let hints = solvemyastro::SolveHints {
        ra: Some(37.232),   // CRVAL1 from XISF header
        dec: Some(60.829),  // CRVAL2 from XISF header
        fov_deg: None,
        pixel_scale_arcsec: None,
        search_radius_deg: Some(15.0),
    };
    let cfg = solvemyastro::SolveConfig::default();
    let path = std::path::Path::new("/Volumes/BigMac/Users/astrobureau/Pictures/Astro/Platesolve Bench/Light_Pane 4_300.0s_Bin1_O_gain111_20211011-024956_-10.0C_0002_c_lps_r.xisf");

    match solvemyastro::solve(path, &hints, &cache, &cfg, None) {
        Ok(sol) => println!("SOLVED: RA={:.4} Dec={:.4} scale={:.3}\"/px inliers={}", 
            sol.wcs.crval.0, sol.wcs.crval.1, sol.pixel_scale_arcsec, sol.matched_stars),
        Err(e) => println!("FAILED: {e}"),
    }
}
