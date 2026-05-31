//! solvemyastro CLI — standalone plate solver.
//!
//! Plain `std::env::args` parsing (project convention — no clap).
//!
//! Usage:
//!   solvemyastro solve <image> --catalog <dir> [--ra <d> --dec <d>]
//!                [--fov <d> | --scale <as/px>] [--radius <d>] [--json]
//!   solvemyastro build-cache --from <legacy-dir> --out <cache-dir> [--epoch <year>]
//!   solvemyastro build-bright-cache --from <deep-cache-dir> --out <bright-cache-dir>
//!                [--bright-floor 14.0] [--target-density 100]
//!                [--mode hybrid|floor-only|cap-only]
//!   solvemyastro cache-info <cache-dir>

use std::path::PathBuf;

use anyhow::{bail, Result};

use solvemyastro::cache::{build_cache, build_cache_from_legacy_dir, BuildProgress, StarRecord};
use solvemyastro::{solve, SolveConfig, SolveHints, StarCache};

fn opt(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}
fn opt_f64(args: &[String], flag: &str) -> Option<f64> {
    opt(args, flag).and_then(|v| v.parse().ok())
}
fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).map(String::as_str).unwrap_or("");
    let rest = &argv[2.min(argv.len())..];

    match cmd {
        "solve" => {
            let image = rest
                .first()
                .filter(|a| !a.starts_with("--"))
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("usage: solvemyastro solve <image> --catalog <dir> [...]"))?;
            let catalog = opt(rest, "--catalog")
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("--catalog <cache-dir> is required"))?;
            let cache = StarCache::open(&catalog)?;
            let hints = SolveHints {
                ra: opt_f64(rest, "--ra"),
                dec: opt_f64(rest, "--dec"),
                fov_deg: opt_f64(rest, "--fov"),
                pixel_scale_arcsec: opt_f64(rest, "--scale"),
                search_radius_deg: opt_f64(rest, "--radius"),
                epoch: opt_f64(rest, "--epoch"),
            };
            let sol = solve(&image, &hints, &solvemyastro::Caches::deep_only(&cache), &SolveConfig::default(), None)?;
            if has(rest, "--json") {
                println!(
                    "{{\"ra\":{},\"dec\":{},\"scale\":{},\"rot\":{},\"matched\":{}}}",
                    sol.wcs.crval.0,
                    sol.wcs.crval.1,
                    sol.pixel_scale_arcsec,
                    sol.field_rotation_deg,
                    sol.matched_stars
                );
            } else {
                println!("{sol:#?}");
            }
            Ok(())
        }
        "build-cache" => {
            let from = opt(rest, "--from")
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("--from <legacy-dir> is required"))?;
            let out = opt(rest, "--out")
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("--out <cache-dir> is required"))?;
            // Julian year of the catalog positions. Gaia DR3 = 2016.0.
            let epoch = opt_f64(rest, "--epoch").unwrap_or(2016.0);
            std::fs::create_dir_all(&out)
                .map_err(|e| anyhow::anyhow!("create --out {}: {e}", out.display()))?;
            eprintln!(
                "build-cache: {} -> {} (epoch J{epoch})",
                from.display(),
                out.display()
            );
            let total = build_cache_from_legacy_dir(&from, &out, epoch, |p| match p {
                BuildProgress::Ingesting { records } => {
                    eprintln!("  ingesting … {records} records");
                }
                BuildProgress::Finalizing { shards_done, shards_total } => {
                    eprintln!("  finalizing {shards_done}/{shards_total} shards");
                }
                BuildProgress::Complete { records } => {
                    eprintln!("  done — {records} records");
                }
            })?;
            println!(
                "build-cache: wrote {total} stars to {}",
                out.join("stars.smac").display()
            );
            Ok(())
        }
        "build-bright-cache" => {
            // Derive a smaller "bright" star cache from an existing deep
            // cache. Hybrid algorithm: per HEALPix-6 pixel, include all
            // stars with mag < bright_floor (floor), then if the floor
            // count is below target_density, top up with the next-brightest
            // stars (continuing past the floor) until reaching the target
            // or exhausting the pixel.
            let from = opt(rest, "--from")
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("--from <deep-cache-dir> is required"))?;
            let out = opt(rest, "--out")
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("--out <bright-cache-dir> is required"))?;
            let bright_floor: f32 = opt_f64(rest, "--bright-floor")
                .unwrap_or(14.0) as f32;
            let target_density: usize = opt(rest, "--target-density")
                .and_then(|v| v.parse().ok())
                .unwrap_or(100);
            let mode: String = opt(rest, "--mode").unwrap_or_else(|| "hybrid".to_string());
            anyhow::ensure!(
                matches!(mode.as_str(), "hybrid" | "floor-only" | "cap-only"),
                "--mode must be one of: hybrid (default), floor-only, cap-only (got {mode})"
            );
            std::fs::create_dir_all(&out)
                .map_err(|e| anyhow::anyhow!("create --out {}: {e}", out.display()))?;
            let src = StarCache::open(&from)?;
            let src_epoch = src.catalog_epoch();
            let src_count = src.star_count();
            eprintln!(
                "build-bright-cache: {} (J{src_epoch}, {src_count} stars) → {} \
                 [mode={mode} floor={bright_floor} target={target_density}]",
                from.display(), out.display(),
            );

            // Stream per-pixel selections into the existing build_cache.
            // Buffer per pixel is small (< 1000 stars typically); pre-sort
            // happens on disk (records are mag-sorted ascending per pixel).
            const N_PIXELS: u64 = 49_152; // depth=6
            let bright_floor_v = bright_floor;
            let target_density_v = target_density;
            let mode_v = mode.clone();
            let pixel_iter = (0..N_PIXELS).flat_map(move |px| {
                let records = src.iter_pixel_records(px).unwrap_or_default();
                let selected: Vec<StarRecord> = match mode_v.as_str() {
                    "floor-only" => {
                        records.into_iter().take_while(|r| r.mag < bright_floor_v).collect()
                    }
                    "cap-only" => {
                        records.into_iter().take(target_density_v).collect()
                    }
                    _ /* hybrid */ => {
                        let floor: Vec<StarRecord> = records.iter()
                            .take_while(|r| r.mag < bright_floor_v)
                            .copied()
                            .collect();
                        if floor.len() >= target_density_v {
                            floor
                        } else {
                            let mut s = floor;
                            let need = target_density_v - s.len();
                            s.extend(records.iter().skip(s.len()).take(need).copied());
                            s
                        }
                    }
                };
                selected.into_iter()
            });

            let total = build_cache(pixel_iter, &out, src_epoch, |p| match p {
                BuildProgress::Ingesting { records } => {
                    if records % 500_000 == 0 {
                        eprintln!("  ingesting … {records} records");
                    }
                }
                BuildProgress::Finalizing { shards_done, shards_total } => {
                    if shards_done % 5000 == 0 || shards_done == shards_total {
                        eprintln!("  finalizing {shards_done}/{shards_total} pixels");
                    }
                }
                BuildProgress::Complete { records } => {
                    eprintln!("  done — {records} records");
                }
            })?;

            // Sidecar metadata for debugging + future stale-cache detection.
            let meta = format!(
                "{{\n  \"source_smac_size\": {},\n  \"source_epoch\": {},\n  \
                 \"bright_floor\": {},\n  \"target_density\": {},\n  \
                 \"mode\": \"{}\",\n  \"output_star_count\": {}\n}}\n",
                src_count, src_epoch, bright_floor, target_density, mode, total,
            );
            std::fs::write(out.join("bright.meta.json"), &meta)
                .map_err(|e| anyhow::anyhow!("write bright.meta.json: {e}"))?;
            println!(
                "build-bright-cache: wrote {total} stars to {}",
                out.join("stars.smac").display()
            );
            Ok(())
        }
        "cache-info" => {
            let dir = rest
                .first()
                .filter(|a| !a.starts_with("--"))
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("usage: solvemyastro cache-info <cache-dir>"))?;
            let cache = StarCache::open(&dir)?;
            println!("cache-info: {}", dir.display());
            println!("  stars:         {}", cache.star_count());
            println!("  catalog epoch: J{}", cache.catalog_epoch());
            Ok(())
        }
        "" | "-h" | "--help" => {
            println!(
                "solvemyastro — standalone plate solver (precise WCS + SIP)\n\n\
                 USAGE:\n  \
                 solvemyastro solve <image> --catalog <dir> [--ra --dec --fov|--scale --radius] [--json]\n  \
                 solvemyastro build-cache --from <src-dir> --out <cache-dir>\n  \
                 solvemyastro cache-info <cache-dir>"
            );
            Ok(())
        }
        other => bail!("unknown subcommand '{other}' (try --help)"),
    }
}
