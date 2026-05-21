//! solvemyastro CLI — standalone plate solver.
//!
//! Plain `std::env::args` parsing (project convention — no clap). Phase 0:
//! subcommands parse and dispatch into the (stubbed) lib so the interface
//! is fixed; bodies land with their lib phases.
//!
//! Usage:
//!   solvemyastro solve <image> --catalog <dir> [--ra <d> --dec <d>]
//!                [--fov <d> | --scale <as/px>] [--radius <d>] [--json]
//!   solvemyastro build-cache --from <src-dir> --out <cache-dir>
//!   solvemyastro cache-info <cache-dir>

use std::path::PathBuf;

use anyhow::{bail, Result};

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
            };
            let sol = solve(&image, &hints, &cache, &SolveConfig::default(), None)?;
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
                .ok_or_else(|| anyhow::anyhow!("--from <src-dir> is required"))?;
            let out = opt(rest, "--out")
                .ok_or_else(|| anyhow::anyhow!("--out <cache-dir> is required"))?;
            bail!("build-cache not implemented yet (plan Phase 1): {from} -> {out}")
        }
        "cache-info" => {
            let dir = rest
                .first()
                .map(PathBuf::from)
                .ok_or_else(|| anyhow::anyhow!("usage: solvemyastro cache-info <cache-dir>"))?;
            let _ = StarCache::open(&dir)?;
            bail!("cache-info not implemented yet (plan Phase 1)")
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
