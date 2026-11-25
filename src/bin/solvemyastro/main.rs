//! The `solvemyastro` command: one subcommand per stage, from Gaia files to a
//! solved frame.
//!
//! ```text
//! solvemyastro ingest --gaia <dir> --out <bins>            Gaia DR3 files -> bins
//! solvemyastro build-tiers --bins <bins> --out <tiers>     bins -> density tiers
//! solvemyastro build-index --tiers <tiers> --out <pack>    tiers -> a quad-index pack
//! solvemyastro package --tiers <tiers> --index base=<pack> --out <publish>
//! solvemyastro solve <image> --tiers <tiers> [--index <pack>]...
//! solvemyastro info <dir>
//! ```
//!
//! Results and each command's closing summary go to stdout, progress to stderr
//! through `tracing`. Exit code 0 on success, 1 on a failure or a refused solve, 2 on
//! bad arguments (clap).

mod catalog;
mod info;
mod log;
mod solve;

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "solvemyastro", version, about = "Standalone plate solver and its catalog tools")]
struct Cli {
    /// Also write every log event as one JSON line to this file.
    #[arg(long, global = true, value_name = "FILE")]
    log: Option<PathBuf>,

    /// Show warnings and errors only (RUST_LOG still overrides).
    #[arg(short, long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Gaia DR3 `GaiaSource_*.csv.gz` files in a folder -> HEALPix bins.
    Ingest(catalog::IngestArgs),
    /// Bins -> the four density tiers `tier_500` ... `tier_8000`.
    BuildTiers(catalog::BuildTiersArgs),
    /// Bins -> one `stars.smac`.
    BuildCache(catalog::BuildCacheArgs),
    /// A smaller bright-star cache derived from a deep one.
    BuildBrightCache(catalog::BuildBrightCacheArgs),
    /// Tiers -> a quad-index pack (`quads.smaq`).
    BuildIndex(catalog::BuildIndexArgs),
    /// Tiers and index packs -> the publish tree (zips, checksums, manifests).
    Package(catalog::PackageArgs),
    /// Solve an image.
    Solve(solve::SolveArgs),
    /// What a folder holds: a cache, tiers, an index pack, bins or a publish tree.
    #[command(alias = "cache-info")]
    Info(info::InfoArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(e) = log::init(cli.log.as_deref(), cli.quiet) {
        eprintln!("error: {e:#}");
        return ExitCode::from(1);
    }
    let result = match cli.command {
        Command::Ingest(a) => catalog::ingest(a),
        Command::BuildTiers(a) => catalog::build_tiers(a),
        Command::BuildCache(a) => catalog::build_cache(a),
        Command::BuildBrightCache(a) => catalog::build_bright_cache(a),
        Command::BuildIndex(a) => catalog::build_index(a),
        Command::Package(a) => catalog::package(a),
        Command::Solve(a) => solve::run(a),
        Command::Info(a) => info::run(a),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            match e.downcast_ref::<solve::Refused>() {
                // A refusal is the gate's answer, not a failure of the run: the
                // reason stands alone.
                Some(r) => eprintln!("{r}"),
                None => eprintln!("error: {e:#}"),
            }
            ExitCode::from(1)
        }
    }
}
