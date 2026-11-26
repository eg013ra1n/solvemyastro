//! `solve`: one image against a catalog, with the acceptance gate.

use std::fmt;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::Serialize;

use solvemyastro::accept::{gate_tol_px, Gate, GateMetrics, GateRefusal};
use solvemyastro::layout::discover_layers;
use solvemyastro::{solve_with_index, Caches, QuadIndex, SolveConfig, SolveHints, StarCache};

#[derive(Args)]
pub struct SolveArgs {
    /// The FITS or XISF frame.
    image: PathBuf,
    /// A folder of density tiers (`tier_*`).
    #[arg(long, value_name = "DIR", conflicts_with = "catalog")]
    tiers: Option<PathBuf>,
    /// A single star cache folder.
    #[arg(long, value_name = "DIR")]
    catalog: Option<PathBuf>,
    /// A quad-index pack; repeat for several.
    #[arg(long, value_name = "DIR")]
    index: Vec<PathBuf>,
    /// Right ascension of the field centre, degrees.
    #[arg(long, allow_negative_numbers = true, requires = "dec")]
    ra: Option<f64>,
    /// Declination of the field centre, degrees.
    #[arg(long, allow_negative_numbers = true, requires = "ra")]
    dec: Option<f64>,
    /// Field of view, degrees.
    #[arg(long, conflicts_with = "scale")]
    fov: Option<f64>,
    /// Pixel scale, arcseconds per pixel. Also the scale the gate holds the
    /// solution to.
    #[arg(long)]
    scale: Option<f64>,
    /// Search radius around the position, degrees.
    #[arg(long)]
    radius: Option<f64>,
    /// Julian year of the observation, to bring catalog positions to (proper motion). Without it the
    /// solver assumes 2024.04.
    #[arg(long)]
    epoch: Option<f64>,
    /// Print the solution as one JSON object.
    #[arg(long)]
    json: bool,
    /// Print a solution the acceptance gate would refuse.
    #[arg(long)]
    no_gate: bool,
}

/// The gate refused the solution. The reason is the whole message.
#[derive(Debug)]
pub struct Refused(String);

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

/// The `--json` object.
#[derive(Serialize)]
struct JsonOut {
    ra: f64,
    dec: f64,
    scale: f64,
    rot: f64,
    matched: usize,
    rms_px: f64,
    inlier_ratio: f64,
    log10_odds: f64,
    reference_stars: usize,
    hypotheses: usize,
    gate: &'static str,
}

/// What a refusal says: the check that failed, with the numbers it judged.
fn refusal_message(r: &GateRefusal) -> String {
    format!("rejected: {r}")
}

/// The layers `--tiers` or `--catalog` name.
fn open_layers(a: &SolveArgs) -> Result<Vec<StarCache>> {
    if let Some(dir) = &a.tiers {
        let layers = discover_layers(dir);
        if layers.is_empty() {
            bail!("no tier_* caches under {}", dir.display());
        }
        return layers.iter().map(|d| StarCache::open(d)).collect();
    }
    match &a.catalog {
        Some(dir) => Ok(vec![StarCache::open(dir)?]),
        None => bail!("--catalog <cache-dir> or --tiers <dir> is required"),
    }
}

pub fn run(a: SolveArgs) -> Result<()> {
    let layers = open_layers(&a)?;
    let refs: Vec<&StarCache> = layers.iter().collect();
    let caches = if refs.len() == 1 { Caches::deep_only(refs[0]) } else { Caches::layered(&refs) };
    let indexes = a
        .index
        .iter()
        .map(|d| QuadIndex::open(d).with_context(|| format!("open index pack {}", d.display())))
        .collect::<Result<Vec<_>>>()?;
    let index_refs: Vec<&QuadIndex> = indexes.iter().collect();
    let hints = SolveHints {
        ra: a.ra,
        dec: a.dec,
        fov_deg: a.fov,
        pixel_scale_arcsec: a.scale,
        search_radius_deg: a.radius,
        epoch: a.epoch,
    };
    let sol = solve_with_index(&a.image, &hints, &caches, &index_refs, &SolveConfig::default(), None)?;

    let gate = Gate::default();
    let metrics = GateMetrics {
        inliers: sol.matched_stars,
        rms_px: sol.rms_residual_px,
        tol_px: gate_tol_px(sol.pixel_scale_arcsec, gate.base_tolerance_arcsec),
        log10_odds: sol.log10_odds,
        recovered_scale_arcsec: sol.pixel_scale_arcsec,
        header_scale_arcsec: a.scale,
    };
    let refusal = gate.refusal(&metrics);
    if let Some(r) = &refusal {
        if !a.no_gate {
            return Err(Refused(refusal_message(r)).into());
        }
        tracing::warn!(
            inliers = sol.matched_stars,
            log10_odds = sol.log10_odds,
            scale = sol.pixel_scale_arcsec,
            refusal = %r,
            "the gate would refuse this solution; printed because of --no-gate"
        );
    }
    let gate_word = if a.no_gate { "off" } else { "passed" };
    if a.json {
        let out = JsonOut {
            ra: sol.wcs.crval.0,
            dec: sol.wcs.crval.1,
            scale: sol.pixel_scale_arcsec,
            rot: sol.field_rotation_deg,
            matched: sol.matched_stars,
            rms_px: sol.rms_residual_px,
            inlier_ratio: sol.inlier_ratio,
            log10_odds: sol.log10_odds,
            reference_stars: sol.reference_stars_in_frame,
            hypotheses: sol.hypotheses_verified,
            gate: gate_word,
        };
        println!("{}", serde_json::to_string(&out).context("encode the solution")?);
    } else {
        println!("{sol:#?}");
        println!(
            "odds: 10^{:.1} to 1 ({} reference stars, {} hypotheses)",
            sol.log10_odds, sol.reference_stars_in_frame, sol.hypotheses_verified
        );
        println!("gate: {gate_word}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_keys_are_the_documented_set() {
        let out = JsonOut {
            ra: 1.0,
            dec: -2.0,
            scale: 1.5,
            rot: 10.0,
            matched: 40,
            rms_px: 0.3,
            inlier_ratio: 0.9,
            log10_odds: 31.5,
            reference_stars: 60,
            hypotheses: 12,
            gate: "passed",
        };
        let v = serde_json::to_value(&out).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        let mut want = [
            "ra", "dec", "scale", "rot", "matched", "rms_px", "inlier_ratio", "log10_odds",
            "reference_stars", "hypotheses", "gate",
        ];
        want.sort_unstable();
        assert_eq!(keys, want);
        assert_eq!(keys.len(), 11);
        assert_eq!(v["gate"], "passed");
    }

    #[test]
    fn a_refusal_reads_rejected_and_names_the_check() {
        let msg = refusal_message(&GateRefusal::TooFewMatched { matched: 5, needed: 6 });
        assert_eq!(msg, "rejected: only 5 stars matched (6 needed)");
        assert_eq!(Refused(msg.clone()).to_string(), msg);
    }
}
