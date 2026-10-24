//! Synthetic star fields for plate-solve testing.
//!
//! Renders 16-bit FITS frames from the density-tier catalog at random sky
//! positions, for focal lengths from wide lenses to long-focus telescopes.
//! Each frame carries the headers a capture program writes (pointing with a
//! deliberate error, FOCALLEN, XPIXSZ, binning, DATE-OBS) plus the true
//! solution in `SYN*` keywords, and `manifest.json` lists them all.
//!
//! ```text
//! cargo run --release --example synth_fields -- \
//!     --tiers <dir with tier_*> --out <dir> \
//!     [--per-focal 3] [--seed 1] [--sensor apsc|ff|m43|small|dslr] \
//!     [--focals 35,50,85,135,200,300,400,600,800,1000,1500,2000,3000,4000]
//! ```
//!
//! Realism knobs, all on by default: catalog magnitudes jittered (the sensor
//! band is not Gaia G), a few percent of catalog stars dropped and extra
//! stars the catalog lacks added, barrel distortion on short lenses, hot
//! pixels, saturation, and a mirrored frame now and then. The frames are
//! still drawn from the same catalog the solver uses, so they test the search
//! and the fits, not catalog errors.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use solvemyastro::{LayeredCones, StarCache, WcsSolution};

const DEFAULT_FOCALS: &[f64] =
    &[35.0, 50.0, 85.0, 135.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0];

/// Sensor preset: pixel count and pixel size (µm).
struct Sensor {
    name: &'static str,
    width: usize,
    height: usize,
    pixel_um: f64,
}

const SENSORS: &[Sensor] = &[
    Sensor { name: "apsc", width: 6248, height: 4176, pixel_um: 3.76 }, // IMX571
    Sensor { name: "ff", width: 9576, height: 6388, pixel_um: 3.76 },   // IMX455
    Sensor { name: "m43", width: 4144, height: 2822, pixel_um: 4.63 },  // IMX294
    Sensor { name: "small", width: 3008, height: 3008, pixel_um: 3.76 }, // IMX533
    Sensor { name: "dslr", width: 6000, height: 4000, pixel_um: 5.93 }, // 24 MP full frame
];

/// Background level and its noise, ADU.
const SKY_ADU: f64 = 1000.0;
const NOISE_ADU: f64 = 20.0;
/// Aperture SNR of a star at the frame's limiting magnitude.
const SNR_AT_LIMIT: f64 = 5.0;
/// Finest scale a frame is left at; finer setups are binned (as people do).
const MIN_SCALE_ARCSEC: f64 = 0.4;

struct Args {
    tiers: PathBuf,
    out: PathBuf,
    per_focal: usize,
    seed: u64,
    sensor: &'static Sensor,
    focals: Vec<f64>,
}

fn parse_args() -> Args {
    let mut tiers = None;
    let mut out = None;
    let mut per_focal = 3;
    let mut seed = 1;
    let mut sensor = &SENSORS[0];
    let mut focals = DEFAULT_FOCALS.to_vec();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || it.next().unwrap_or_else(|| usage(&format!("{a} needs a value")));
        match a.as_str() {
            "--tiers" => tiers = Some(PathBuf::from(val())),
            "--out" => out = Some(PathBuf::from(val())),
            "--per-focal" => per_focal = val().parse().unwrap_or_else(|_| usage("bad --per-focal")),
            "--seed" => seed = val().parse().unwrap_or_else(|_| usage("bad --seed")),
            "--sensor" => {
                let v = val();
                sensor = SENSORS
                    .iter()
                    .find(|s| s.name == v)
                    .unwrap_or_else(|| usage(&format!("unknown sensor {v}")));
            }
            "--focals" => {
                focals = val()
                    .split(',')
                    .map(|s| s.trim().parse().unwrap_or_else(|_| usage("bad --focals")))
                    .collect()
            }
            _ => usage(&format!("unknown argument {a}")),
        }
    }
    Args {
        tiers: tiers.unwrap_or_else(|| usage("--tiers is required")),
        out: out.unwrap_or_else(|| usage("--out is required")),
        per_focal,
        seed,
        sensor,
        focals,
    }
}

fn usage(msg: &str) -> ! {
    eprintln!("{msg}\nusage: synth_fields --tiers <dir> --out <dir> [--per-focal N] [--seed S] [--sensor apsc|ff|m43|small|dslr] [--focals 35,50,...]");
    std::process::exit(2)
}

/// Optics for a focal length: aperture (mm) and limiting magnitude.
fn optics(focal_mm: f64) -> (f64, f64) {
    let f_ratio = if focal_mm <= 200.0 {
        2.8
    } else if focal_mm <= 1000.0 {
        6.0
    } else {
        8.0
    };
    // Rough depth of a few-minute sub: ~10.5 at 35 mm, ~18.7 at 4000 mm.
    let mlim = (10.5 + 4.0 * (focal_mm / 35.0).log10()).clamp(10.0, 19.0);
    (focal_mm / f_ratio, mlim)
}

/// Standard normal deviate (Box–Muller).
fn gauss(rng: &mut Xoshiro256StarStar) -> f64 {
    let u1: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Uniform point on the sphere, (ra, dec) degrees.
fn random_sky(rng: &mut Xoshiro256StarStar) -> (f64, f64) {
    let ra: f64 = rng.random_range(0.0..360.0);
    let z: f64 = rng.random_range(-1.0..1.0);
    (ra, z.asin().to_degrees())
}

/// Move `(ra, dec)` by `dist` degrees towards position angle `pa` degrees.
fn offset_sky(ra: f64, dec: f64, dist: f64, pa: f64) -> (f64, f64) {
    let (d, p) = (dist.to_radians(), pa.to_radians());
    let (dr, rr) = (dec.to_radians(), ra.to_radians());
    let dec2 = (dr.sin() * d.cos() + dr.cos() * d.sin() * p.cos()).asin();
    let ra2 = rr + (p.sin() * d.sin() * dr.cos()).atan2(d.cos() - dr.sin() * dec2.sin());
    (ra2.to_degrees().rem_euclid(360.0), dec2.to_degrees())
}

fn ra_hms(ra: f64) -> String {
    let h = ra / 15.0;
    let (hh, mm) = (h.floor(), (h.fract() * 60.0).floor());
    let ss = ((h * 60.0).fract() * 60.0).min(59.999);
    format!("{:02} {:02} {:06.3}", hh as u32, mm as u32, ss)
}

fn dec_dms(dec: f64) -> String {
    let a = dec.abs();
    let (dd, mm) = (a.floor(), (a.fract() * 60.0).floor());
    let ss = ((a * 60.0).fract() * 60.0).min(59.99);
    format!("{}{:02} {:02} {:05.2}", if dec < 0.0 { '-' } else { '+' }, dd as u32, mm as u32, ss)
}

/// One rendered field's truth, as written to `manifest.json`.
struct Truth {
    file: String,
    focal_mm: f64,
    ra: f64,
    dec: f64,
    scale: f64,
    rot_deg: f64,
    flipped: bool,
    mlim: f64,
    n_stars: usize,
    hint_ra: f64,
    hint_dec: f64,
}

/// Render one frame and write it; returns its truth record.
#[allow(clippy::too_many_arguments)]
fn render(
    rng: &mut Xoshiro256StarStar,
    layers: &[&StarCache],
    sensor: &Sensor,
    focal_mm: f64,
    idx: usize,
    out: &Path,
) -> Truth {
    let (_aperture, mlim0) = optics(focal_mm);
    let mlim = mlim0 + rng.random_range(-0.5..0.5);
    let native_scale = 206.264_806 * sensor.pixel_um / focal_mm;
    let bin = ((MIN_SCALE_ARCSEC / native_scale).ceil() as usize).clamp(1, 4);
    let (w, h) = (sensor.width / bin, sensor.height / bin);
    let scale = native_scale * bin as f64;

    let (ra, dec) = random_sky(rng);
    let rot_deg: f64 = rng.random_range(0.0..360.0);
    let flipped = rng.random_bool(0.15);
    let epoch = rng.random_range(2023.0..2026.5);

    // CD: north up / east left at rot 0, rotated by `rot_deg`; a flip mirrors x.
    let s = scale / 3600.0;
    let (sr, cr) = rot_deg.to_radians().sin_cos();
    let px = if flipped { 1.0 } else { -1.0 };
    let wcs = WcsSolution {
        crpix: ((w as f64 - 1.0) * 0.5, (h as f64 - 1.0) * 0.5),
        crval: (ra, dec),
        cd: [[px * s * cr, s * sr], [-px * s * sr, s * cr]],
        sip_forward: None,
        sip_reverse: None,
    };

    // Barrel distortion on short lenses, relative to the half-diagonal.
    let half_diag = ((w * w + h * h) as f64).sqrt() * 0.5;
    let k1 = if focal_mm <= 135.0 { -0.015 * (50.0 / focal_mm).min(1.5) } else { 0.0 };

    let fwhm_arcsec = rng.random_range(2.0..4.0);
    let fwhm_px = (fwhm_arcsec / scale).max(rng.random_range(1.6..2.4));
    let sigma = fwhm_px / 2.354_820_045;
    let aperture_px = std::f64::consts::PI * (1.5 * fwhm_px).powi(2);
    let flux_lim = SNR_AT_LIMIT * NOISE_ADU * aperture_px.sqrt();
    let zp = mlim + 2.5 * flux_lim.log10();

    // Catalog stars in the frame.
    let radius = (half_diag * scale / 3600.0 * 1.05).min(80.0);
    let cones = LayeredCones::new(layers, epoch);
    let cat = cones
        .cone(layers.len(), ra, dec, radius, (mlim + 0.5) as f32, epoch)
        .expect("catalog cone");

    let mut stars: Vec<(f64, f64, f64)> = Vec::with_capacity(cat.len());
    for c in &cat {
        if rng.random_bool(0.03) {
            continue; // variable / missing in this band
        }
        let (mut x, mut y) = wcs.sky_to_pixel(c.ra, c.dec);
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        if k1 != 0.0 {
            let (dx, dy) = (x - wcs.crpix.0, y - wcs.crpix.1);
            let r2 = (dx * dx + dy * dy) / (half_diag * half_diag);
            x = wcs.crpix.0 + dx * (1.0 + k1 * r2);
            y = wcs.crpix.1 + dy * (1.0 + k1 * r2);
        }
        if x < -10.0 || y < -10.0 || x > w as f64 + 10.0 || y > h as f64 + 10.0 {
            continue;
        }
        let m = c.mag as f64 + 0.15 * gauss(rng);
        stars.push((x, y, 10f64.powf(-0.4 * (m - zp))));
    }
    // Stars the catalog does not have (fainter tail, other band, artefacts).
    let n_extra = stars.len() / 20;
    for _ in 0..n_extra {
        let m = rng.random_range(mlim - 3.0..mlim);
        stars.push((rng.random_range(0.0..w as f64), rng.random_range(0.0..h as f64), 10f64.powf(-0.4 * (m - zp))));
    }

    // Background + noise, then stars, then hot pixels.
    let mut img: Vec<f32> = (0..w * h).map(|_| (SKY_ADU + NOISE_ADU * gauss(rng)) as f32).collect();
    let norm = 1.0 / (std::f64::consts::TAU * sigma * sigma);
    for &(x, y, flux) in &stars {
        let peak = flux * norm;
        // Stamp out to where the profile drops below a tenth of the noise.
        let r = (sigma * (2.0 * (peak / (0.1 * NOISE_ADU)).max(1.0).ln()).sqrt()).clamp(3.0 * sigma, 60.0 * sigma);
        let (x0, x1) = ((x - r).floor().max(0.0) as usize, ((x + r).ceil() as usize).min(w - 1));
        let (y0, y1) = ((y - r).floor().max(0.0) as usize, ((y + r).ceil() as usize).min(h - 1));
        if x0 > x1 || y0 > y1 {
            continue;
        }
        let inv = 1.0 / (2.0 * sigma * sigma);
        for yy in y0..=y1 {
            let dy2 = (yy as f64 - y).powi(2);
            let row = &mut img[yy * w..(yy + 1) * w];
            for xx in x0..=x1 {
                let d2 = (xx as f64 - x).powi(2) + dy2;
                row[xx] += (peak * (-d2 * inv).exp()) as f32;
            }
        }
    }
    for _ in 0..(w * h / 200_000) {
        let i = rng.random_range(0..w * h);
        img[i] = rng.random_range(20_000.0..65_535.0);
    }

    let (hint_ra, hint_dec) = offset_sky(ra, dec, rng.random_range(0.0..1.0), rng.random_range(0.0..360.0));
    let file = format!("synth_{:04}mm_{:03}.fits", focal_mm as u32, idx);
    let y = epoch.floor() as i32;
    let doy = ((epoch - y as f64) * 365.0) as u32;
    let (mon, day) = (1 + (doy / 31).min(11), 1 + doy % 28);
    let mut hdr: Vec<(String, String)> = vec![
        ("SIMPLE".into(), "T".into()),
        ("BITPIX".into(), "16".into()),
        ("NAXIS".into(), "2".into()),
        ("NAXIS1".into(), w.to_string()),
        ("NAXIS2".into(), h.to_string()),
        ("BZERO".into(), "32768".into()),
        ("BSCALE".into(), "1".into()),
        ("OBJCTRA".into(), format!("'{}'", ra_hms(hint_ra))),
        ("OBJCTDEC".into(), format!("'{}'", dec_dms(hint_dec))),
        ("FOCALLEN".into(), format!("{focal_mm:.1}")),
        ("XPIXSZ".into(), format!("{:.2}", sensor.pixel_um)),
        ("YPIXSZ".into(), format!("{:.2}", sensor.pixel_um)),
        ("XBINNING".into(), bin.to_string()),
        ("YBINNING".into(), bin.to_string()),
        ("EXPTIME".into(), "120.0".into()),
        ("DATE-OBS".into(), format!("'{y:04}-{mon:02}-{day:02}T22:00:00'")),
        ("SYNRA".into(), format!("{ra:.8}")),
        ("SYNDEC".into(), format!("{dec:.8}")),
        ("SYNSCALE".into(), format!("{scale:.6}")),
        ("SYNROT".into(), format!("{rot_deg:.4}")),
        ("SYNFLIP".into(), if flipped { "T" } else { "F" }.into()),
        ("SYNMLIM".into(), format!("{mlim:.2}")),
        ("SYNK1".into(), format!("{k1:.4}")),
        ("SYNFWHM".into(), format!("{fwhm_px:.2}")),
    ];
    hdr.push(("SYNSTARS".into(), stars.len().to_string()));
    write_fits(&out.join(&file), &hdr, &img);

    Truth {
        file,
        focal_mm,
        ra,
        dec,
        scale,
        rot_deg,
        flipped,
        mlim,
        n_stars: stars.len(),
        hint_ra,
        hint_dec,
    }
}

/// Write a 16-bit unsigned FITS (BZERO 32768) with the given header cards.
fn write_fits(path: &Path, hdr: &[(String, String)], img: &[f32]) {
    let mut head = String::new();
    for (k, v) in hdr {
        // Strings start at column 11; everything else is right-aligned to 30.
        let card = if v.starts_with('\'') { format!("{k:<8}= {v}") } else { format!("{k:<8}= {v:>20}") };
        let _ = write!(head, "{card:<80}");
    }
    let _ = write!(head, "{:<80}", "END");
    while head.len() % 2880 != 0 {
        head.push(' ');
    }
    let mut bytes = head.into_bytes();
    bytes.reserve(img.len() * 2 + 2880);
    for &v in img {
        let u = v.round().clamp(0.0, 65_535.0) as i32;
        bytes.extend_from_slice(&((u - 32_768) as i16).to_be_bytes());
    }
    while bytes.len() % 2880 != 0 {
        bytes.push(0);
    }
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&bytes))
        .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all(&args.out).expect("create --out");
    let caches: Vec<StarCache> = ["tier_500", "tier_2000", "tier_5000", "tier_8000"]
        .iter()
        .map(|t| args.tiers.join(t))
        .filter(|p| p.is_dir())
        .map(|p| StarCache::open(&p).unwrap_or_else(|e| panic!("open {}: {e}", p.display())))
        .collect();
    assert!(!caches.is_empty(), "no tier_* caches under {}", args.tiers.display());
    let layers: Vec<&StarCache> = caches.iter().collect();

    let mut rng = Xoshiro256StarStar::seed_from_u64(args.seed);
    let mut manifest = String::from("[\n");
    let mut first = true;
    for &f in &args.focals {
        for i in 0..args.per_focal {
            let t = render(&mut rng, &layers, args.sensor, f, i, &args.out);
            println!(
                "{:<26} ra={:8.3} dec={:+7.3} {:6.3}\"/px rot={:5.1} flip={} mlim={:4.1} stars={}",
                t.file, t.ra, t.dec, t.scale, t.rot_deg, t.flipped as u8, t.mlim, t.n_stars
            );
            if !first {
                manifest.push_str(",\n");
            }
            first = false;
            let _ = write!(
                manifest,
                "  {{\"file\": \"{}\", \"focal_mm\": {}, \"sensor\": \"{}\", \"ra\": {:.8}, \"dec\": {:.8}, \"scale_arcsec\": {:.6}, \"rot_deg\": {:.4}, \"flipped\": {}, \"mlim\": {:.2}, \"n_stars\": {}, \"hint_ra\": {:.6}, \"hint_dec\": {:.6}}}",
                t.file, t.focal_mm, args.sensor.name, t.ra, t.dec, t.scale, t.rot_deg, t.flipped, t.mlim, t.n_stars, t.hint_ra, t.hint_dec
            );
        }
    }
    manifest.push_str("\n]\n");
    std::fs::write(args.out.join("manifest.json"), manifest).expect("write manifest");
}
