//! Precomputed quad index — the blind-search path.
//!
//! The spiral search builds catalog quads on the fly for every sky cell and
//! FOV rung, so a blind solve costs ~1/FOV² cells. This index holds the
//! catalog's quads once, keyed by a 4-D shape code, so an image quad is
//! looked up directly and a single match yields a full WCS hypothesis.
//!
//! # Build
//!
//! - **Bands.** One band per HEALPix depth `d`: quads whose AB (most distant
//!   pair) is in `[side_d/√2, side_d·√2)`, `side_d` the depth's mean pixel side.
//! - **Uniform stars.** For band `d`, the sky is cut into depth `d+1` cells
//!   (half the band's AB) and the [`BuildParams::stars_per_cell`] brightest
//!   stars of each cell are kept, so index stars spread over the sky the way
//!   a frame's brightest detections do.
//! - **Quads.** For each cell, pairs A–B from the cell and its neighbours,
//!   brightest first, with AB in the band and AB's midpoint in the cell (so
//!   every quad has one home cell); C and D are the two brightest stars inside
//!   the circle on AB. At most [`BuildParams::quads_per_cell`] per cell.
//! - **Code.** In the tangent plane at AB's midpoint, A ↦ (0,0), B ↦ (1,1);
//!   the code is (Cx, Cy, Dx, Dy), made canonical by swapping A/B when
//!   Cx + Dx > 1 and C/D when Cx > Dx. Mirror images are not stored: a frame's
//!   code is looked up twice, plain and reflected.
//! - **Tree.** Codes are stored as an implicit kd-tree (array in tree order,
//!   split axis = level mod 4), so the file is searched as mapped.
//!
//! # File (`quads.smaq`, little-endian)
//!
//! | Part | Layout |
//! |------|--------|
//! | header (64 B) | magic `SMAQIDX1`, `u32` version, `u32` n_bands, 8 B 0, `f64` catalog epoch |
//! | band table | per band 64 B: `u32` depth `d`, `u32` cell depth `u = d+1`, `f64` ab_lo, `f64` ab_hi (rad), `u64` n_quads, `u64` records / cell directory / stars offsets, `u64` n_stars |
//! | records (16 B) | `n_quads` × (4 `u16` code ([`CODE_MIN`] + q·[`CODE_STEP`]), `u32` home cell, 4 `u8` star refs A, B, C, D), kd-tree order |
//! | cell directory | `n_cells(u) + 1` × `u32`: cell `c`'s stars are `dir[c]..dir[c+1]` |
//! | stars (4 B) | 2 `i16` offsets (ξ, η) from the cell centre in [`star_unit`] steps — only stars some quad uses |
//!
//! A star reference counts through the stars of the home cell and its
//! neighbours, in ascending cell order.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use cdshealpix::nested;
use memmap2::Mmap;

use crate::cache::StarCache;
use crate::wcs::{tan, WcsSolution};

const MAGIC: &[u8; 8] = b"SMAQIDX1";
const VERSION: u32 = 3;
const HEADER_SIZE: usize = 64;
const BAND_ENTRY_SIZE: usize = 64;
const RECORD_SIZE: usize = 16;
const STAR_SIZE: usize = 4;
/// File name inside an index directory.
pub const INDEX_FILE: &str = "quads.smaq";
/// Codes lie in the circle on AB — x, y ∈ [0.5 − √½, 0.5 + √½] ⊂ [−0.25, 1.25] —
/// and are stored as `u16` steps of [`CODE_STEP`] (2.3e-5, far below any
/// lookup tolerance).
const CODE_MIN: f64 = -0.25;
const CODE_STEP: f64 = 1.5 / 65535.0;

fn quantize(c: f64) -> u16 {
    ((c - CODE_MIN) / CODE_STEP).round().clamp(0.0, 65535.0) as u16
}

/// kd-tree leaf size: ranges this small are scanned linearly.
const LEAF: usize = 16;
/// Depth of the `.smac` star caches the index is built from.
const SMAC_DEPTH: u8 = 6;

// ---------------------------------------------------------------------------
// Quad code
// ---------------------------------------------------------------------------

/// Canonical code of 4 points in a plane, and the order of the points as
/// (A, B, C, D). `None` when the shape is degenerate or C/D lie outside the
/// circle on AB (both sides of the index apply the same rule).
pub(crate) fn quad_code(p: [(f64, f64); 4]) -> Option<([f64; 4], [usize; 4])> {
    // AB = the most distant pair.
    let mut best = (0usize, 1usize, -1.0f64);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let d2 = (p[i].0 - p[j].0).powi(2) + (p[i].1 - p[j].1).powi(2);
            if d2 > best.2 {
                best = (i, j, d2);
            }
        }
    }
    let (ia, ib, ab2) = best;
    if ab2 < 1e-24 {
        return None;
    }
    let mut rest = (0..4).filter(|&k| k != ia && k != ib);
    let (ic, id) = (rest.next()?, rest.next()?);

    // z' = (z − A)·(1+i)/(B − A): A ↦ 0, B ↦ 1+i.
    let (ax, ay) = p[ia];
    let (bx, by) = (p[ib].0 - ax, p[ib].1 - ay);
    let map = |(x, y): (f64, f64)| {
        let (zx, zy) = (x - ax, y - ay);
        // (z / (B−A)) then × (1+i)
        let (qx, qy) = ((zx * bx + zy * by) / ab2, (zy * bx - zx * by) / ab2);
        (qx - qy, qx + qy)
    };
    let (mut c, mut d) = (map(p[ic]), map(p[id]));
    let in_circle = |(x, y): (f64, f64)| (x - 0.5).powi(2) + (y - 0.5).powi(2) <= 0.5;
    if !in_circle(c) || !in_circle(d) {
        return None;
    }
    let (mut a_i, mut b_i, mut c_i, mut d_i) = (ia, ib, ic, id);
    if c.0 + d.0 > 1.0 {
        std::mem::swap(&mut a_i, &mut b_i);
        c = (1.0 - c.0, 1.0 - c.1);
        d = (1.0 - d.0, 1.0 - d.1);
    }
    if c.0 > d.0 {
        std::mem::swap(&mut c, &mut d);
        std::mem::swap(&mut c_i, &mut d_i);
    }
    Some(([c.0, c.1, d.0, d.1], [a_i, b_i, c_i, d_i]))
}

// ---------------------------------------------------------------------------
// Sphere helpers
// ---------------------------------------------------------------------------

fn unit(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let (sr, cr) = ra_deg.to_radians().sin_cos();
    let (sd, cd) = dec_deg.to_radians().sin_cos();
    [cd * cr, cd * sr, sd]
}

fn radec(v: [f64; 3]) -> (f64, f64) {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let ra = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
    (ra, (v[2] / n).clamp(-1.0, 1.0).asin().to_degrees())
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Angle between unit vectors, radians (stable for small angles).
fn angle(a: [f64; 3], b: [f64; 3]) -> f64 {
    let c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().atan2(dot(a, b))
}

fn midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let m = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
    let n = dot(m, m).sqrt();
    [m[0] / n, m[1] / n, m[2] / n]
}

/// Mean HEALPix pixel side at `depth`, radians.
fn side(depth: u8) -> f64 {
    (4.0 * std::f64::consts::PI / nested::n_hash(depth) as f64).sqrt()
}

fn cell_of(depth: u8, v: [f64; 3]) -> u64 {
    let (ra, dec) = radec(v);
    nested::hash(depth, ra.to_radians(), dec.to_radians())
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/// Index build parameters.
#[derive(Clone, Copy, Debug)]
pub struct BuildParams {
    /// Coarsest band depth (largest quads). Depth 2 ≈ 10–21° AB.
    pub min_depth: u8,
    /// Finest band depth (smallest quads). Depth 7 ≈ 0.32–0.65° AB, which
    /// serves fields down to ~1.1°.
    pub max_depth: u8,
    /// Brightest stars kept per depth `d+1` cell (at most 28: a quad's star
    /// is a one-byte reference into 9 cells' lists).
    pub stars_per_cell: usize,
    /// Quads kept per depth `d+1` cell.
    pub quads_per_cell: usize,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self { min_depth: 2, max_depth: 7, stars_per_cell: 10, quads_per_cell: 8 }
    }
}

/// A uniform-selection star: magnitude and unit vector.
type CellStar = (f32, [f64; 3]);

/// One stored quad: canonical code, home cell, and its stars as
/// `(cell, index in that cell's list)` pairs, A, B, C, D.
struct BuiltQuad {
    code: [u16; 4],
    cell: u32,
    stars: [(u32, u16); 4],
}

/// One band, ready to write.
struct BuiltBand {
    depth: u8,
    u: u8,
    ab_lo: f64,
    ab_hi: f64,
    /// Records in kd-tree order: code, home cell, one-byte star references.
    records: Vec<([u16; 4], u32, [u8; 4])>,
    /// `dir[c]..dir[c + 1]` are cell `c`'s stars in `stars`.
    dir: Vec<u32>,
    /// Star offsets from their cell centre, in [`star_unit`] steps.
    stars: Vec<[i16; 2]>,
}

/// Build an index from `layers` (bright → faint additive tiers) into
/// `out_dir/quads.smaq`. Returns `(n_stars, n_quads)`.
pub fn build_index(
    layers: &[&StarCache],
    out_dir: &Path,
    params: &BuildParams,
    progress: impl Fn(&str),
) -> Result<(usize, usize)> {
    ensure!(!layers.is_empty(), "no star layers");
    ensure!(params.min_depth <= params.max_depth, "min_depth > max_depth");
    ensure!(
        (2..=28).contains(&params.stars_per_cell),
        "stars_per_cell must be 2..=28 (one-byte references into 9 cells)"
    );
    let epoch = layers[0].catalog_epoch();
    let mut bands = Vec::new();
    for d in params.min_depth..=params.max_depth {
        let u = d + 1;
        let cells = uniform_stars(layers, u, params.stars_per_cell)?;
        let quads = build_quads(&cells, d, u, params.quads_per_cell);
        let band = pack_band(&cells, quads, d, u);
        progress(&format!(
            "band depth {d}: AB {:.3}–{:.3}°, {} quads, {} stars",
            band.ab_lo.to_degrees(),
            band.ab_hi.to_degrees(),
            band.records.len(),
            band.stars.len()
        ));
        bands.push(band);
    }
    std::fs::create_dir_all(out_dir)?;
    write_index(&out_dir.join(INDEX_FILE), epoch, &bands)?;
    Ok((bands.iter().map(|b| b.stars.len()).sum(), bands.iter().map(|b| b.records.len()).sum()))
}

/// The `k` brightest stars of every depth-`u` cell, in magnitude order.
/// Layers are filled in order, a fainter layer only where a cell is still
/// short.
fn uniform_stars(layers: &[&StarCache], u: u8, k: usize) -> Result<Vec<Vec<CellStar>>> {
    let n_cells = nested::n_hash(u) as usize;
    let mut best: Vec<Vec<CellStar>> = vec![Vec::new(); n_cells];
    for (li, layer) in layers.iter().enumerate() {
        for px in 0..nested::n_hash(SMAC_DEPTH) {
            // Skip a pixel whose cells are all full already.
            if li > 0 {
                let full = if u >= SMAC_DEPTH {
                    let shift = 2 * (u - SMAC_DEPTH) as u64;
                    ((px << shift)..((px + 1) << shift)).all(|c| best[c as usize].len() >= k)
                } else {
                    best[(px >> (2 * (SMAC_DEPTH - u) as u64)) as usize].len() >= k
                };
                if full {
                    continue;
                }
            }
            for r in layer.iter_pixel_records(px)? {
                let c = nested::hash(u, r.ra.to_radians(), r.dec.to_radians()) as usize;
                let list = &mut best[c];
                if list.len() >= k && r.mag >= list[k - 1].0 {
                    continue;
                }
                let pos = list.partition_point(|e| e.0 <= r.mag);
                list.insert(pos, (r.mag, unit(r.ra, r.dec)));
                list.truncate(k);
            }
        }
    }
    Ok(best)
}

/// Cell `c` and its neighbours, ascending — the order a quad's one-byte star
/// references count through.
fn neighbourhood(u: u8, c: u64) -> Vec<u64> {
    let mut v = nested::neighbours(u, c, true).values_vec();
    v.sort_unstable();
    v
}

/// Quads of band `d` from the uniform star lists of depth-`u` cells.
fn build_quads(cells: &[Vec<CellStar>], d: u8, u: u8, qcap: usize) -> Vec<BuiltQuad> {
    let s = side(d);
    let (ab_lo, ab_hi) = (s / std::f64::consts::SQRT_2, s * std::f64::consts::SQRT_2);
    let mut quads = Vec::new();
    // (mag, xyz, cell, index in cell)
    let mut nb: Vec<(f32, [f64; 3], u32, u16)> = Vec::new();
    for c in 0..cells.len() {
        nb.clear();
        for n in neighbourhood(u, c as u64) {
            for (i, &(m, v)) in cells[n as usize].iter().enumerate() {
                nb.push((m, v, n as u32, i as u16));
            }
        }
        nb.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut made = 0;
        'pairs: for j in 1..nb.len() {
            for i in 0..j {
                let ab = angle(nb[i].1, nb[j].1);
                if ab < ab_lo || ab >= ab_hi {
                    continue;
                }
                let m = midpoint(nb[i].1, nb[j].1);
                if cell_of(u, m) as usize != c {
                    continue;
                }
                // The two brightest other stars inside the circle on AB.
                let r = ab * 0.5;
                let mut cd = (0..nb.len()).filter(|&k| k != i && k != j && angle(nb[k].1, m) <= r);
                let (Some(kc), Some(kd)) = (cd.next(), cd.next()) else { continue };
                let (mra, mdec) = radec(m);
                let quad = [i, j, kc, kd];
                let pts = quad.map(|q| {
                    let (ra, dec) = radec(nb[q].1);
                    tan::sky_to_tangent(ra, dec, mra, mdec)
                });
                let Some((code, order)) = quad_code(pts) else { continue };
                quads.push(BuiltQuad {
                    code: code.map(quantize),
                    cell: c as u32,
                    stars: order.map(|o| (nb[quad[o]].2, nb[quad[o]].3)),
                });
                made += 1;
                if made >= qcap {
                    break 'pairs;
                }
            }
        }
    }
    quads
}

/// Star position step at depth `u`: offsets of ±2 cell sides fit an `i16`.
fn star_unit(u: u8) -> f64 {
    2.0 * side(u).to_degrees() / 32767.0
}

/// Keep the stars some quad uses, store them per cell as offsets from the
/// cell centre, turn each quad's stars into one-byte references into its
/// neighbourhood, and put the records in kd-tree order.
fn pack_band(cells: &[Vec<CellStar>], quads: Vec<BuiltQuad>, d: u8, u: u8) -> BuiltBand {
    // Renumber used stars per cell (u16::MAX = unused).
    let mut new_idx: Vec<Vec<u16>> = cells.iter().map(|l| vec![u16::MAX; l.len()]).collect();
    for q in &quads {
        for &(c, i) in &q.stars {
            new_idx[c as usize][i as usize] = 0;
        }
    }
    let mut dir = Vec::with_capacity(cells.len() + 1);
    let mut stars = Vec::new();
    let unit_deg = star_unit(u);
    for (c, list) in cells.iter().enumerate() {
        dir.push(stars.len() as u32);
        let (lon, lat) = nested::center(u, c as u64);
        let (ra0, dec0) = (lon.to_degrees(), lat.to_degrees());
        let mut n = 0u16;
        for (i, &(_, v)) in list.iter().enumerate() {
            if new_idx[c][i] == u16::MAX {
                continue;
            }
            new_idx[c][i] = n;
            n += 1;
            let (ra, dec) = radec(v);
            let (xi, eta) = tan::sky_to_tangent(ra, dec, ra0, dec0);
            stars.push([xi, eta].map(|t| (t / unit_deg).round().clamp(-32767.0, 32767.0) as i16));
        }
    }
    dir.push(stars.len() as u32);

    let mut records: Vec<([u16; 4], u32, [u8; 4])> = quads
        .iter()
        .map(|q| {
            let hood = neighbourhood(u, q.cell as u64);
            let refs = q.stars.map(|(c, i)| {
                let mut r = 0usize;
                for &n in &hood {
                    if n as u32 == c {
                        r += new_idx[c as usize][i as usize] as usize;
                        break;
                    }
                    r += (dir[n as usize + 1] - dir[n as usize]) as usize;
                }
                u8::try_from(r).expect("neighbourhood holds < 256 stars")
            });
            (q.code, q.cell, refs)
        })
        .collect();
    tree_order(&mut records);
    let s = side(d);
    BuiltBand {
        depth: d,
        u,
        ab_lo: s / std::f64::consts::SQRT_2,
        ab_hi: s * std::f64::consts::SQRT_2,
        records,
        dir,
        stars,
    }
}

/// Reorder records into implicit kd-tree order: the median of `[lo, hi)` on
/// code axis `level % 4` sits at `(lo + hi) / 2`.
fn tree_order<T: Copy>(records: &mut [([u16; 4], T, [u8; 4])]) {
    fn rec<T>(r: &mut [([u16; 4], T, [u8; 4])], level: usize) {
        if r.len() <= LEAF {
            return;
        }
        let axis = level % 4;
        let mid = r.len() / 2;
        r.select_nth_unstable_by_key(mid, |e| e.0[axis]);
        let (left, right) = r.split_at_mut(mid);
        rec(left, level + 1);
        rec(&mut right[1..], level + 1);
    }
    rec(records, 0);
}

fn write_index(path: &Path, epoch: f64, bands: &[BuiltBand]) -> Result<()> {
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    let mut header = [0u8; HEADER_SIZE];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(bands.len() as u32).to_le_bytes());
    header[24..32].copy_from_slice(&epoch.to_le_bytes());
    w.write_all(&header)?;

    let mut off = HEADER_SIZE + bands.len() * BAND_ENTRY_SIZE;
    for b in bands {
        let rec_off = off;
        let dir_off = rec_off + b.records.len() * RECORD_SIZE;
        let stars_off = dir_off + b.dir.len() * 4;
        off = stars_off + b.stars.len() * STAR_SIZE;
        w.write_all(&(b.depth as u32).to_le_bytes())?;
        w.write_all(&(b.u as u32).to_le_bytes())?;
        w.write_all(&b.ab_lo.to_le_bytes())?;
        w.write_all(&b.ab_hi.to_le_bytes())?;
        for x in [b.records.len(), rec_off, dir_off, stars_off, b.stars.len()] {
            w.write_all(&(x as u64).to_le_bytes())?;
        }
    }
    for b in bands {
        for (code, cell, refs) in &b.records {
            for x in code {
                w.write_all(&x.to_le_bytes())?;
            }
            w.write_all(&cell.to_le_bytes())?;
            w.write_all(refs)?;
        }
        for x in &b.dir {
            w.write_all(&x.to_le_bytes())?;
        }
        for s in &b.stars {
            w.write_all(&s[0].to_le_bytes())?;
            w.write_all(&s[1].to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// One scale band of an open index.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    /// HEALPix depth the band is named after.
    pub depth: u8,
    /// AB range, radians.
    pub ab_lo: f64,
    pub ab_hi: f64,
    /// Depth of the cells its stars are stored by.
    u: u8,
    n_quads: usize,
    rec_off: usize,
    dir_off: usize,
    stars_off: usize,
    n_stars: usize,
}

impl Band {
    /// Number of quads in the band.
    pub fn n_quads(&self) -> usize {
        self.n_quads
    }

    /// Number of stars the band's quads use.
    pub fn n_stars(&self) -> usize {
        self.n_stars
    }
}

/// A memory-mapped quad index.
pub struct QuadIndex {
    mmap: Mmap,
    bands: Vec<Band>,
}

fn u16_at(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

fn f64_at(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

impl QuadIndex {
    /// Open `dir/quads.smaq`.
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join(INDEX_FILE);
        let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: the file is opened read-only and never written while mapped
        // by this process; the index is a build artefact, not a live file.
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        let b = &mmap[..];
        ensure!(b.len() >= HEADER_SIZE && &b[..8] == MAGIC, "{} is not a quad index", path.display());
        let version = u32_at(b, 8);
        if version != VERSION {
            bail!("{}: index version {version}, expected {VERSION} — rebuild it", path.display());
        }
        let n_bands = u32_at(b, 12) as usize;
        ensure!(b.len() >= HEADER_SIZE + n_bands * BAND_ENTRY_SIZE, "{} is truncated", path.display());
        let mut bands = Vec::with_capacity(n_bands);
        for i in 0..n_bands {
            let o = HEADER_SIZE + i * BAND_ENTRY_SIZE;
            let band = Band {
                depth: u32_at(b, o) as u8,
                u: u32_at(b, o + 4) as u8,
                ab_lo: f64_at(b, o + 8),
                ab_hi: f64_at(b, o + 16),
                n_quads: u64_at(b, o + 24) as usize,
                rec_off: u64_at(b, o + 32) as usize,
                dir_off: u64_at(b, o + 40) as usize,
                stars_off: u64_at(b, o + 48) as usize,
                n_stars: u64_at(b, o + 56) as usize,
            };
            let n_cells = nested::n_hash(band.u) as usize;
            ensure!(
                band.rec_off + band.n_quads * RECORD_SIZE <= b.len()
                    && band.dir_off + (n_cells + 1) * 4 <= b.len()
                    && band.stars_off + band.n_stars * STAR_SIZE <= b.len(),
                "{}: band {i} points beyond the file",
                path.display()
            );
            bands.push(band);
        }
        Ok(Self { mmap, bands })
    }

    /// The index's scale bands, coarsest first.
    pub fn bands(&self) -> &[Band] {
        &self.bands
    }

    /// Smallest field of view (degrees) the index serves reliably: four times
    /// its smallest quad. Narrower fields need a finer pack or the spiral.
    pub fn min_fov_deg(&self) -> f64 {
        self.bands.iter().map(|b| b.ab_lo).fold(f64::INFINITY, f64::min).to_degrees() * 4.0
    }

    fn code(&self, band: &Band, q: usize) -> [f64; 4] {
        let o = band.rec_off + q * RECORD_SIZE;
        let b = &self.mmap[..];
        std::array::from_fn(|k| CODE_MIN + u16_at(b, o + 2 * k) as f64 * CODE_STEP)
    }

    /// Unit vectors of the stars (A, B, C, D) of quad `q` in `band`.
    pub fn quad_stars(&self, band: &Band, q: usize) -> [[f64; 3]; 4] {
        let b = &self.mmap[..];
        let o = band.rec_off + q * RECORD_SIZE;
        let cell = u32_at(b, o + 8) as u64;
        let hood = neighbourhood(band.u, cell);
        let first = |c: u64| u32_at(b, band.dir_off + c as usize * 4) as usize;
        let unit_deg = star_unit(band.u);
        std::array::from_fn(|k| {
            let mut r = b[o + 12 + k] as usize;
            for &c in &hood {
                let n = first(c + 1) - first(c);
                if r < n {
                    let so = band.stars_off + (first(c) + r) * STAR_SIZE;
                    let xi = i16::from_le_bytes([b[so], b[so + 1]]) as f64 * unit_deg;
                    let eta = i16::from_le_bytes([b[so + 2], b[so + 3]]) as f64 * unit_deg;
                    let (lon, lat) = nested::center(band.u, c);
                    let (ra, dec) = tan::tangent_to_sky(xi, eta, lon.to_degrees(), lat.to_degrees());
                    return unit(ra, dec);
                }
                r -= n;
            }
            [f64::NAN; 3]
        })
    }

    /// Quads of `band` whose code is within `tol` of `code` on every axis.
    pub fn range_query(&self, band: &Band, code: [f64; 4], tol: f64, out: &mut Vec<usize>) {
        out.clear();
        let mut stack = vec![(0usize, band.n_quads, 0usize)];
        while let Some((lo, hi, level)) = stack.pop() {
            if hi - lo <= LEAF {
                for q in lo..hi {
                    let c = self.code(band, q);
                    if (0..4).all(|k| (c[k] - code[k]).abs() <= tol) {
                        out.push(q);
                    }
                }
                continue;
            }
            let axis = level % 4;
            let mid = lo + (hi - lo) / 2;
            let c = self.code(band, mid);
            let pivot = c[axis];
            if (0..4).all(|k| (c[k] - code[k]).abs() <= tol) {
                out.push(mid);
            }
            if code[axis] - tol <= pivot {
                stack.push((lo, mid, level + 1));
            }
            if code[axis] + tol >= pivot {
                stack.push((mid + 1, hi, level + 1));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// Search parameters for [`search`].
#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    /// Plausible pixel scale range, arcsec/px.
    pub scale_lo: f64,
    pub scale_hi: f64,
    /// Brightest image stars to build quads from.
    pub max_stars: usize,
    /// Brightest stars inside an AB circle considered for C and D.
    pub cd_cap: usize,
    /// Shortest AB, pixels (centroid noise dominates below).
    pub min_ab_px: f64,
    /// Stop after this many code lookups.
    pub max_queries: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            scale_lo: 0.1,
            scale_hi: 120.0,
            max_stars: 150,
            // 5 rather than 8: the first true hit on a dense nebular field
            // came after 86 k lookups instead of 145 k.
            cd_cap: 5,
            min_ab_px: 30.0,
            // ~1 µs per lookup: about a second of search at worst.
            max_queries: 1_000_000,
        }
    }
}

/// Code tolerance for a band: measured code errors grow on wide fields
/// (lens distortion, gnomonic anisotropy), and coarse bands hold so few
/// quads that a wider tolerance costs no false hits.
fn tolerance(depth: u8) -> f64 {
    match depth {
        0..=2 => 0.015,
        3 => 0.008,
        _ => 0.004,
    }
}

/// A WCS hypothesis from one index match.
#[derive(Clone, Debug)]
pub struct Hypothesis {
    pub wcs: WcsSolution,
    /// Image star whose addition produced the matching quad.
    pub star_n: usize,
    /// Code lookups made so far.
    pub queries: usize,
}

/// Similarity (with optional reflection) mapping image points to tangent-plane
/// points, as complex `w = α·z + β` with `z = x ± i·y`. Returns `(α, β)`.
fn fit_similarity(img: &[(f64, f64)], sky: &[(f64, f64)], mirror: bool) -> ((f64, f64), (f64, f64)) {
    let n = img.len() as f64;
    let z: Vec<(f64, f64)> = img.iter().map(|&(x, y)| (x, if mirror { -y } else { y })).collect();
    let zm = (z.iter().map(|p| p.0).sum::<f64>() / n, z.iter().map(|p| p.1).sum::<f64>() / n);
    let wm = (sky.iter().map(|p| p.0).sum::<f64>() / n, sky.iter().map(|p| p.1).sum::<f64>() / n);
    let (mut num_re, mut num_im, mut den) = (0.0, 0.0, 0.0);
    for (zi, wi) in z.iter().zip(sky) {
        let (dz, dw) = ((zi.0 - zm.0, zi.1 - zm.1), (wi.0 - wm.0, wi.1 - wm.1));
        // dw · conj(dz)
        num_re += dw.0 * dz.0 + dw.1 * dz.1;
        num_im += dw.1 * dz.0 - dw.0 * dz.1;
        den += dz.0 * dz.0 + dz.1 * dz.1;
    }
    let alpha = (num_re / den, num_im / den);
    let beta = (wm.0 - (alpha.0 * zm.0 - alpha.1 * zm.1), wm.1 - (alpha.0 * zm.1 + alpha.1 * zm.0));
    (alpha, beta)
}

/// CD matrix (deg/px) of `w = α·z + β`.
fn cd_of(alpha: (f64, f64), mirror: bool) -> [[f64; 2]; 2] {
    let (p, q) = alpha;
    if mirror {
        // (p + iq)(x − iy) = (px + qy) + i(qx − py)
        [[p, q], [q, -p]]
    } else {
        // (p + iq)(x + iy) = (px − qy) + i(qx + py)
        [[p, -q], [q, p]]
    }
}

/// WCS from 4 image ↔ index star pairs: a similarity fitted in the tangent
/// plane at the stars' midpoint, then re-fitted about the frame centre it
/// implies so the CD belongs to that plane.
fn hypothesis_wcs(
    img: &[(f64, f64); 4],
    sky: &[[f64; 3]; 4],
    mirror: bool,
    image_center: (f64, f64),
) -> Option<WcsSolution> {
    let mut tangent = radec(midpoint(sky[0], sky[1]));
    let mut wcs = None;
    for _ in 0..2 {
        let pts: Vec<(f64, f64)> = sky
            .iter()
            .map(|&v| {
                let (ra, dec) = radec(v);
                tan::sky_to_tangent(ra, dec, tangent.0, tangent.1)
            })
            .collect();
        if pts.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
            return None;
        }
        let (alpha, beta) = fit_similarity(img, &pts, mirror);
        let (cx, cy) = image_center;
        let zc = (cx, if mirror { -cy } else { cy });
        let wc = (alpha.0 * zc.0 - alpha.1 * zc.1 + beta.0, alpha.0 * zc.1 + alpha.1 * zc.0 + beta.1);
        let crval = tan::tangent_to_sky(wc.0, wc.1, tangent.0, tangent.1);
        wcs = Some(WcsSolution {
            crpix: image_center,
            crval,
            cd: cd_of(alpha, mirror),
            sip_forward: None,
            sip_reverse: None,
        });
        tangent = crval;
    }
    wcs
}

/// Enumerate image quads brightest-first ("quads whose faintest member is
/// star n"), look each up in every band of `indexes` (plain and mirrored —
/// e.g. a base pack and a narrow-field pack), and pass every
/// plausible hypothesis to `emit`. Stops when `emit` returns `true`, when the
/// query budget is spent, or when `cancelled()` turns true.
pub fn search(
    indexes: &[&QuadIndex],
    pts: &[(f64, f64)],
    image_center: (f64, f64),
    params: &SearchParams,
    cancelled: impl Fn() -> bool,
    mut emit: impl FnMut(Hypothesis) -> bool,
) {
    let m = pts.len().min(params.max_stars);
    let pts = &pts[..m];
    let arcsec = |rad: f64| rad.to_degrees() * 3600.0;
    let mut queries = 0usize;
    let mut hits = Vec::new();
    let mut cd: Vec<usize> = Vec::with_capacity(params.cd_cap);

    // Look up one image quad (indices into `pts`); true = stop.
    let mut try_quad = |quad: [usize; 4], n: usize, queries: &mut usize| -> bool {
        let p = quad.map(|i| pts[i]);
        let ab_px = {
            let (a, b) = (p[0], p[1]);
            ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
        };
        for mirror in [false, true] {
            let q = if mirror { p.map(|(x, y)| (x, -y)) } else { p };
            let Some((code, order)) = quad_code(q) else { continue };
            for (index, band) in indexes.iter().flat_map(|ix| ix.bands().iter().map(move |b| (*ix, b))) {
                // AB in pixels × plausible scale must reach the band.
                if ab_px * params.scale_hi < arcsec(band.ab_lo)
                    || ab_px * params.scale_lo >= arcsec(band.ab_hi)
                {
                    continue;
                }
                *queries += 1;
                index.range_query(band, code, tolerance(band.depth), &mut hits);
                for &h in &hits {
                    let img = order.map(|o| p[o]);
                    let sky = index.quad_stars(band, h);
                    let Some(wcs) = hypothesis_wcs(&img, &sky, mirror, image_center) else {
                        continue;
                    };
                    let scale = wcs.pixel_scale_arcsec();
                    if !(params.scale_lo..=params.scale_hi).contains(&scale) {
                        continue;
                    }
                    if emit(Hypothesis { wcs, star_n: n, queries: *queries }) {
                        return true;
                    }
                }
            }
        }
        false
    };

    let d2 = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2);
    let in_circle = |a: (f64, f64), b: (f64, f64), k: (f64, f64)| {
        let c = ((a.0 + b.0) * 0.5, (a.1 + b.1) * 0.5);
        d2(c, k) <= d2(a, b) * 0.25
    };
    for n in 3..m {
        if cancelled() || queries >= params.max_queries {
            return;
        }
        // n as B: AB = (a, n), C and D brighter than n.
        for a in 0..n {
            if d2(pts[a], pts[n]).sqrt() < params.min_ab_px {
                continue;
            }
            cd.clear();
            cd.extend((0..n).filter(|&k| k != a && in_circle(pts[a], pts[n], pts[k])).take(params.cd_cap));
            for i in 0..cd.len() {
                for j in (i + 1)..cd.len() {
                    if try_quad([a, n, cd[i], cd[j]], n, &mut queries) {
                        return;
                    }
                }
            }
        }
        // n as C or D: AB = (a, b) brighter than n, n inside its circle.
        for b in 1..n {
            for a in 0..b {
                if d2(pts[a], pts[b]).sqrt() < params.min_ab_px || !in_circle(pts[a], pts[b], pts[n]) {
                    continue;
                }
                cd.clear();
                cd.extend(
                    (0..n)
                        .filter(|&k| k != a && k != b && in_circle(pts[a], pts[b], pts[k]))
                        .take(params.cd_cap - 1),
                );
                for &k in &cd {
                    if try_quad([a, b, n, k], n, &mut queries) {
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_is_similarity_invariant_and_canonical() {
        let p = [(0.0, 0.0), (10.0, 3.0), (4.0, 2.5), (6.5, 0.5)];
        let (c0, o0) = quad_code(p).expect("code");
        for &(rot, s, tx, ty) in &[(0.7_f64, 3.0, 100.0, -50.0), (2.9, 0.01, -3.0, 7.0), (-1.3, 250.0, 0.0, 0.0)] {
            let (sr, cr) = rot.sin_cos();
            // Shuffle the input order too.
            let q = [p[2], p[0], p[3], p[1]].map(|(x, y)| (s * (cr * x - sr * y) + tx, s * (sr * x + cr * y) + ty));
            let (c, o) = quad_code(q).expect("code");
            for k in 0..4 {
                assert!((c[k] - c0[k]).abs() < 1e-9, "code {c:?} vs {c0:?}");
            }
            // Same stars in the same canonical roles.
            let back = [2usize, 0, 3, 1];
            assert_eq!(o.map(|i| back[i]), o0);
        }
        assert!(c0[0] + c0[2] <= 1.0 + 1e-12 && c0[0] <= c0[2]);
    }

    /// An open index is shared read-only between solves and threads.
    #[test]
    fn quad_index_is_send_and_sync() {
        fn shared<T: Send + Sync>() {}
        shared::<QuadIndex>();
    }

    #[test]
    fn code_rejects_points_outside_the_circle() {
        assert!(quad_code([(0.0, 0.0), (10.0, 0.0), (5.0, 4.9), (5.0, 5.2)]).is_none());
    }

    /// Build a small index from a synthetic full-sky catalog: every quad's
    /// decoded stars reproduce its stored code, and tree lookups find exactly
    /// what a scan of all codes finds.
    #[test]
    fn built_index_round_trips_and_lookups_match_a_scan() {
        let mut s = 7u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let stars: Vec<crate::cache::StarRecord> = (0..40_000)
            .map(|_| crate::cache::StarRecord {
                ra: rnd() * 360.0,
                dec: (rnd() * 2.0 - 1.0).asin().to_degrees(),
                mag: (4.0 + rnd() * 10.0) as f32,
                pmra_mas_yr: 0.0,
                pmdec_mas_yr: 0.0,
            })
            .collect();
        let cat_dir = tempfile::tempdir().expect("tmp");
        crate::cache::build_cache(stars, cat_dir.path(), 2016.0, |_| {}).expect("cache");
        let cache = StarCache::open(cat_dir.path()).expect("open cache");
        let idx_dir = tempfile::tempdir().expect("tmp");
        let params = BuildParams { min_depth: 2, max_depth: 2, stars_per_cell: 10, quads_per_cell: 8 };
        let (_, n_quads) = build_index(&[&cache], idx_dir.path(), &params, |_| {}).expect("build");
        assert!(n_quads > 1000, "only {n_quads} quads");
        let idx = QuadIndex::open(idx_dir.path()).expect("open index");
        let band = idx.bands()[0];

        for q in 0..band.n_quads() {
            let v = idx.quad_stars(&band, q);
            let (mra, mdec) = radec(midpoint(v[0], v[1]));
            let pts = v.map(|x| {
                let (ra, dec) = radec(x);
                tan::sky_to_tangent(ra, dec, mra, mdec)
            });
            let stored = idx.code(&band, q);
            // A, B, C, D are stored in canonical order, so the recomputed
            // code must agree with them as given (no re-sorting).
            let (code, order) = quad_code(pts).expect("decoded quad is valid");
            assert_eq!(order, [0, 1, 2, 3], "quad {q}: roles changed after decoding");
            for k in 0..4 {
                assert!((code[k] - stored[k]).abs() < 2e-3, "quad {q}: code {code:?} vs {stored:?}");
            }
        }

        let all: Vec<[f64; 4]> = (0..band.n_quads()).map(|q| idx.code(&band, q)).collect();
        let mut out = Vec::new();
        for _ in 0..200 {
            let q = [rnd(), rnd(), rnd(), rnd()];
            let tol = 0.02 + rnd() * 0.08;
            idx.range_query(&band, q, tol, &mut out);
            let mut got = out.clone();
            got.sort_unstable();
            let want: Vec<usize> =
                (0..all.len()).filter(|&i| (0..4).all(|k| (all[i][k] - q[k]).abs() <= tol)).collect();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn similarity_hypothesis_recovers_a_known_wcs() {
        // A 1.5"/px frame rotated 30° about (120, 35), mirrored and not.
        for mirror in [false, true] {
            let s = 1.5 / 3600.0;
            let (sr, cr) = 30f64.to_radians().sin_cos();
            let m = if mirror { -1.0 } else { 1.0 };
            let truth = WcsSolution {
                crpix: (2000.0, 1500.0),
                crval: (120.0, 35.0),
                cd: [[s * cr, -m * s * sr], [s * sr, m * s * cr]],
                sip_forward: None,
                sip_reverse: None,
            };
            let img = [(300.0, 400.0), (3500.0, 2600.0), (1800.0, 1200.0), (2500.0, 1900.0)];
            let sky = img.map(|(x, y)| {
                let (ra, dec) = truth.pixel_to_sky(x, y);
                unit(ra, dec)
            });
            let w = hypothesis_wcs(&img, &sky, mirror, (2000.0, 1500.0)).expect("wcs");
            let (ra, dec) = w.pixel_to_sky(100.0, 2900.0);
            let (tra, tdec) = truth.pixel_to_sky(100.0, 2900.0);
            let err = crate::math::great_circle_deg(ra, dec, tra, tdec) * 3600.0;
            assert!(err < 2.0, "mirror={mirror}: corner off by {err}\"");
            assert!((w.pixel_scale_arcsec() - 1.5).abs() < 0.01);
        }
    }
}
