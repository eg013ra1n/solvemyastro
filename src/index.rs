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
//! | header (64 B) | magic `SMAQIDX1`, `u32` version, `u32` n_bands, `u64` n_stars, `f64` catalog epoch |
//! | band table | per band 48 B: `u32` depth, `u32` 0, `f64` ab_lo, `f64` ab_hi (rad), `u64` n_quads, `u64` codes offset, `u64` ids offset |
//! | stars | `n_stars` × `f32` x, y, z unit vector — only stars some quad uses |
//! | per band | codes `n_quads` × 4 `u16` ([`CODE_MIN`] + q·[`CODE_STEP`]), tree order; ids `n_quads` × 4 `u32` (A, B, C, D) |

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use cdshealpix::nested;
use memmap2::Mmap;

use crate::cache::StarCache;
use crate::wcs::{tan, WcsSolution};

const MAGIC: &[u8; 8] = b"SMAQIDX1";
const VERSION: u32 = 2;
const HEADER_SIZE: usize = 64;
const BAND_ENTRY_SIZE: usize = 48;
const STAR_SIZE: usize = 12;
const CODE_SIZE: usize = 8;
const IDS_SIZE: usize = 16;
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
    /// Brightest stars kept per depth `d+1` cell.
    pub stars_per_cell: usize,
    /// Quads kept per depth `d+1` cell.
    pub quads_per_cell: usize,
}

impl Default for BuildParams {
    fn default() -> Self {
        Self { min_depth: 2, max_depth: 7, stars_per_cell: 10, quads_per_cell: 8 }
    }
}

/// Global star table under construction, deduplicated across bands.
struct StarTable {
    key_to_id: HashMap<(u8, u32, u32), u32>,
    xyz: Vec<[f64; 3]>,
    mag: Vec<f32>,
}

impl StarTable {
    fn id(&mut self, key: (u8, u32, u32), xyz: [f64; 3], mag: f32) -> u32 {
        let next = self.xyz.len() as u32;
        *self.key_to_id.entry(key).or_insert_with(|| {
            self.xyz.push(xyz);
            self.mag.push(mag);
            next
        })
    }
}

struct BuiltBand {
    depth: u8,
    ab_lo: f64,
    ab_hi: f64,
    codes: Vec<[u16; 4]>,
    ids: Vec<[u32; 4]>,
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
    let epoch = layers[0].catalog_epoch();
    let mut stars = StarTable { key_to_id: HashMap::new(), xyz: Vec::new(), mag: Vec::new() };
    let mut bands = Vec::new();
    for d in params.min_depth..=params.max_depth {
        let u = d + 1;
        let cells = uniform_stars(layers, u, params.stars_per_cell, &mut stars)?;
        let mut band = build_band(&cells, &stars, d, u, params.quads_per_cell);
        tree_order(&mut band.codes, &mut band.ids);
        progress(&format!(
            "band depth {d}: AB {:.3}–{:.3}°, {} quads",
            band.ab_lo.to_degrees(),
            band.ab_hi.to_degrees(),
            band.codes.len()
        ));
        bands.push(band);
    }
    std::fs::create_dir_all(out_dir)?;
    let n_quads = bands.iter().map(|b| b.codes.len()).sum();
    let n_stars = write_index(&out_dir.join(INDEX_FILE), epoch, &stars, &bands)?;
    Ok((n_stars, n_quads))
}

/// The `k` brightest stars of every depth-`u` cell (global star ids, mag
/// order). Layers are filled in order, a fainter layer only where a cell is
/// still short.
fn uniform_stars(
    layers: &[&StarCache],
    u: u8,
    k: usize,
    stars: &mut StarTable,
) -> Result<Vec<Vec<u32>>> {
    let n_cells = nested::n_hash(u) as usize;
    // (mag, layer, pixel, idx, xyz) per cell, kept to the k brightest.
    let mut best: Vec<Vec<(f32, u8, u32, u32, [f64; 3])>> = vec![Vec::new(); n_cells];
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
            for (ri, r) in layer.iter_pixel_records(px)?.into_iter().enumerate() {
                let v = unit(r.ra, r.dec);
                let c = nested::hash(u, r.ra.to_radians(), r.dec.to_radians()) as usize;
                let list = &mut best[c];
                if list.len() >= k && r.mag >= list[k - 1].0 {
                    continue;
                }
                let pos = list.partition_point(|e| e.0 <= r.mag);
                list.insert(pos, (r.mag, li as u8, px as u32, ri as u32, v));
                list.truncate(k);
            }
        }
    }
    Ok(best
        .into_iter()
        .map(|list| {
            list.into_iter().map(|(m, l, p, i, v)| stars.id((l, p, i), v, m)).collect()
        })
        .collect())
}

/// Quads of band `d` from the uniform star lists of depth-`u` cells.
fn build_band(cells: &[Vec<u32>], stars: &StarTable, d: u8, u: u8, qcap: usize) -> BuiltBand {
    let s = side(d);
    let (ab_lo, ab_hi) = (s / std::f64::consts::SQRT_2, s * std::f64::consts::SQRT_2);
    let mut codes = Vec::new();
    let mut ids = Vec::new();
    let mut nb: Vec<u32> = Vec::new();
    for (c, _) in cells.iter().enumerate() {
        nb.clear();
        for n in nested::neighbours(u, c as u64, true).values_vec() {
            nb.extend_from_slice(&cells[n as usize]);
        }
        nb.sort_by(|a, b| {
            stars.mag[*a as usize].partial_cmp(&stars.mag[*b as usize]).unwrap_or(std::cmp::Ordering::Equal)
        });
        nb.dedup();
        let v: Vec<[f64; 3]> = nb.iter().map(|&i| stars.xyz[i as usize]).collect();
        let mut made = 0;
        'pairs: for j in 1..nb.len() {
            for i in 0..j {
                let ab = angle(v[i], v[j]);
                if ab < ab_lo || ab >= ab_hi {
                    continue;
                }
                let m = midpoint(v[i], v[j]);
                if cell_of(u, m) as usize != c {
                    continue;
                }
                // The two brightest other stars inside the circle on AB.
                let r = ab * 0.5;
                let mut cd = (0..nb.len()).filter(|&k| k != i && k != j && angle(v[k], m) <= r);
                let (Some(kc), Some(kd)) = (cd.next(), cd.next()) else { continue };
                let (mra, mdec) = radec(m);
                let quad = [i, j, kc, kd];
                let pts = quad.map(|q| {
                    let (ra, dec) = radec(v[q]);
                    tan::sky_to_tangent(ra, dec, mra, mdec)
                });
                let Some((code, order)) = quad_code(pts) else { continue };
                codes.push(code.map(quantize));
                ids.push(order.map(|o| nb[quad[o]]));
                made += 1;
                if made >= qcap {
                    break 'pairs;
                }
            }
        }
    }
    BuiltBand { depth: d, ab_lo, ab_hi, codes, ids }
}

/// Reorder `codes` (and `ids` with them) into implicit kd-tree order: the
/// median of `[lo, hi)` on axis `level % 4` sits at `(lo + hi) / 2`.
fn tree_order(codes: &mut [[u16; 4]], ids: &mut [[u32; 4]]) {
    let mut perm: Vec<usize> = (0..codes.len()).collect();
    fn rec(perm: &mut [usize], codes: &[[u16; 4]], level: usize) {
        if perm.len() <= LEAF {
            return;
        }
        let axis = level % 4;
        let mid = perm.len() / 2;
        perm.select_nth_unstable_by_key(mid, |&a| codes[a][axis]);
        let (left, right) = perm.split_at_mut(mid);
        rec(left, codes, level + 1);
        rec(&mut right[1..], codes, level + 1);
    }
    rec(&mut perm, codes, 0);
    let c: Vec<[u16; 4]> = perm.iter().map(|&p| codes[p]).collect();
    let i: Vec<[u32; 4]> = perm.iter().map(|&p| ids[p]).collect();
    codes.copy_from_slice(&c);
    ids.copy_from_slice(&i);
}

fn write_index(path: &Path, epoch: f64, stars: &StarTable, bands: &[BuiltBand]) -> Result<usize> {
    // Keep only the stars some quad uses, renumbered in first-use order.
    let mut remap = vec![u32::MAX; stars.xyz.len()];
    let mut kept: Vec<[f64; 3]> = Vec::new();
    for b in bands {
        for q in &b.ids {
            for &i in q {
                if remap[i as usize] == u32::MAX {
                    remap[i as usize] = kept.len() as u32;
                    kept.push(stars.xyz[i as usize]);
                }
            }
        }
    }
    let mut w = BufWriter::new(File::create(path).with_context(|| format!("create {}", path.display()))?);
    let mut header = [0u8; HEADER_SIZE];
    header[..8].copy_from_slice(MAGIC);
    header[8..12].copy_from_slice(&VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&(bands.len() as u32).to_le_bytes());
    header[16..24].copy_from_slice(&(kept.len() as u64).to_le_bytes());
    header[24..32].copy_from_slice(&epoch.to_le_bytes());
    w.write_all(&header)?;

    let stars_off = HEADER_SIZE + bands.len() * BAND_ENTRY_SIZE;
    let mut off = stars_off + kept.len() * STAR_SIZE;
    for b in bands {
        let n = b.codes.len();
        let (codes_off, ids_off) = (off, off + n * CODE_SIZE);
        off = ids_off + n * IDS_SIZE;
        w.write_all(&(b.depth as u32).to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?;
        w.write_all(&b.ab_lo.to_le_bytes())?;
        w.write_all(&b.ab_hi.to_le_bytes())?;
        w.write_all(&(n as u64).to_le_bytes())?;
        w.write_all(&(codes_off as u64).to_le_bytes())?;
        w.write_all(&(ids_off as u64).to_le_bytes())?;
    }
    for v in &kept {
        for x in [v[0] as f32, v[1] as f32, v[2] as f32] {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    for b in bands {
        for c in &b.codes {
            for x in c {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        for q in &b.ids {
            for &x in q {
                w.write_all(&remap[x as usize].to_le_bytes())?;
            }
        }
    }
    w.flush()?;
    Ok(kept.len())
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
    n_quads: usize,
    codes_off: usize,
    ids_off: usize,
}

impl Band {
    /// Number of quads in the band.
    pub fn n_quads(&self) -> usize {
        self.n_quads
    }
}

/// A memory-mapped quad index.
pub struct QuadIndex {
    mmap: Mmap,
    bands: Vec<Band>,
    n_stars: usize,
    stars_off: usize,
}

fn f32_at(b: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
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
            bail!("{}: index version {version}, expected {VERSION}", path.display());
        }
        let n_bands = u32_at(b, 12) as usize;
        let n_stars = u64_at(b, 16) as usize;
        let stars_off = HEADER_SIZE + n_bands * BAND_ENTRY_SIZE;
        ensure!(b.len() >= stars_off + n_stars * STAR_SIZE, "{} is truncated", path.display());
        let mut bands = Vec::with_capacity(n_bands);
        for i in 0..n_bands {
            let o = HEADER_SIZE + i * BAND_ENTRY_SIZE;
            let band = Band {
                depth: u32_at(b, o) as u8,
                ab_lo: f64::from_le_bytes(b[o + 8..o + 16].try_into().expect("8 bytes")),
                ab_hi: f64::from_le_bytes(b[o + 16..o + 24].try_into().expect("8 bytes")),
                n_quads: u64_at(b, o + 24) as usize,
                codes_off: u64_at(b, o + 32) as usize,
                ids_off: u64_at(b, o + 40) as usize,
            };
            ensure!(
                band.codes_off + band.n_quads * CODE_SIZE <= b.len()
                    && band.ids_off + band.n_quads * IDS_SIZE <= b.len(),
                "{}: band {i} points beyond the file",
                path.display()
            );
            bands.push(band);
        }
        Ok(Self { mmap, bands, n_stars, stars_off })
    }

    /// The index's scale bands, coarsest first.
    pub fn bands(&self) -> &[Band] {
        &self.bands
    }

    /// Number of index stars.
    pub fn n_stars(&self) -> usize {
        self.n_stars
    }

    /// Unit vector of index star `i`.
    pub fn star(&self, i: u32) -> [f64; 3] {
        let o = self.stars_off + i as usize * STAR_SIZE;
        let b = &self.mmap[..];
        [f32_at(b, o) as f64, f32_at(b, o + 4) as f64, f32_at(b, o + 8) as f64]
    }

    fn code(&self, band: &Band, q: usize) -> [f64; 4] {
        let o = band.codes_off + q * CODE_SIZE;
        let b = &self.mmap[..];
        std::array::from_fn(|k| {
            CODE_MIN + u16::from_le_bytes([b[o + 2 * k], b[o + 2 * k + 1]]) as f64 * CODE_STEP
        })
    }

    /// Star ids (A, B, C, D) of quad `q` in `band`.
    pub fn quad_stars(&self, band: &Band, q: usize) -> [u32; 4] {
        let o = band.ids_off + q * IDS_SIZE;
        let b = &self.mmap[..];
        [u32_at(b, o), u32_at(b, o + 4), u32_at(b, o + 8), u32_at(b, o + 12)]
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
/// star n"), look each up in the index (plain and mirrored), and pass every
/// plausible hypothesis to `emit`. Stops when `emit` returns `true`, when the
/// query budget is spent, or when `cancelled()` turns true.
pub fn search(
    index: &QuadIndex,
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
            for band in index.bands() {
                // AB in pixels × plausible scale must reach the band.
                if ab_px * params.scale_hi < arcsec(band.ab_lo)
                    || ab_px * params.scale_lo >= arcsec(band.ab_hi)
                {
                    continue;
                }
                *queries += 1;
                index.range_query(band, code, tolerance(band.depth), &mut hits);
                for &h in &hits {
                    let ids = index.quad_stars(band, h);
                    let img = order.map(|o| p[o]);
                    let sky = ids.map(|i| index.star(i));
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

    #[test]
    fn code_rejects_points_outside_the_circle() {
        assert!(quad_code([(0.0, 0.0), (10.0, 0.0), (5.0, 4.9), (5.0, 5.2)]).is_none());
    }

    #[test]
    fn kd_range_query_matches_linear_scan() {
        let mut s = 1u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let n = 5000;
        let mut codes: Vec<[u16; 4]> = (0..n).map(|_| [rnd(), rnd(), rnd(), rnd()].map(quantize)).collect();
        let mut ids: Vec<[u32; 4]> = (0..n as u32).map(|i| [i, 0, 0, 0]).collect();
        let orig = codes.clone();
        tree_order(&mut codes, &mut ids);
        let dir = tempfile::tempdir().expect("tmp");
        let band = BuiltBand { depth: 5, ab_lo: 0.01, ab_hi: 0.02, codes, ids };
        let stars = StarTable { key_to_id: HashMap::new(), xyz: vec![[1.0, 0.0, 0.0]; n], mag: vec![1.0; n] };
        write_index(&dir.path().join(INDEX_FILE), 2016.0, &stars, &[band]).expect("write");
        let idx = QuadIndex::open(dir.path()).expect("open");
        let b = idx.bands()[0];
        let mut out = Vec::new();
        for _ in 0..300 {
            let q = [rnd(), rnd(), rnd(), rnd()];
            let tol = 0.05 + rnd() * 0.1;
            idx.range_query(&b, q, tol, &mut out);
            // Compare the matched codes (star ids are renumbered on write).
            let key = |c: [f64; 4]| c.map(|x| ((x - CODE_MIN) / CODE_STEP).round() as u16);
            let mut got: Vec<[u16; 4]> = out.iter().map(|&h| key(idx.code(&b, h))).collect();
            got.sort_unstable();
            let mut want: Vec<[u16; 4]> = orig
                .iter()
                .filter(|c| (0..4).all(|k| (CODE_MIN + c[k] as f64 * CODE_STEP - q[k]).abs() <= tol))
                .copied()
                .collect();
            want.sort_unstable();
            assert!(!want.is_empty() || got.is_empty());
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
