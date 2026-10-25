# Quad index: building and using it

The quad index makes blind and scale-free solves fast. Without it the solver
searches the sky with a spiral of catalog cones, and the cost grows with the
sky area and the scale range it has to cover: seconds to minutes blind. With
it, an image quad is looked up directly in a precomputed table of catalog
quads. One match gives a full WCS hypothesis, and the search costs the same
anywhere on the sky.

Fully hinted frames (position **and** scale) never use the index. They keep
the spiral search, which is near-instant around a good hint, so adding an
index does not change their solutions.

## Packs

An index is built as one or more **packs**. Each pack is a directory holding
a single file, `quads.smaq`. A solve can use several packs at once.

| Pack | Depths | Fields served | Quads | File | Build time | Peak RAM |
|------|--------|---------------|-------|------|------------|----------|
| base | 2–7 | ~1.1° – ~54° | 8.4 M | 147 MB | 26 s | 1.5 GB |
| narrow | 8 (4 quads/cell) | ~0.55° – ~1.1° | 12.6 M | 236 MB | 55 s | 4.8 GB |

Build times and memory were measured on a 4-core cloud VM.

- **Base** covers the common case: 35 mm lenses to ~1000 mm on APS-C.
- **Narrow** extends blind solving to ~2000 mm on APS-C. It is optional:
  frames this narrow usually come with mount coordinates, and those solve
  without an index.

Frames narrower than ~0.5° (e.g. 3000–4000 mm) are not covered by either
pack. Blind they fail fast (see [Behaviour](#behaviour)); with a position hint
they solve through the spiral.

## Building

### Input

The density-tier catalog: a directory containing `tier_500/` and
`tier_2000/`, each holding a `stars.smac`. This is the same layout Athenaeum
installs under `catalogs/smac_gaia/` and the one
`scripts/fetch_bench_data.sh` downloads.

- `tier_500` is enough for the base pack.
- The narrow pack also needs `tier_2000`.
- Deeper tiers are not read.

### Commands

```
solvemyastro build-index --tiers <catalog dir> --out <index dir>/base
solvemyastro build-index --tiers <catalog dir> --out <index dir>/narrow \
    --min-depth 8 --max-depth 8 --quads-per-cell 4
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--tiers <dir>` | required | Directory containing `tier_500` (and `tier_2000`). |
| `--out <dir>` | required | Output directory; `quads.smaq` is written into it. |
| `--min-depth <d>` | 2 | Coarsest band. Depth 2 = quads of 10–21°. |
| `--max-depth <d>` | 7 | Finest band. Depth 7 = 0.32–0.65°, depth 8 = 0.16–0.32°. |
| `--stars-per-cell <k>` | 10 | Brightest stars kept per cell (2–28). |
| `--quads-per-cell <q>` | 8 | Quads kept per cell. 4 roughly halves a pack; 8 gives a few more true matches. |

A band at depth `d` holds quads whose longest side (AB) is in
`[side_d/√2, side_d·√2)`, where `side_d` is the mean HEALPix pixel side at that
depth. It serves fields of roughly 3–6× AB.

Build output is **deterministic**: the same tiers and flags give a
byte-identical `quads.smaq`. A published pack can therefore be verified by its
SHA-256.

### Format version

`quads.smaq` carries a format version, currently **3**. `QuadIndex::open`
refuses any other version with an error ending in "rebuild it". Packs built by
an older solvemyastro must be rebuilt, not converted. The layout is documented
at the top of `src/index.rs`.

## Distributing packs (suggested)

This mirrors how the catalog tiers are shipped. The Athenaeum side is not
implemented yet.

```
catalogs/
  smac_gaia/tier_500 … tier_8000/stars.smac
  smac_index/
    manifest.json
    base/quads.smaq
    narrow/quads.smaq
```

A manifest in the style of the tiers one:

```json
{
  "version": 1,
  "format": 3,
  "catalog_epoch": 2016.0,
  "packs": [
    { "name": "base",   "zip": "index_base.zip",   "sha256": "index_base.zip.sha256",
      "dir": "base",   "size_bytes": 147079024, "min_fov_deg": 1.3 },
    { "name": "narrow", "zip": "index_narrow.zip", "sha256": "index_narrow.zip.sha256",
      "dir": "narrow", "size_bytes": 235801384, "min_fov_deg": 0.65 }
  ]
}
```

- **`format`** lets the app skip packs it cannot open.
- **`min_fov_deg`** equals `QuadIndex::min_fov_deg()`, so the app can explain
  what each pack adds.
- **Compression.** The file compresses only modestly: gzip takes base from
  147 to 129 MB and narrow from 236 to 205 MB. Ship it zipped for the
  checksum and resumable download, not for size.

## Using it from the CLI

```
solvemyastro solve <image> --tiers <catalog dir> \
    --index <index dir>/base --index <index dir>/narrow [--json]
```

`--index` may be repeated. `--catalog <dir>` (a single cache) works too.
Hints are optional:

| Flag | Meaning |
|------|---------|
| `--ra <deg>`, `--dec <deg>` | Position hint. |
| `--fov <deg>` or `--scale <arcsec/px>` | Scale hint. |
| `--radius <deg>` | Search radius around the position hint. |
| `--epoch <year>` | Observation epoch. |

Example: a 3° M42 frame solved blind takes 11.2 s without `--index` and 1.6 s
with both packs.

## Using it from the library

Open the packs once and keep them for the life of the process. Then pass them
to `solve_with_index` instead of `solve`.

```rust
use solvemyastro::{solve_with_index, Caches, QuadIndex, SolveConfig, SolveHints, StarCache};

// At startup. Opening only maps the file; nothing is read or rebuilt.
let tiers: Vec<StarCache> = ["tier_500", "tier_2000", "tier_5000", "tier_8000"]
    .iter()
    .map(|t| StarCache::open(&catalog_dir.join(t)))
    .collect::<anyhow::Result<_>>()?;
let packs: Vec<QuadIndex> = ["base", "narrow"]
    .iter()
    .map(|p| index_dir.join(p))
    .filter(|d| d.join(solvemyastro::index::INDEX_FILE).is_file())
    .filter_map(|d| match QuadIndex::open(&d) {
        Ok(ix) => Some(ix),
        Err(e) => {
            // Wrong format version or a damaged file: solve without this
            // pack rather than fail, and offer a re-download.
            tracing::warn!(error = %e, "quad index unavailable");
            None
        }
    })
    .collect();

// Per solve.
let tier_refs: Vec<&StarCache> = tiers.iter().collect();
let index_refs: Vec<&QuadIndex> = packs.iter().collect();
let solution = solve_with_index(
    &image_path,
    &hints, // SolveHints; any field may be None
    &Caches::layered(&tier_refs),
    &index_refs, // may be empty: then this is exactly `solve`
    &SolveConfig::default(),
    Some(&cancel_flag),
)?;
```

### Lifetime and threading

- **Thread safety.** `QuadIndex` is `Send + Sync` and read-only. One opened
  set of packs can serve concurrent solves on several threads.
- **Memory.** The file is memory-mapped. Memory use is whatever pages the OS
  keeps cached; no copy is made.
- **Keep it open.** Opening is cheap, so reopening per solve also works.
  Keeping it open just avoids repeating it; the OS page cache is what makes
  later solves fast either way.
- **Cold start.** The first blind solve after a reboot faults in index pages
  from disk. This was not measured. Estimate: well under a second on an SSD,
  seconds on a slow disk. Reading the file once in the background at startup
  warms it.

### When the index is used

`solve_with_index` decides per solve, from the hints:

| Hints | Index used? | Scale range searched |
|-------|-------------|----------------------|
| position + scale | no: spiral, exactly as `solve` | — |
| position only | yes; matches far from the hint are dropped | 0.1–120″/px |
| scale only | yes, whole sky | hint ÷5 … ×5 |
| none | yes, whole sky | 0.1–120″/px |

`SolveHints::fov_deg` counts as a scale hint. The position filter keeps
centres within `search_radius_deg` plus one field width of the hint.

The retries inside `solve_with_index` (the scale-recovery steps and the
position-cleared retry) apply the same rule, so a retry without a usable scale
uses the index too.

### Behaviour

1. Image quads are built from the 150 brightest stars, brightest first, and
   looked up in every band of every pack (both mirror parities). The budget is
   1 M lookups, about a second.
2. The first 16 hypotheses are verified directly. After that, a hypothesis is
   verified only once a second, independent match agrees with it (same centre,
   scale, rotation and parity). Chance matches scatter; a true field repeats.
3. Verification is the same pipeline as the spiral: seed inliers and log-odds,
   refit, SIP, then the bright-star completeness gate at `log(1e9)`.
4. If no hypothesis verifies, the spiral runs only where the index did not
   look:
   - rungs of 6° and wider, which are cheap and catch wide lenses whose
     distortion defeats 4-star hypotheses;
   - with a position hint only, the rungs finer than the packs reach.

   A frame nothing can solve fails in ~20–30 s instead of minutes.

Cancellation (`cancel` set to `true`) is checked each time the search moves
on to the next image star and before each verification.

### What to expect

Measured blind, i.e. no hints, with both packs:

| Set | Result |
|-----|--------|
| 37 real bench frames | All solved. Every solution matches the hinted solution (centre within 11″, scale within 0.3 %). 0.2–4.7 s each, 43 s total. Without the index, blind: 330 s total, up to 101 s per frame. |
| Synthetic 50–2000 mm | All solved, 0.2–1.4 s each, 0 wrong. |
| Synthetic 3000–4000 mm | Partly solved by the index; the rest fail in ~25 s. |
| Synthetic 35 mm (38°, barrel distortion) | 3 of 4 solved, 2 of them imprecise. Wide-lens distortion is out of scope. |

### Logging

| Level | Event | Fields |
|-------|-------|--------|
| `debug` | `quad-index search` | `hypotheses`, `rejected`, `queries_at_last_hypothesis`, `search_ms`, `verify_ms`, `solved` |
| `debug` | `bright-star completeness` | Per verified candidate. |
| `info` | `solve succeeded` | Carries `path = "index"` when the index found the solution. |
| `warn` | (fallback) | When the index found nothing and the spiral takes over. |

## Testing with an index

```
SOLVEMYASTRO_SYNTH_DIR=<synthetic frames> SOLVEMYASTRO_TIER_DIR=<catalog dir> \
SOLVEMYASTRO_INDEX_DIR=<index dir>/base:<index dir>/narrow SYNTH_MODE=blind \
  cargo test --release --test synth_bench -- --ignored --nocapture
```

- `SOLVEMYASTRO_INDEX_DIR` takes a `:`-separated list of pack directories.
- `SYNTH_MODE` selects `hinted`, `position`, `scale` or `blind`.
- The gate fails on any wrong solution.
- Synthetic frames come from `cargo run --release --example synth_fields`
  (see README).

## Limitations

- **Narrow fields.** Fields under ~0.5° need a depth-9 pack, not built yet
  (estimated ~1 GB). They need its stars from `tier_2000` and deeper.
- **Distortion.** Heavy distortion (fisheye, ultra-wide) is out of scope. The
  4-star hypothesis assumes a near-gnomonic lens across the quad.
- **Version coupling.** Index and catalog are built from the same Gaia epoch
  (J2016). Quads use catalog positions without proper motion; verification
  uses the tiers with proper motion applied to the observation epoch.
