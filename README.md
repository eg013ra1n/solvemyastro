# solvemyastro

Standalone astrometric plate solver. Produces **precise WCS including SIP
distortion** in-process — no external binary, no subprocess. Depends only
on `rustafits` (`astroimage`) for FITS/XISF decode + raw star detection.
Used by Athenaeum as a library and runnable as a standalone CLI, which also
builds the star catalog from Gaia data.

## Install

```
cargo install solvemyastro --features cli --locked
```

The library alone (`solvemyastro = "0.3"` in `Cargo.toml`) has no command line
and none of the catalog-building dependencies: those sit behind the `catalog`
(Gaia ingest, tiers, packages) and `cli` (the `solvemyastro` command) features.

## CLI

One subcommand per stage, from Gaia files to a solved frame. Results go to
stdout, progress to stderr (`-q` shows warnings only, `RUST_LOG` overrides,
`--log <file>` also writes every event as a JSON line). Exit code 0 on
success, 1 on a failure or a refused solve, 2 on bad arguments.

```
solvemyastro ingest --gaia <gaia dir> --out <bins> [--mag-limit 21] [--jobs 4] [--verify]
solvemyastro build-tiers --bins <bins> --out <tiers> [--epoch 2016.0]
solvemyastro build-index --tiers <tiers> --out <pack> (--preset base|narrow | [--min-depth --max-depth --stars-per-cell --quads-per-cell])
solvemyastro package [--tiers <tiers>] [--index <name>=<pack>]... --out <publish>
solvemyastro solve <image> (--tiers <tiers> | --catalog <cache>) [--index <pack>]... [--ra --dec] [--fov | --scale] [--radius] [--epoch] [--json] [--no-gate]
solvemyastro info <dir>
solvemyastro build-cache (--bins | --from) <bins> --out <cache> [--epoch 2016.0]
solvemyastro build-bright-cache --from <cache> --out <bright cache> [--bright-floor 14.0] [--target-density 100] [--mode hybrid|floor-only|cap-only]
```

`solve` checks every solution with the acceptance gate (inlier floor, inlier
ratio in a dense field, RMS against the pixel tolerance, plausible scale, and
the scale against `--scale` when given). A refused solve prints the reason
(`rejected: ...`) on stderr and nothing on stdout; `--no-gate` prints it anyway.
`--json` prints `ra, dec, scale, rot, matched, rms_px, inlier_ratio,
expected_in_fov` and `gate` (`"passed"` or `"off"`).

```
solvemyastro solve frame.fits --tiers ~/catalog --index ~/catalog/index/base --json
```

Known issue: the gate's in-field radius takes the square root of the detected
star count for the image side, so `expected_in_fov` comes out far too low and the
dense-field ratio check rarely arms. A fix changes verdicts and is planned.

`info` says what a folder is: a star cache, a folder of tiers, an index pack or a
folder of packs, a folder of bins or a publish tree, with its counts. `cache-info` is the old name.

## Building the star catalog

The chain from Gaia files to what the solver reads. Every command runs its
stage from scratch each time. `build-tiers` builds each tier in
`tier_<d>.partial/` and gives it its name only when the tier is finished, so a
killed run never leaves a tier that looks complete (the next run clears what it
left). Tiers are renamed one at a time, so run `package` only after a
`build-tiers` that finished. `build-cache` writes its folder in place: if it
fails midway, delete the folder and run it again.

1. **Gaia data.** Download the Gaia DR3 `GaiaSource_*.csv.gz` files from ESA's
   archive (<https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/>) into one
   folder; the archive's `_MD5SUM.txt` next to them lets `--verify` check the
   download. A subset works: see "Complete cells" below.
2. **Bins.** `solvemyastro ingest --gaia <dir> --out <bins>` keeps the stars
   brighter than G 21 with RUWE under 1.4 and writes one
   `healpix_NNNNNN.bin` per HEALPix level-6 cell, sorted by magnitude, plus
   `coverage.json` (which level-8 pixel ranges the file names cover). The output
   does not depend on `--jobs` or the file order. The folder must not hold bins
   already.
3. **Tiers.** `solvemyastro build-tiers --bins <bins> --out <tiers>` cuts every
   cell by rank into four additive caches, `tier_500`, `tier_2000`, `tier_5000`
   and `tier_8000` (stars per square degree), each a `stars.smac`.
4. **Index packs.** `solvemyastro build-index --tiers <tiers> --out <pack>
   --preset base` (depths 2-7 over `tier_500`, fields from about 1.1 degrees) and
   `--preset narrow` (depth 8 over `tier_500` + `tier_2000`) build the packs
   that were published; without a preset the flags set the parameters, over
   `tier_500` + `tier_2000`. See [QUAD_INDEX.md](QUAD_INDEX.md).
5. **Publish tree.** `solvemyastro package --tiers <tiers> --index
   base=<base pack> --index narrow=<narrow pack> --out <publish>` writes
   `tier_<d>.zip` and `index_<name>.zip` with `.sha256` sidecars, and
   `manifest.json` and `index_manifest.json` that name them.

```
solvemyastro ingest --gaia gaia --out bins
solvemyastro build-tiers --bins bins --out tiers
solvemyastro build-index --tiers tiers --out packs/base --preset base
solvemyastro build-index --tiers tiers --out packs/narrow --preset narrow
solvemyastro package --tiers tiers --index base=packs/base --index narrow=packs/narrow --out publish
solvemyastro info publish
```

**Complete cells.** `ingest` takes whatever files are in the folder. Gaia
splits its files by the HEALPix index inside `source_id` and the bins use the
DR3 `ra`/`dec`, so a cell is "complete by file range" when the file names cover
all of its level-8 pixels, and a star near a cell edge can still sit in a
neighbouring file. `solvemyastro info <bins>` prints the count
(`complete cells: N`); `build-tiers` warns when cells are partial, because their
tiers are thinner than the density they are named for.

**Credit.** Catalogs built from Gaia data should carry ESA's line: "This work
has made use of data from the European Space Agency (ESA) mission Gaia
(https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and
Analysis Consortium (DPAC)". `package` writes no credit of its own: whoever
publishes catalogs built from Gaia carries that line.

## Benchmarks

Bench data is not in the repository. `scripts/fetch_bench_data.sh <dest>`
downloads both sets and unpacks them:

- frames: <https://artfrom.space/builds/platesolve_bench/platesolve_bench.zip>
  (38 FITS/XISF frames) → `<dest>/bench`
- density-tier Gaia catalog: <https://artfrom.space/catalogs/> (`tier_500` …
  `tier_8000`, SHA-256 checked) → `<dest>/catalogs`

```
SOLVEMYASTRO_BENCH_DIR=<dest>/bench SOLVEMYASTRO_TIER_DIR=<dest>/catalogs \
  cargo test --release --test corpus_bench -- --ignored corpus_layered_tiers --nocapture
```

Quad index for blind and scale-free solves (`solve_with_index`; building,
shipping and wiring it in: [QUAD_INDEX.md](QUAD_INDEX.md)): a base pack
for fields from ~1.1° (depth 2–7, ~150 MB) and an optional narrow-field pack
from ~0.55° (depth 8, ~240 MB with `--quads-per-cell 4`), both built from the
tiers:

```
solvemyastro build-index --tiers <dest>/catalogs --out <dest>/index/base --preset base
solvemyastro build-index --tiers <dest>/catalogs --out <dest>/index/narrow --preset narrow
```

Synthetic fields from 35 mm to 4000 mm, rendered from the same catalog with
the true solution in their headers:

```
cargo run --release --example synth_fields -- --tiers <dest>/catalogs --out <synth>
SOLVEMYASTRO_SYNTH_DIR=<synth> SOLVEMYASTRO_TIER_DIR=<dest>/catalogs \
  cargo test --release --test synth_bench -- --ignored --nocapture   # SYNTH_MODE=hinted|position|scale|blind
# with SOLVEMYASTRO_INDEX_DIR=<dest>/index/base:<dest>/index/narrow for the index path
```

## Status

In use: solvemyastro is the only solver Athenaeum ships. The catalog cache,
blind and hinted solving, the pixel-space SIP distortion fit (sub-pixel on
13/14 bench frames), the opt-in precise profile and frame-to-frame
registration are all functional and held by a corpus gate against a frozen
baseline.

Released by tag from this repository; `rustafits` is a version dependency
(`1.9.1`), so the crate builds on its own. See `CHANGELOG.md`.
