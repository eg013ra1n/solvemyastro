# solvemyastro

Standalone astrometric plate solver. Produces **precise WCS including SIP
distortion** in-process — no external binary, no subprocess. Depends only
on `rustafits` (`astroimage`) for FITS/XISF decode + raw star detection.
Used by Athenaeum as a library and runnable as a standalone CLI.

## CLI

```
solvemyastro solve <image.fits> --catalog <cache-dir> [--ra --dec --fov|--scale --radius] [--json]
solvemyastro build-cache --from <gaia dir> --out <cache-dir>
solvemyastro cache-info <cache-dir>
```

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

Synthetic fields from 35 mm to 4000 mm, rendered from the same catalog with
the true solution in their headers:

```
cargo run --release --example synth_fields -- --tiers <dest>/catalogs --out <synth>
SOLVEMYASTRO_SYNTH_DIR=<synth> SOLVEMYASTRO_TIER_DIR=<dest>/catalogs \
  cargo test --release --test synth_bench -- --ignored --nocapture   # SYNTH_MODE=hinted|position|scale|blind
```

## Status

In use: solvemyastro is the only solver Athenaeum ships. The catalog cache,
blind and hinted solving, the pixel-space SIP distortion fit (sub-pixel on
13/14 bench frames), the opt-in precise profile and frame-to-frame
registration are all functional and held by a corpus gate against a frozen
baseline.

Released by tag from this repository; `rustafits` is a path dependency, so the
crate builds inside the Athenaeum workspace layout and is not published to
crates.io. See `CHANGELOG.md`.
