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

## Status

In use: solvemyastro is the only solver Athenaeum ships. The catalog cache,
blind and hinted solving, the pixel-space SIP distortion fit (sub-pixel on
13/14 bench frames), the opt-in precise profile and frame-to-frame
registration are all functional and held by a corpus gate against a frozen
baseline.

Released by tag from this repository; `rustafits` is a path dependency, so the
crate builds inside the Athenaeum workspace layout and is not published to
crates.io. See `CHANGELOG.md`.
