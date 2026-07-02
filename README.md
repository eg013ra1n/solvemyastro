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

Phased build (see the implementation plan). Phases 0–2 complete: catalog
cache, blind plate solver, and pixel-space SIP distortion fit are
functional and corpus-gated (sub-pixel SIP on 13/14 bench frames).
Next: Athenaeum integration (Phase 3).
