# Catalog external-truth fixtures

These CSV files are ground-truth cone queries fetched from the ESA Gaia DR3
TAP service. `tests/catalog_truth.rs` compares them against the local
`smac_gaia` catalog to prove the catalog *data* (not just its file format)
matches Gaia DR3.

Each file is the verbatim CSV body of:

```
SELECT ra,dec,phot_g_mean_mag,pmra,pmdec FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag < 16
AND 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',<ra>,<dec>,<radius>))
```

for the region defined in `catalog_truth.rs::REGIONS`.

## Regenerating

The harness prints the exact `curl` command for any missing fixture — run
the test, copy the command. To regenerate all of them, delete the `*.csv`
files and run:

```
cargo test -p solvemyastro --test catalog_truth -- --ignored --nocapture
```

## After a RUWE-filtered catalog rebuild

When the catalog has been rebuilt with the `ruwe < 1.4` quality cut, flip
`RUWE_FILTERED` to `true` in `catalog_truth.rs` and re-fetch every fixture —
the printed ADQL then includes `AND ruwe < 1.4` so the truth set matches how
the catalog was built.

These files are small (public Gaia data, a few hundred rows total) and are
committed so the test runs offline and deterministically.
