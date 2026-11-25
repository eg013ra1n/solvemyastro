//! RAM-safe streaming binner. The full catalog (hundreds of millions of
//! records) is never held in memory: each chunk is grouped by depth-6 HEALPix
//! pixel and appended to a per-pixel scratch file (`p_NNNNNN.raw`);
//! [`HealpixBinner::finalize`] then makes one pass per pixel, reads it back and
//! writes `healpix_NNNNNN.bin` through [`crate::bins::write_records`]. That
//! writer sorts by a total key, so the bytes do not depend on the order the
//! chunks arrived in.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::bins::{self, BinRecord, RECORD_SIZE};
use crate::healpix;

pub struct HealpixBinner {
    scratch_dir: PathBuf,
}

impl HealpixBinner {
    /// Use `scratch_dir` (created when missing) for the per-pixel scratch files.
    pub fn open(scratch_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(scratch_dir)
            .with_context(|| format!("create scratch dir {}", scratch_dir.display()))?;
        Ok(Self { scratch_dir: scratch_dir.to_path_buf() })
    }

    /// Append one chunk's records to their per-pixel scratch files. Only this
    /// chunk is held in memory; a scratch file is opened and closed per pixel.
    pub fn push_tile(&self, records: &[BinRecord]) -> Result<()> {
        let mut by_pixel: HashMap<u64, Vec<&BinRecord>> = HashMap::new();
        for r in records {
            let pixel = healpix::sky_to_pixel(r.ra as f64, r.dec as f64);
            by_pixel.entry(pixel).or_default().push(r);
        }
        for (pixel, recs) in by_pixel {
            let path = self.scratch_dir.join(format!("p_{pixel:06}.raw"));
            let f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .with_context(|| format!("append scratch {}", path.display()))?;
            let mut w = BufWriter::new(f);
            for r in recs {
                r.write_to(&mut w).with_context(|| format!("write scratch {}", path.display()))?;
            }
            w.flush().with_context(|| format!("flush scratch {}", path.display()))?;
        }
        Ok(())
    }

    /// Read every pixel scratch file, sort and write it as
    /// `out_dir/healpix_NNNNNN.bin`, delete the scratch, and return the total
    /// record count. A scratch file that is not a whole number of records is an
    /// error.
    pub fn finalize(self, out_dir: &Path) -> Result<usize> {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("create catalog dir {}", out_dir.display()))?;
        let mut total = 0usize;
        for entry in std::fs::read_dir(&self.scratch_dir)
            .with_context(|| format!("list scratch {}", self.scratch_dir.display()))?
        {
            let path = entry
                .with_context(|| format!("read an entry of scratch {}", self.scratch_dir.display()))?
                .path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let Some(pixel_str) = name.strip_prefix("p_").and_then(|s| s.strip_suffix(".raw")) else {
                continue;
            };
            let pixel: u64 = pixel_str
                .parse()
                .with_context(|| format!("parse scratch pixel id in {}", path.display()))?;

            let data = std::fs::read(&path).with_context(|| format!("read scratch {}", path.display()))?;
            if data.len() % RECORD_SIZE != 0 {
                anyhow::bail!(
                    "scratch {} holds {} bytes, not a whole number of {RECORD_SIZE}-byte records",
                    path.display(),
                    data.len()
                );
            }
            let mut records = Vec::with_capacity(data.len() / RECORD_SIZE);
            for chunk in data.chunks_exact(RECORD_SIZE) {
                let rec = BinRecord::read_from(&mut &chunk[..])
                    .with_context(|| format!("decode scratch {}", path.display()))?;
                records.push(rec);
            }
            total += records.len();

            let out = out_dir.join(bins::bin_file_name(pixel));
            let mut w = BufWriter::new(
                std::fs::File::create(&out).with_context(|| format!("create {}", out.display()))?,
            );
            bins::write_records(&mut w, &mut records).with_context(|| format!("write {}", out.display()))?;
            w.flush().with_context(|| format!("flush {}", out.display()))?;
            std::fs::remove_file(&path).with_context(|| format!("remove scratch {}", path.display()))?;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(ra: f32, dec: f32, mag: f32) -> BinRecord {
        BinRecord::from_values(ra, dec, mag, 0.0, 0.0)
    }

    #[test]
    fn bins_are_the_same_for_any_chunking_and_order() {
        let stars: Vec<BinRecord> = (0..200)
            .map(|i| rec(10.0 + (i % 7) as f32 * 0.01, 20.0 + (i % 5) as f32 * 0.01, 12.0 + (i % 3) as f32))
            .collect();
        let run = |chunks: Vec<Vec<BinRecord>>| {
            let tmp = tempfile::tempdir().unwrap();
            let binner = HealpixBinner::open(&tmp.path().join("s")).unwrap();
            for c in &chunks {
                binner.push_tile(c).unwrap();
            }
            let total = binner.finalize(&tmp.path().join("o")).unwrap();
            let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(tmp.path().join("o"))
                .unwrap()
                .map(|e| {
                    let p = e.unwrap().path();
                    (p.file_name().unwrap().to_str().unwrap().to_string(), std::fs::read(&p).unwrap())
                })
                .collect();
            files.sort();
            assert_eq!(std::fs::read_dir(tmp.path().join("s")).unwrap().count(), 0, "scratch consumed");
            (total, files)
        };
        let forward = run(vec![stars.clone()]);
        let mut reversed = stars.clone();
        reversed.reverse();
        let split = run(vec![reversed[..60].to_vec(), reversed[60..].to_vec()]);
        assert_eq!(forward.0, 200);
        assert_eq!(forward, split);
        // Sorted by magnitude within each bin.
        for (_, bytes) in &forward.1 {
            let recs = bins::read_records_until_mag(bytes, 99.0);
            assert!(recs.windows(2).all(|w| w[0].mag_raw <= w[1].mag_raw));
        }
    }

    #[test]
    fn a_torn_scratch_file_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let binner = HealpixBinner::open(&tmp.path().join("s")).unwrap();
        std::fs::write(tmp.path().join("s").join("p_000001.raw"), [0u8; RECORD_SIZE + 3]).unwrap();
        assert!(binner.finalize(&tmp.path().join("o")).is_err());
    }
}
