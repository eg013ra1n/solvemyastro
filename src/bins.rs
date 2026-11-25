use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Read, Write};

/// On-disk star record: 14 bytes per star.
///
/// | Offset | Type | Field |
/// |--------|------|-------|
/// | 0      | f32  | ra (degrees, catalog epoch) |
/// | 4      | f32  | dec (degrees, catalog epoch) |
/// | 8      | u16  | mag (magnitude x 1000) |
/// | 10     | i16  | pmra (proper motion RA, 0.01 mas/yr) |
/// | 12     | i16  | pmdec (proper motion Dec, 0.01 mas/yr) |
pub const RECORD_SIZE: usize = 14;

/// A raw star record as stored on disk (a `healpix_NNNNNN.bin` entry).
#[derive(Clone, Debug, PartialEq)]
pub struct BinRecord {
    pub ra: f32,
    pub dec: f32,
    pub mag_raw: u16,
    pub pmra_raw: i16,
    pub pmdec_raw: i16,
}

impl BinRecord {
    pub fn mag(&self) -> f32 {
        self.mag_raw as f32 / 1000.0
    }

    pub fn pmra_mas_yr(&self) -> f64 {
        self.pmra_raw as f64 * 0.01
    }

    pub fn pmdec_mas_yr(&self) -> f64 {
        self.pmdec_raw as f64 * 0.01
    }

    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let ra = reader.read_f32::<LittleEndian>()?;
        let dec = reader.read_f32::<LittleEndian>()?;
        let mag_raw = reader.read_u16::<LittleEndian>()?;
        let pmra_raw = reader.read_i16::<LittleEndian>()?;
        let pmdec_raw = reader.read_i16::<LittleEndian>()?;
        Ok(BinRecord {
            ra,
            dec,
            mag_raw,
            pmra_raw,
            pmdec_raw,
        })
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_f32::<LittleEndian>(self.ra)?;
        writer.write_f32::<LittleEndian>(self.dec)?;
        writer.write_u16::<LittleEndian>(self.mag_raw)?;
        writer.write_i16::<LittleEndian>(self.pmra_raw)?;
        writer.write_i16::<LittleEndian>(self.pmdec_raw)?;
        Ok(())
    }

    /// Create from float values (for catalog conversion).
    pub fn from_values(ra: f32, dec: f32, mag: f32, pmra_mas_yr: f64, pmdec_mas_yr: f64) -> Self {
        BinRecord {
            ra,
            dec,
            mag_raw: (mag * 1000.0).round() as u16,
            pmra_raw: (pmra_mas_yr / 0.01).round().clamp(-32768.0, 32767.0) as i16,
            pmdec_raw: (pmdec_mas_yr / 0.01).round().clamp(-32768.0, 32767.0) as i16,
        }
    }
}

/// Read all star records from a byte slice (typically memory-mapped).
/// Stars are sorted by magnitude — returns records up to mag_limit.
pub fn read_records_until_mag(data: &[u8], mag_limit: f32) -> Vec<BinRecord> {
    let mut records = Vec::new();
    let mut offset = 0;

    while offset + RECORD_SIZE <= data.len() {
        let mut cursor = io::Cursor::new(&data[offset..offset + RECORD_SIZE]);
        match BinRecord::read_from(&mut cursor) {
            Ok(record) => {
                if record.mag() > mag_limit {
                    break; // sorted by mag — stop early
                }
                records.push(record);
            }
            Err(e) => {
                tracing::warn!(offset, error = %e, "star record decode failed, truncating read");
                break;
            }
        }
        offset += RECORD_SIZE;
    }

    records
}

/// File name of one HEALPix bin: `healpix_NNNNNN.bin`.
pub fn bin_file_name(pixel: u64) -> String {
    format!("healpix_{pixel:06}.bin")
}

/// Write star records to a writer, sorted by magnitude ascending. Records of
/// equal magnitude order by the rest of the record, so any input order gives
/// the same bytes.
pub fn write_records<W: Write>(writer: &mut W, records: &mut [BinRecord]) -> io::Result<()> {
    records.sort_by_key(|r| (r.mag_raw, r.ra.to_bits(), r.dec.to_bits(), r.pmra_raw, r.pmdec_raw));
    for record in records.iter() {
        record.write_to(writer)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let original = BinRecord::from_values(180.123, -45.678, 8.5, -12.34, 56.78);

        let mut buf = Vec::new();
        original.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), RECORD_SIZE);

        let mut cursor = io::Cursor::new(&buf);
        let read_back = BinRecord::read_from(&mut cursor).unwrap();

        assert!((read_back.ra - original.ra).abs() < 1e-5);
        assert!((read_back.dec - original.dec).abs() < 1e-5);
        assert_eq!(read_back.mag_raw, original.mag_raw);
        assert_eq!(read_back.pmra_raw, original.pmra_raw);
        assert_eq!(read_back.pmdec_raw, original.pmdec_raw);
    }

    #[test]
    fn mag_conversion() {
        let r = BinRecord::from_values(0.0, 0.0, 10.567, 0.0, 0.0);
        assert!((r.mag() - 10.567).abs() < 0.001);
    }

    #[test]
    fn proper_motion_conversion() {
        // i16 at 0.01 mas/yr resolution: range ±327.67 mas/yr
        let r = BinRecord::from_values(0.0, 0.0, 5.0, -123.45, 250.78);
        assert!((r.pmra_mas_yr() - (-123.45)).abs() < 0.02);
        assert!((r.pmdec_mas_yr() - 250.78).abs() < 0.02);
    }

    #[test]
    fn early_termination_on_mag() {
        let mut records = vec![
            BinRecord::from_values(10.0, 20.0, 5.0, 0.0, 0.0),
            BinRecord::from_values(11.0, 21.0, 8.0, 0.0, 0.0),
            BinRecord::from_values(12.0, 22.0, 11.0, 0.0, 0.0),
            BinRecord::from_values(13.0, 23.0, 14.0, 0.0, 0.0),
        ];

        let mut buf = Vec::new();
        write_records(&mut buf, &mut records).unwrap();

        let result = read_records_until_mag(&buf, 10.0);
        assert_eq!(result.len(), 2, "Should stop after mag 10.0");
    }

    #[test]
    fn equal_magnitudes_sort_by_the_rest_of_the_record() {
        let a = BinRecord::from_values(10.0, 5.0, 12.345, 0.0, 0.0);
        let b = BinRecord::from_values(9.0, 5.0, 12.345, 0.0, 0.0);
        let c = BinRecord::from_values(9.0, 4.0, 12.345, 0.0, 0.0);
        let d = BinRecord::from_values(9.0, 4.0, 12.345, 1.0, 0.0);
        let e = BinRecord::from_values(9.0, 4.0, 12.345, 0.0, 1.0);
        let mut one = vec![a.clone(), b.clone(), c.clone(), d.clone(), e.clone()];
        let mut two = vec![e, d, c, a, b];
        let (mut x, mut y) = (Vec::new(), Vec::new());
        write_records(&mut x, &mut one).unwrap();
        write_records(&mut y, &mut two).unwrap();
        assert_eq!(x, y); // any input order gives the same bytes
    }

    #[test]
    fn bin_file_name_is_six_digits() {
        assert_eq!(bin_file_name(42), "healpix_000042.bin");
        assert_eq!(bin_file_name(49151), "healpix_049151.bin");
    }
}
