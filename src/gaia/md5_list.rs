//! The archive's `_MD5SUM.txt`: the file order and the checksums.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result};
use md5::{Digest, Md5};

/// One entry of the checksum list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ListedFile {
    /// `GaiaSource_NNNNNN-NNNNNN.csv.gz`.
    pub name: String,
    /// 32 characters, lowercase hex.
    pub md5_hex: String,
}

/// Parse the body of `_MD5SUM.txt`. Each line is `<md5hex>  <filename>` (one or
/// more whitespace characters between them; a trailing `\r` is trimmed). Lines
/// that do not fit are skipped. Only `*.csv.gz` names are kept: the list also
/// names `_disclaimer.txt` and the like.
pub fn parse_md5_list(text: &str) -> Vec<ListedFile> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(md5), Some(name)) = (parts.next(), parts.next()) else {
            continue;
        };
        if md5.len() != 32 || !md5.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        }
        if !name.ends_with(".csv.gz") {
            continue;
        }
        out.push(ListedFile { name: name.to_string(), md5_hex: md5.to_ascii_lowercase() });
    }
    out
}

/// Read and parse a checksum list file. An empty result is an error.
pub fn read_md5_list(path: &Path) -> Result<Vec<ListedFile>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read checksum list {}", path.display()))?;
    let files = parse_md5_list(&text);
    if files.is_empty() {
        anyhow::bail!("checksum list {} names no .csv.gz file", path.display());
    }
    Ok(files)
}

/// Stream a file through MD5 and return the lowercase hex digest.
pub fn md5_of_file(path: &Path) -> Result<String> {
    let mut f = File::open(path).with_context(|| format!("open {} for md5", path.display()))?;
    let mut hasher = Md5::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).with_context(|| format!("read {} for md5", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex_lower(&hasher.finalize()))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xF) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_list_lines() {
        let body = "\
# header comment, ignore
d41d8cd98f00b204e9800998ecf8427e  GaiaSource_000000-003111.csv.gz
ABCDEF0123456789abcdef0123456789  GaiaSource_003112-005263.csv.gz
not-a-hash _disclaimer.txt
00112233445566778899aabbccddeeff  _disclaimer.txt
00112233445566778899aabbccddeeff  GaiaSource_005264-006601.csv.gz
";
        let v = parse_md5_list(body);
        assert_eq!(v.len(), 3, "got: {v:?}");
        assert_eq!(v[0].name, "GaiaSource_000000-003111.csv.gz");
        assert_eq!(v[0].md5_hex, "d41d8cd98f00b204e9800998ecf8427e");
        assert!(v.iter().all(|f| f.name.ends_with(".csv.gz")));
        assert_eq!(v[1].md5_hex, "abcdef0123456789abcdef0123456789", "hex is lower-cased");
    }

    #[test]
    fn parses_crlf_lines() {
        let body = "d41d8cd98f00b204e9800998ecf8427e  a.csv.gz\r\n00112233445566778899aabbccddeeff  b.csv.gz\r\n";
        let v = parse_md5_list(body);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].name, "a.csv.gz");
        assert_eq!(v[1].name, "b.csv.gz");
        assert_eq!(v[1].md5_hex, "00112233445566778899aabbccddeeff");
    }

    #[test]
    fn md5_of_file_matches_known_digest() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("e.bin");
        std::fs::write(&p, b"").unwrap();
        assert_eq!(md5_of_file(&p).unwrap(), "d41d8cd98f00b204e9800998ecf8427e");
        let p2 = tmp.path().join("abc.bin");
        std::fs::write(&p2, b"abc").unwrap();
        assert_eq!(md5_of_file(&p2).unwrap(), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn list_file_round_trip_and_empty_list_is_an_error() {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("_MD5SUM.txt");
        std::fs::write(&p, "d41d8cd98f00b204e9800998ecf8427e  a.csv.gz\n").unwrap();
        let back = read_md5_list(&p).unwrap();
        assert_eq!(back, vec![ListedFile { name: "a.csv.gz".into(), md5_hex: "d41d8cd98f00b204e9800998ecf8427e".into() }]);
        std::fs::write(&p, "nothing useful\n").unwrap();
        assert!(read_md5_list(&p).is_err());
    }
}
