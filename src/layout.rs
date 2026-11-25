//! Where the installed catalog lives: the density-tier directories and the
//! quad-index pack directories under a catalog root.
//!
//! The plate-solver consumes an additive stack of disjoint density tiers,
//! installed as `tier_<density>/stars.smac` under the catalog root. This
//! module finds those dirs (ascending density = base to deepest) so the caller
//! can open each as a `StarCache` and pass `Caches::layered(..)` to the solver.
//! A legacy single `stars.smac` directly under the root is supported as a
//! 1-layer fallback so old installs keep solving.

use std::path::{Path, PathBuf};

/// The star cache file inside a tier (or a legacy catalog root).
pub const STARS_FILE: &str = "stars.smac";

/// Directory name of the tier with `density` stars per square degree.
pub fn tier_dir_name(density: u32) -> String {
    format!("tier_{density}")
}

/// Suffix of the folder a tier is built in. It is renamed to the tier's own
/// name only when the build has finished, so a killed build leaves nothing
/// that looks like a tier. Nothing in this crate reads such a folder.
pub const PARTIAL_SUFFIX: &str = ".partial";

/// Directory name a tier of `density` stars per square degree is built in.
pub fn tier_partial_dir_name(density: u32) -> String {
    format!("{}{PARTIAL_SUFFIX}", tier_dir_name(density))
}

/// Ordered `tier_<density>/` dirs (ascending density) under `catalog_root`.
///
/// Falls back to `[catalog_root]` when no tiers exist but a legacy
/// `stars.smac` is present directly under the root. Returns an empty `Vec`
/// when neither tiers nor a legacy cache are found (caller surfaces the
/// "catalog not installed" error).
pub fn discover_layers(catalog_root: &Path) -> Vec<PathBuf> {
    let mut tiers: Vec<(u32, PathBuf)> = match std::fs::read_dir(catalog_root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_str()?.to_string();
                let density: u32 = name.strip_prefix("tier_")?.parse().ok()?;
                if e.path().join(STARS_FILE).is_file() {
                    Some((density, e.path()))
                } else {
                    None
                }
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    tiers.sort_by_key(|(d, _)| *d);
    if !tiers.is_empty() {
        return tiers.into_iter().map(|(_, p)| p).collect();
    }
    if catalog_root.join(STARS_FILE).is_file() {
        return vec![catalog_root.to_path_buf()];
    }
    Vec::new()
}

/// Pack directories under `root` that hold a `quads.smaq`, sorted by name.
/// Whether the file opens is for the caller to find out.
pub fn discover_index_packs(root: &Path) -> Vec<PathBuf> {
    let mut packs: Vec<(String, PathBuf)> = match std::fs::read_dir(root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_str()?.to_string();
                if e.path().join(crate::index::INDEX_FILE).is_file() {
                    Some((name, e.path()))
                } else {
                    None
                }
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    packs.sort_by(|a, b| a.0.cmp(&b.0));
    packs.into_iter().map(|(_, p)| p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_tiers_in_density_order_else_legacy() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        for d in ["tier_2000", "tier_500", "tier_8000"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
            std::fs::write(root.join(d).join(STARS_FILE), b"x").unwrap();
        }
        // A `tier_*` dir without stars.smac must be ignored.
        std::fs::create_dir_all(root.join("tier_9999")).unwrap();
        let got = discover_layers(root);
        let names: Vec<_> = got
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["tier_500", "tier_2000", "tier_8000"]);

        // Legacy fallback: a bare stars.smac, no tiers.
        let leg = tempfile::tempdir().unwrap();
        std::fs::write(leg.path().join(STARS_FILE), b"x").unwrap();
        assert_eq!(discover_layers(leg.path()), vec![leg.path().to_path_buf()]);

        // Neither tiers nor legacy -> empty.
        let empty = tempfile::tempdir().unwrap();
        assert!(discover_layers(empty.path()).is_empty());
    }

    #[test]
    fn index_packs_are_discovered_by_name() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("catalogs").join("smac_index");
        assert!(discover_index_packs(&root).is_empty());
        for (pack, bytes) in [("narrow", &b"yy"[..]), ("base", &b"x"[..])] {
            std::fs::create_dir_all(root.join(pack)).unwrap();
            std::fs::write(root.join(pack).join("quads.smaq"), bytes).unwrap();
        }
        // A dir without quads.smaq, and a stray file, are ignored.
        std::fs::create_dir_all(root.join("empty")).unwrap();
        std::fs::write(root.join("manifest.json"), b"{}").unwrap();
        let names: Vec<_> = discover_index_packs(&root)
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["base", "narrow"]);
    }

    #[test]
    fn a_partial_tier_folder_is_never_a_tier() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        // A killed build-tiers leaves the folder it was building in, with a
        // star cache that opens and is not whole.
        let partial = root.join(tier_partial_dir_name(500));
        std::fs::create_dir_all(&partial).unwrap();
        std::fs::write(partial.join(STARS_FILE), b"x").unwrap();
        assert!(discover_layers(root).is_empty(), "only a partial folder is no catalog");

        // Next to a real tier it is not listed either.
        std::fs::create_dir_all(root.join("tier_2000")).unwrap();
        std::fs::write(root.join("tier_2000").join(STARS_FILE), b"x").unwrap();
        let names: Vec<_> = discover_layers(root)
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(names, vec!["tier_2000"]);
        assert_eq!(tier_partial_dir_name(500), "tier_500.partial");
    }

    #[test]
    fn the_names_are_the_installed_layout() {
        assert_eq!(tier_dir_name(500), "tier_500");
        assert_eq!(STARS_FILE, "stars.smac");
    }
}
