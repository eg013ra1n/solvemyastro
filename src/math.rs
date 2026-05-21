//! Shared numerical helpers used across modules.

/// Great-circle (angular) separation between two sky positions, in degrees.
///
/// Numerically stable across all separations including antipode and pole
/// (Vincenty form: `atan2(|a×b|, a·b)` rather than `acos(a·b)` which loses
/// precision for small angles). Inputs are degrees, output is degrees in
/// `[0, 180]`.
#[inline]
pub fn great_circle_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let dra = (ra1 - ra2).to_radians();
    let lat1 = dec1.to_radians();
    let lat2 = dec2.to_radians();
    let (s1, c1) = lat1.sin_cos();
    let (s2, c2) = lat2.sin_cos();
    let (sd, cd) = dra.sin_cos();
    let a = (c2 * sd).hypot(c1 * s2 - s1 * c2 * cd);
    let b = s1 * s2 + c1 * c2 * cd;
    a.atan2(b).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::great_circle_deg;

    #[test]
    fn identical_points_yield_zero() {
        assert!(great_circle_deg(123.4, -56.7, 123.4, -56.7).abs() < 1e-10);
    }

    #[test]
    fn antipodes_yield_180() {
        let d = great_circle_deg(10.0, 20.0, 190.0, -20.0);
        assert!((d - 180.0).abs() < 1e-9, "expected 180, got {d}");
    }

    #[test]
    fn equatorial_one_degree() {
        let d = great_circle_deg(0.0, 0.0, 1.0, 0.0);
        assert!((d - 1.0).abs() < 1e-12, "expected 1.0, got {d}");
    }

    #[test]
    fn symmetry() {
        let d12 = great_circle_deg(83.78, -5.33, 86.68, 0.04);
        let d21 = great_circle_deg(86.68, 0.04, 83.78, -5.33);
        assert!((d12 - d21).abs() < 1e-12);
    }

    #[test]
    fn pole_to_pole() {
        let d = great_circle_deg(0.0, 90.0, 0.0, -90.0);
        assert!((d - 180.0).abs() < 1e-9);
    }
}
