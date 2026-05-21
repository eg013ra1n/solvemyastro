//! WCS solution with optional SIP distortion.
//!
//! Phase 0: the linear (CD-matrix + gnomonic TAN) transform is fully
//! implemented and usable. SIP polynomials are carried but evaluate to 0
//! until Phase 2 fits them — so the type behaves as a clean linear WCS
//! today and gains precision (sub-pixel, distortion) without an API change.

/// SIP forward/reverse distortion polynomial. `coeffs[i][j]` is the
/// coefficient of `u^i v^j`. Empty ⇒ identity (0 distortion).
#[derive(Clone, Debug, Default)]
pub struct SipCoefficients {
    pub order: u8,
    /// Triangular coefficient table indexed `[i][j]`, i+j ≤ order.
    pub coeffs: Vec<Vec<f64>>,
}

impl SipCoefficients {
    /// Distortion offset for intermediate pixel (u, v). Returns 0 until
    /// Phase 2 populates `coeffs`.
    pub fn evaluate(&self, u: f64, v: f64) -> f64 {
        let mut acc = 0.0;
        for (i, row) in self.coeffs.iter().enumerate() {
            for (j, &c) in row.iter().enumerate() {
                if c != 0.0 {
                    acc += c * u.powi(i as i32) * v.powi(j as i32);
                }
            }
        }
        acc
    }
}

/// Gnomonic (TAN) projection helpers. (ra,dec,ra0,dec0) in degrees;
/// tangent-plane (xi,eta) in degrees.
pub(crate) mod tan {
    pub fn sky_to_tangent(ra: f64, dec: f64, ra0: f64, dec0: f64) -> (f64, f64) {
        let (rar, decr) = (ra.to_radians(), dec.to_radians());
        let (ra0r, dec0r) = (ra0.to_radians(), dec0.to_radians());
        let d = rar - ra0r;
        let cos_c = dec0r.sin() * decr.sin() + dec0r.cos() * decr.cos() * d.cos();
        let xi = decr.cos() * d.sin() / cos_c;
        let eta =
            (dec0r.cos() * decr.sin() - dec0r.sin() * decr.cos() * d.cos()) / cos_c;
        (xi.to_degrees(), eta.to_degrees())
    }

    pub fn tangent_to_sky(xi_deg: f64, eta_deg: f64, ra0: f64, dec0: f64) -> (f64, f64) {
        let (xi, eta) = (xi_deg.to_radians(), eta_deg.to_radians());
        let (ra0r, dec0r) = (ra0.to_radians(), dec0.to_radians());
        let rho = (xi * xi + eta * eta).sqrt();
        if rho < 1e-15 {
            return (ra0, dec0);
        }
        let c = rho.atan();
        let dec = (c.cos() * dec0r.sin() + eta * c.sin() * dec0r.cos() / rho).asin();
        let ra = ra0r
            + (xi * c.sin())
                .atan2(rho * dec0r.cos() * c.cos() - eta * dec0r.sin() * c.sin());
        (ra.to_degrees().rem_euclid(360.0), dec.to_degrees())
    }
}

/// A complete WCS: linear CD + gnomonic, plus optional SIP distortion.
#[derive(Clone, Debug)]
pub struct WcsSolution {
    /// Reference pixel (0-based), typically image centre.
    pub crpix: (f64, f64),
    /// Sky at the reference pixel (RA, Dec) degrees.
    pub crval: (f64, f64),
    /// CD matrix (deg/px): `cd[0]=CD1_*`, `cd[1]=CD2_*`.
    pub cd: [[f64; 2]; 2],
    /// SIP forward (A, B) applied to pixel offsets before CD.
    pub sip_forward: Option<(SipCoefficients, SipCoefficients)>,
    /// SIP reverse (AP, BP) for sky→pixel.
    pub sip_reverse: Option<(SipCoefficients, SipCoefficients)>,
}

impl WcsSolution {
    pub fn pixel_to_sky(&self, x: f64, y: f64) -> (f64, f64) {
        let (mut u, mut v) = (x - self.crpix.0, y - self.crpix.1);
        if let Some((a, b)) = &self.sip_forward {
            let (du, dv) = (a.evaluate(u, v), b.evaluate(u, v));
            u += du;
            v += dv;
        }
        let xi = self.cd[0][0] * u + self.cd[0][1] * v;
        let eta = self.cd[1][0] * u + self.cd[1][1] * v;
        tan::tangent_to_sky(xi, eta, self.crval.0, self.crval.1)
    }

    pub fn sky_to_pixel(&self, ra: f64, dec: f64) -> (f64, f64) {
        let (xi, eta) = tan::sky_to_tangent(ra, dec, self.crval.0, self.crval.1);
        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        if det.abs() < 1e-30 {
            return (f64::NAN, f64::NAN);
        }
        let inv = 1.0 / det;
        let mut u = inv * (self.cd[1][1] * xi - self.cd[0][1] * eta);
        let mut v = inv * (-self.cd[1][0] * xi + self.cd[0][0] * eta);
        if let Some((ap, bp)) = &self.sip_reverse {
            let (du, dv) = (ap.evaluate(u, v), bp.evaluate(u, v));
            u += du;
            v += dv;
        }
        (u + self.crpix.0, v + self.crpix.1)
    }

    /// Pixel scale (arcsec/px), geometric mean of the two axes.
    pub fn pixel_scale_arcsec(&self) -> f64 {
        let sx = (self.cd[0][0].powi(2) + self.cd[1][0].powi(2)).sqrt();
        let sy = (self.cd[0][1].powi(2) + self.cd[1][1].powi(2)).sqrt();
        (sx * sy).sqrt() * 3600.0
    }

    /// Field rotation (deg, N through E).
    pub fn field_rotation_deg(&self) -> f64 {
        let mut d = (-self.cd[0][1]).atan2(self.cd[1][1]).to_degrees();
        if d < 0.0 {
            d += 360.0;
        }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_wcs_round_trips() {
        // 1.5"/px, no rotation, centre at (180, 40), 0-based crpix.
        let s = 1.5 / 3600.0;
        let w = WcsSolution {
            crpix: (2000.0, 1500.0),
            crval: (180.0, 40.0),
            cd: [[s, 0.0], [0.0, s]],
            sip_forward: None,
            sip_reverse: None,
        };
        let (ra, dec) = w.pixel_to_sky(2000.0, 1500.0);
        assert!((ra - 180.0).abs() < 1e-9 && (dec - 40.0).abs() < 1e-9);
        let (x, y) = w.sky_to_pixel(ra, dec);
        assert!((x - 2000.0).abs() < 1e-6 && (y - 1500.0).abs() < 1e-6);
        let (rx, ry) = w.sky_to_pixel(w.pixel_to_sky(2123.0, 1402.0).0, w.pixel_to_sky(2123.0, 1402.0).1);
        assert!((rx - 2123.0).abs() < 1e-4 && (ry - 1402.0).abs() < 1e-4);
        assert!((w.pixel_scale_arcsec() - 1.5).abs() < 1e-6);
    }
}
