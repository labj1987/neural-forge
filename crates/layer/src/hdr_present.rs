//! HDR10 swapchains on the finished-frame path (docs/NATIVE_BACKEND.md, "HDR on the finished frame (3.10.0)").
//!
//! A 10-bit swapchain in the HDR10 colour space (`HDR10_ST2084_EXT`) holds ST 2084 (PQ) code with BT.2020
//! primaries: absolute light, which the network (trained on display-referred SDR pictures) cannot take as it is.
//! The capture copies the frame raw (`proxy_format::PQ10_RGB`/`PQ10_BGR`); the model server converts it to an
//! 8-bit SDR picture (`shaders/native_post_hdr.comp`) and runs the network on that; the composition
//! (`shaders/compose_hdr.comp`) carries the network's change back onto the untouched HDR frame in linear light.
//!
//! The SDR picture, per pixel: PQ decode to nits (BT.2020), to BT.709, divided by the reference white
//! ([`reference_nits`], 203 nits by default, BT.2408's HDR reference white), the pre-upscaler encode's
//! per-channel shoulder (knee 0.75), the sRGB OETF, 8 bits. The composition: out = PQ(lin(base) + R x
//! (S⁻¹(answer) - S⁻¹(SDR(proxy)))), S⁻¹ the sRGB EOTF and the shoulder's inverse, R the reference white,
//! in linear BT.709 nits taken to BT.2020; a pixel whose answer equals SDR(proxy) keeps the frame's texel bit
//! for bit, and one whose SDR(proxy) peaks at the shoulder's top keeps it too (faded in from 0.954, as the
//! pre-upscaler decode keeps a clamped highlight).
//!
//! The idea of running the network on an SDR-mapped copy of the HDR frame and restoring its change onto the
//! untouched HDR frame comes from Veyra-NRVideo (ATTRIBUTION.md); ST 2084 and the BT.2020/BT.709 matrices are
//! public standards (SMPTE ST 2084, ITU-R BT.2087).

/// `off` keeps HDR10 swapchains out of the effect (presented untouched), as before the HDR path existed. A
/// testing aid.
pub(crate) const ENABLE_ENV: &str = "NEURAL_FORGE_HDR_PRESENT";

/// The reference white in nits: the HDR level the SDR picture's 1.0 (before the shoulder) stands for.
pub(crate) const REFERENCE_ENV: &str = "NEURAL_FORGE_HDR_REFERENCE_NITS";

/// BT.2408's HDR reference white (graphics white).
pub(crate) const DEFAULT_REFERENCE_NITS: f32 = 203.0;

/// The range a [`REFERENCE_ENV`] value is clamped to.
pub(crate) const MIN_REFERENCE_NITS: f32 = 80.0;
pub(crate) const MAX_REFERENCE_NITS: f32 = 1000.0;

/// `compose_hdr.comp`'s flags bit: keep the frame where SDR(proxy) sits at the shoulder's top.
pub(crate) const KEEP_CLAMPED: u32 = 1;

/// `compose_hdr.comp`'s flags: [`KEEP_CLAMPED`] unless the pre-upscaler decode's highlight protection is off
/// (`NEURAL_FORGE_PREUPSCALE_HIGHLIGHTS=off`, the same testing aid for both).
pub(crate) fn compose_flags() -> u32 {
    if crate::preupscale::hdr::highlight_flags() & crate::preupscale::hdr::KEEP_CLAMPED != 0 {
        KEEP_CLAMPED
    } else {
        0
    }
}

/// Whether `value` (of [`ENABLE_ENV`]) leaves the HDR path on: anything but `off`.
pub(crate) fn parse_enabled(value: Option<&str>) -> bool {
    value.map(str::trim) != Some("off")
}

/// Whether HDR10 swapchains get the network ([`ENABLE_ENV`]), read once.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = parse_enabled(neural_forge_protocol::env::var(ENABLE_ENV).as_deref());
        if !on {
            crate::log!("[layer] {ENABLE_ENV}=off: HDR10 swapchains present untouched");
        }
        on
    });
    *ON
}

/// `value` (of [`REFERENCE_ENV`]) as a reference white: unset or empty is the default, a number is clamped to
/// [`MIN_REFERENCE_NITS`]..[`MAX_REFERENCE_NITS`], anything else is an error.
pub(crate) fn parse_reference_nits(value: Option<&str>) -> Result<f32, String> {
    match value.map(str::trim) {
        None | Some("") => Ok(DEFAULT_REFERENCE_NITS),
        Some(v) => match v.parse::<f32>() {
            Ok(n) if n.is_finite() => Ok(n.clamp(MIN_REFERENCE_NITS, MAX_REFERENCE_NITS)),
            _ => Err(v.to_string()),
        },
    }
}

/// The reference white for this process ([`REFERENCE_ENV`], default [`DEFAULT_REFERENCE_NITS`]). Read once; the
/// model server's conversion and the composition use the same value.
pub(crate) fn reference_nits() -> f32 {
    static NITS: std::sync::LazyLock<f32> =
        std::sync::LazyLock::new(|| match parse_reference_nits(neural_forge_protocol::env::var(REFERENCE_ENV).as_deref()) {
            Ok(n) => {
                if n != DEFAULT_REFERENCE_NITS {
                    crate::log!("[layer] HDR10 on the finished frame: reference white {n} nits");
                }
                n
            }
            Err(other) => {
                crate::log!("[layer] {REFERENCE_ENV}={other:?} is not a number; using {DEFAULT_REFERENCE_NITS}");
                DEFAULT_REFERENCE_NITS
            }
        });
    *NITS
}

/// The CPU reference of the shaders' maths (`native_post_hdr.comp`, `compose_hdr.comp`), in double precision.
#[cfg(test)]
pub(crate) mod reference {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    /// ST 2084's peak.
    pub(crate) const PEAK_NITS: f64 = 10000.0;
    const KNEE: f64 = 0.75;
    const K: f64 = 5.770780;
    pub(crate) const CLAMPED: f64 = 0.999;
    pub(crate) const FADE_SRGB: f64 = 0.954;

    /// ITU-R BT.2087's linear BT.2020 to BT.709 matrix (rows: R, G, B).
    pub(crate) const BT2020_TO_BT709: [[f64; 3]; 3] = [[1.660491, -0.587641, -0.072850], [-0.124550, 1.132900, -0.008349], [-0.018151, -0.100579, 1.118730]];
    /// Its inverse, BT.709 to BT.2020.
    pub(crate) const BT709_TO_BT2020: [[f64; 3]; 3] = [[0.627404, 0.329283, 0.043313], [0.069097, 0.919540, 0.011362], [0.016391, 0.088013, 0.895595]];

    /// ST 2084's EOTF: code (0 to 1) to nits.
    pub(crate) fn pq_to_nits(e: f64) -> f64 {
        if e <= 0.0 {
            return 0.0;
        }
        let ep = e.min(1.0).powf(1.0 / M2);
        let y = (ep - C1).max(0.0) / (C2 - C3 * ep);
        y.powf(1.0 / M1) * PEAK_NITS
    }

    /// ST 2084's inverse EOTF: nits (clamped to 0..10000) to code.
    pub(crate) fn nits_to_pq(nits: f64) -> f64 {
        let y = (nits / PEAK_NITS).clamp(0.0, 1.0);
        let ym = y.powf(M1);
        ((C1 + C2 * ym) / (1.0 + C3 * ym)).powf(M2)
    }

    pub(crate) fn mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        m.map(|row| row[0] * v[0] + row[1] * v[1] + row[2] * v[2])
    }

    pub(crate) fn shoulder(v: f64) -> f64 {
        if v <= KNEE {
            v
        } else {
            KNEE + 0.25 * (1.0 - (-K * (v - KNEE)).exp())
        }
    }

    pub(crate) fn shoulder_inverse(y: f64) -> f64 {
        let y = y.min(1.0 - 1e-4);
        if y >= KNEE {
            KNEE - (1.0 - (y - KNEE) / 0.25).ln() / K
        } else {
            y
        }
    }

    pub(crate) fn srgb_oetf(c: f64) -> f64 {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    }

    pub(crate) fn srgb_eotf(s: f64) -> f64 {
        let s = s.clamp(0.0, 1.0);
        if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }

    /// The (R, G, B) codes and the alpha bits of a packed word.
    pub(crate) fn unpack(word: u32, b_low: bool) -> ([u32; 3], u32) {
        let c = [word & 1023, (word >> 10) & 1023, (word >> 20) & 1023];
        (if b_low { [c[2], c[1], c[0]] } else { c }, word >> 30)
    }

    pub(crate) fn pack(rgb: [u32; 3], alpha: u32, b_low: bool) -> u32 {
        let c = if b_low { [rgb[2], rgb[1], rgb[0]] } else { rgb };
        c[0] | (c[1] << 10) | (c[2] << 20) | (alpha << 30)
    }

    /// A word's light in linear BT.2020 nits.
    pub(crate) fn linear_2020(word: u32, b_low: bool) -> [f64; 3] {
        unpack(word, b_low).0.map(|c| pq_to_nits(f64::from(c) / 1023.0))
    }

    /// The SDR picture's value per channel (0 to 1, before the 8-bit rounding) of a word.
    pub(crate) fn sdr(word: u32, b_low: bool, reference: f64) -> [f64; 3] {
        mul(&BT2020_TO_BT709, linear_2020(word, b_low)).map(|l| srgb_oetf(shoulder(l.max(0.0) / reference)))
    }

    /// [`sdr`] in 8 bits, as the network is shown it.
    pub(crate) fn sdr8(word: u32, b_low: bool, reference: f64) -> [u8; 3] {
        sdr(word, b_low, reference).map(|v| (v * 255.0).round() as u8)
    }

    /// The composition of one pixel: `base` the frame presented now, `proxy` the frame the network was shown,
    /// `answer` its 8-bit answer (R, G, B).
    pub(crate) fn compose(base: u32, proxy: u32, answer: [u8; 3], b_low: bool, reference: f64) -> u32 {
        let s = sdr8(proxy, b_low, reference);
        if answer == s {
            return base;
        }
        let peak = f64::from(*s.iter().max().unwrap()) / 255.0;
        if peak >= CLAMPED {
            return base;
        }
        let fade = 1.0 - smoothstep(FADE_SRGB, CLAMPED, peak);
        let inv = |v: u8| shoulder_inverse(srgb_eotf(f64::from(v) / 255.0));
        let delta_709 = [0, 1, 2].map(|c| reference * (inv(answer[c]) - inv(s[c])) * fade);
        let delta_2020 = mul(&BT709_TO_BT2020, delta_709);
        let lin = linear_2020(base, b_low);
        let codes = [0, 1, 2].map(|c| (nits_to_pq((lin[c] + delta_2020[c]).clamp(0.0, PEAK_NITS)) * 1023.0).round() as u32);
        pack(codes, base >> 30, b_low)
    }

    pub(crate) fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// The PQ code (0 to 1023) of `nits`.
    pub(crate) fn code_of(nits: f64) -> u32 {
        (nits_to_pq(nits) * 1023.0).round() as u32
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::*;

    #[test]
    fn the_settings_parse_and_clamp() {
        assert!(parse_enabled(None));
        assert!(parse_enabled(Some("on")));
        assert!(!parse_enabled(Some("off")));
        assert!(!parse_enabled(Some(" off ")));
        assert_eq!(parse_reference_nits(None), Ok(203.0));
        assert_eq!(parse_reference_nits(Some("")), Ok(203.0));
        assert_eq!(parse_reference_nits(Some("300")), Ok(300.0));
        assert_eq!(parse_reference_nits(Some("10")), Ok(80.0));
        assert_eq!(parse_reference_nits(Some("5000")), Ok(1000.0));
        assert!(parse_reference_nits(Some("bright")).is_err());
        assert!(parse_reference_nits(Some("NaN")).is_err());
    }

    /// ST 2084's known points: code 0 is black, 1 is 10000 nits, 100 nits is code 0.5081 (520 of 1023), and the
    /// encode and decode undo each other.
    #[test]
    fn pq_round_trips_and_hits_the_standards_points() {
        assert_eq!(pq_to_nits(0.0), 0.0);
        assert!((pq_to_nits(1.0) - 10000.0).abs() < 1e-6);
        assert!((nits_to_pq(100.0) - 0.508_078).abs() < 1e-5, "{}", nits_to_pq(100.0));
        assert_eq!(code_of(100.0), 520);
        assert!((nits_to_pq(10000.0) - 1.0).abs() < 1e-9);
        for nits in [0.001, 0.1, 1.0, 50.0, 203.0, 1000.0, 4000.0, 9999.0] {
            assert!((pq_to_nits(nits_to_pq(nits)) - nits).abs() < nits * 1e-9, "{nits}");
        }
        for code in 0..1024u32 {
            assert_eq!(code_of(pq_to_nits(f64::from(code) / 1023.0)), code);
        }
    }

    #[test]
    fn the_gamut_matrices_undo_each_other_and_keep_white() {
        for v in [[1.0, 1.0, 1.0], [0.2, 0.7, 0.1], [5.0, 0.0, 3.0]] {
            let back = mul(&BT709_TO_BT2020, mul(&BT2020_TO_BT709, v));
            for c in 0..3 {
                assert!((back[c] - v[c]).abs() < 2e-5, "{v:?} -> {back:?}");
            }
        }
        let white = mul(&BT2020_TO_BT709, [1.0, 1.0, 1.0]);
        assert!(white.iter().all(|w| (w - 1.0).abs() < 1e-5), "{white:?}");
    }

    /// The SDR picture: black is 0, the reference white (203 nits) reaches the shoulder's input 1.0, mid-tones
    /// below the knee are the sRGB of nits / 203, and the brightest highlight ends on the shoulder's top.
    #[test]
    fn the_sdr_picture_puts_the_reference_white_at_1() {
        let grey = |nits: f64| {
            let c = code_of(nits);
            pack([c; 3], 3, false)
        };
        assert_eq!(sdr(grey(0.0), false, 203.0), [0.0; 3]);
        // The word's code rounds the light: compare with the light the code stands for.
        let at_white = pq_to_nits(f64::from(code_of(203.0)) / 1023.0);
        assert!((at_white / 203.0 - 1.0).abs() < 0.01, "203 nits within a code: {at_white}");
        let v = sdr(grey(203.0), false, 203.0)[0];
        assert!((v - srgb_oetf(shoulder(at_white / 203.0))).abs() < 1e-6);
        assert!((shoulder_inverse(srgb_eotf(v)) - at_white / 203.0).abs() < 1e-3);
        let mid = pq_to_nits(f64::from(code_of(50.0)) / 1023.0);
        assert!((sdr(grey(50.0), false, 203.0)[1] - srgb_oetf(mid / 203.0)).abs() < 1e-6);
        assert_eq!(sdr8(grey(10000.0), false, 203.0), [255; 3]);
        // A higher reference white darkens the picture.
        assert!(sdr(grey(50.0), false, 400.0)[0] < sdr(grey(50.0), false, 203.0)[0]);
    }

    #[test]
    fn both_channel_orders_read_the_same_light() {
        let rgb = pack([600, 300, 100], 2, false);
        let bgr = pack([600, 300, 100], 2, true);
        assert_eq!(rgb & 1023, 600);
        assert_eq!(bgr & 1023, 100);
        assert_eq!(unpack(rgb, false), ([600, 300, 100], 2));
        assert_eq!(unpack(bgr, true), ([600, 300, 100], 2));
        assert_eq!(sdr(rgb, false, 203.0), sdr(bgr, true, 203.0));
    }

    /// The composition: no edit keeps the frame bit for bit; a brightening edit raises the light by the
    /// reference white times the edit in linear SDR units; a highlight at the shoulder's top keeps the frame.
    #[test]
    fn the_composition_carries_the_change_in_linear_light() {
        let c = code_of(40.0);
        let base = pack([c, c, c], 1, false);
        let s = sdr8(base, false, 203.0);
        assert_eq!(compose(base, base, s, false, 203.0), base);
        let brighter = s.map(|v| v + 10);
        let out = compose(base, base, brighter, false, 203.0);
        assert_eq!(out >> 30, 1, "alpha kept");
        let inv = |v: u8| shoulder_inverse(srgb_eotf(f64::from(v) / 255.0));
        let expected = code_of(pq_to_nits(f64::from(c) / 1023.0) + 203.0 * (inv(brighter[0]) - inv(s[0])));
        let got = unpack(out, false).0;
        assert!(got.iter().all(|&g| g.abs_diff(expected) <= 1), "{got:?} vs {expected}");
        assert!(got[0] > c + 10, "a visible raise: {c} -> {}", got[0]);
        let sun = pack([code_of(4000.0); 3], 3, false);
        assert_eq!(sdr8(sun, false, 203.0), [255; 3]);
        assert_eq!(compose(sun, sun, [200; 3], false, 203.0), sun);
    }
}
