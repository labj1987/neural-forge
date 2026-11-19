//! The self-test: a synthetic frame series with a known edit, scored the way real runs are.
//!
//! The random numbers are Python's (MT19937 with `random.Random(1234)`'s seeding, `randbytes`, `randrange`,
//! `random`), and the bicubic resize is Pillow's, bit for bit. Pillow's Gaussian blur and ellipse rasteriser are
//! approximated (a true Gaussian kernel, a centre-sampled ellipse), so the synthetic scene, and with it the printed
//! scores, are close to the script's but not bit-identical; the checks are the script's.

use super::{compare_runs, Fail, BLOCK, EXCLUDE_TOP, MAGIC, STAMP_BITS};
use crate::hdr_encode::pil::{self, Filter, Rgb};
use crate::hdr_encode::pyfmt::fmt_f;

struct Mt {
    mt: [u32; 624],
    idx: usize,
}

impl Mt {
    /// `random.Random(seed)` for a non-negative seed below 2^32.
    fn new(seed: u32) -> Mt {
        let mut mt = [0u32; 624];
        mt[0] = 19650218;
        for i in 1..624 {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        let key = [seed];
        let (mut i, mut j) = (1usize, 0usize);
        for _ in 0..624.max(key.len()) {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                mt[0] = mt[623];
                i = 1;
            }
        }
        mt[0] = 0x8000_0000;
        Mt { mt, idx: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for k in 0..624 {
                let y = (self.mt[k] & 0x8000_0000) | (self.mt[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.mt[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.mt[k] = v;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `getrandbits(k)` for k <= 32.
    fn bits(&mut self, k: u32) -> u32 {
        self.next_u32() >> (32 - k)
    }

    /// `randbytes(n)`: the little-endian words of `getrandbits(8n)`.
    fn randbytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let rem = n - out.len();
            if rem >= 4 {
                out.extend_from_slice(&self.next_u32().to_le_bytes());
            } else {
                let v = self.next_u32() >> (32 - 8 * rem as u32);
                out.extend_from_slice(&v.to_le_bytes()[..rem]);
            }
        }
        out
    }

    /// `randrange(n)`
    fn below(&mut self, n: u32) -> u32 {
        let k = 32 - n.leading_zeros();
        loop {
            let r = self.bits(k);
            if r < n {
                return r;
            }
        }
    }

    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        lo + self.below(hi - lo)
    }

    fn random(&mut self) -> f64 {
        let (a, b) = ((self.next_u32() >> 5) as f64, (self.next_u32() >> 6) as f64);
        (a * 67108864.0 + b) / 9007199254740992.0
    }
}

type Plane = Vec<f32>;

/// A true Gaussian blur of an 8-bit plane (Pillow's is a box-blur approximation of one).
fn gaussian(p: &[u8], w: usize, h: usize, sigma: f64) -> Vec<u8> {
    let r = (3.0 * sigma).ceil() as i64;
    let k: Vec<f64> = (-r..=r).map(|i| (-((i * i) as f64) / (2.0 * sigma * sigma)).exp()).collect();
    let norm: f64 = k.iter().sum();
    let at = |x: i64, y: i64| p[y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize] as f64;
    let mut tmp = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            tmp[y * w + x] = k.iter().enumerate().map(|(i, kv)| kv * at(x as i64 + i as i64 - r, y as i64)).sum::<f64>() / norm;
        }
    }
    let at2 = |x: i64, y: i64| tmp[y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize];
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = (k.iter().enumerate().map(|(i, kv)| kv * at2(x as i64, y as i64 + i as i64 - r)).sum::<f64>() / norm + 0.5) as u8;
        }
    }
    out
}

fn fill_rect(img: &mut Rgb, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 3], inside: impl Fn(i64, i64) -> bool) {
    for y in y0.max(0)..=y1.min(img.h as i64 - 1) {
        for x in x0.max(0)..=x1.min(img.w as i64 - 1) {
            if inside(x, y) {
                let o = (y as usize * img.w + x as usize) * 3;
                img.data[o..o + 3].copy_from_slice(&c);
            }
        }
    }
}

pub(super) fn stamp(img: &mut Rgb, index: u32) {
    let mut bits: Vec<u8> = MAGIC.to_vec();
    bits.extend((0..STAMP_BITS).map(|i| ((index >> (STAMP_BITS - 1 - i)) & 1) as u8));
    let parity = bits[MAGIC.len()..].iter().map(|&b| b as usize).sum::<usize>() % 2;
    bits.push(parity as u8);
    for (i, b) in bits.iter().enumerate() {
        let c = if *b != 0 { 255 } else { 0 };
        fill_rect(img, (i * BLOCK) as i64, 0, (i * BLOCK + BLOCK - 1) as i64, BLOCK as i64 - 1, [c; 3], |_, _| true);
    }
}

fn synth_original(rng: &mut Mt, w: usize, h: usize, index: u32) -> Result<Rgb, Fail> {
    let mut layers: Vec<Vec<u8>> = Vec::new();
    for _ in 0..3 {
        let small = rng.randbytes((w / 4) * (h / 4));
        let noise = pil::resize_u8(&small, 1, w / 4, h / 4, w, h, Filter::Bicubic);
        let fine = gaussian(&rng.randbytes(w * h), w, h, 0.8);
        // Image.blend(noise, fine, 0.35)
        let alpha = 0.35f32 as f64;
        layers.push(noise.iter().zip(&fine).map(|(&a, &b)| (a as i32 as f64 + alpha * (b as i32 - a as i32) as f64) as u8).collect());
    }
    let mut img = Rgb::filled(w, h, [0, 0, 0]);
    for i in 0..w * h {
        for c in 0..3 {
            img.data[i * 3 + c] = layers[c][i];
        }
    }
    for _ in 0..12 {
        let x = rng.below(w as u32) as i64;
        let y = rng.range(EXCLUDE_TOP as u32, h as u32) as i64;
        let r = rng.range(6, 30) as i64;
        let c = [rng.below(256) as u8, rng.below(256) as u8, rng.below(256) as u8];
        if rng.random() < 0.5 {
            fill_rect(&mut img, x, y, x + r, y + r / 2, c, |_, _| true);
        } else {
            let (cx, cy, rad) = (x as f64 + (r + 1) as f64 / 2.0, y as f64 + (r + 1) as f64 / 2.0, (r + 1) as f64 / 2.0);
            fill_rect(&mut img, x, y, x + r, y + r, c, |i, j| ((i as f64 + 0.5 - cx) / rad).powi(2) + ((j as f64 + 0.5 - cy) / rad).powi(2) <= 1.0);
        }
    }
    // A flat area.
    fill_rect(&mut img, 0, (h * 3 / 4) as i64, (w / 3) as i64, h as i64, [128, 128, 128], |_, _| true);
    // Keep away from 0/255 so a +-40 edit never clips.
    for v in &mut img.data {
        *v = (48 + *v as u32 * 160 / 255) as u8;
    }
    stamp(&mut img, index);
    Ok(img)
}

fn planes(img: &Rgb) -> [Plane; 3] {
    std::array::from_fn(|c| (0..img.w * img.h).map(|i| img.data[i * 3 + c] as f32).collect())
}

fn apply_edit(original: &Rgb, edits: &[Plane; 3]) -> Rgb {
    let mut out = original.clone();
    for i in 0..original.w * original.h {
        for c in 0..3 {
            let v = original.data[i * 3 + c] as f32 + edits[c][i];
            out.data[i * 3 + c] = if v <= 0.0 {
                0
            } else if v >= 255.0 {
                255
            } else {
                v as u8
            };
        }
    }
    out
}

/// A structured, content-dependent 'model edit': local contrast boost plus a colour tint.
fn model_edit(original: &Rgb) -> [Plane; 3] {
    let (w, h) = (original.w, original.h);
    std::array::from_fn(|k| {
        let ch: Vec<u8> = (0..w * h).map(|i| original.data[i * 3 + k]).collect();
        let blurred = gaussian(&ch, w, h, 2.0);
        let tint: f32 = if k == 0 { 3.0 } else { -2.0 };
        ch.iter().zip(&blurred).map(|(&o, &b)| (o as f32 - b as f32) * 0.8f32 + tint).collect()
    })
}

/// `ImageChops.offset`: the data moved by (dx, dy), wrapping around the edges.
fn offset(p: &[f32], w: usize, h: usize, dx: usize, dy: usize) -> Plane {
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            out[((y + dy) % h) * w + (x + dx) % w] = p[y * w + x];
        }
    }
    out
}

fn write_pair(dir: &std::path::Path, seq: usize, original: &Rgb, composited: &Rgb) -> Result<(), Fail> {
    original.save(&dir.join(format!("{seq:06}-original.png")))?;
    composited.save(&dir.join(format!("{seq:06}-composited.png")))
}

pub fn self_test() -> Result<u8, Fail> {
    let mut rng = Mt::new(1234);
    let (w, h) = (384usize, 216usize);
    let frames = 6usize;
    let first_index = 1000u32;
    let tmp = std::env::temp_dir().join(format!("agreement-selftest-{}", std::process::id()));
    let names = ["ref", "same", "shifted1", "shifted", "noise", "partial"];
    let dirs: Vec<std::path::PathBuf> = names.iter().map(|n| tmp.join(n)).collect();
    for d in &dirs {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let result = (|| -> Result<u8, Fail> {
        for f in 0..frames {
            let index = first_index + f as u32;
            let original = synth_original(&mut rng, w, h, index)?;
            let edit = model_edit(&original);
            let reference = apply_edit(&original, &edit);
            write_pair(&dirs[0], f, &original, &reference)?;
            write_pair(&dirs[1], f + 7, &original, &reference)?; // different capture numbering, same frames
            let s1: [Plane; 3] = std::array::from_fn(|c| offset(&edit[c], w, h, 1, 0));
            write_pair(&dirs[2], f, &original, &apply_edit(&original, &s1))?;
            let s2: [Plane; 3] = std::array::from_fn(|c| offset(&edit[c], w, h, 3, 2));
            write_pair(&dirs[3], f, &original, &apply_edit(&original, &s2))?;
            let noise: [Plane; 3] = std::array::from_fn(|_| rng.randbytes(w * h).iter().map(|&n| (n as f32 - 127.5) * 0.12f32).collect());
            write_pair(&dirs[4], f, &original, &apply_edit(&original, &noise))?;
            if f != 2 {
                write_pair(&dirs[5], f, &original, &reference)?;
            }
        }
        let _ = planes;
        let cmp = |i: usize| compare_runs(dirs[0].to_str().unwrap_or(""), dirs[i].to_str().unwrap_or(""), false, false);
        let (same, shifted1, shifted, noise, partial) = (cmp(1)?, cmp(2)?, cmp(3)?, cmp(4)?, cmp(5)?);
        let get = |v: Option<f64>| v.ok_or_else(|| "TypeError: unsupported format string passed to NoneType.__format__".to_string());
        let (sn, s1n, s2n, nn) = (get(same.ncc_mean)?, get(shifted1.ncc_mean)?, get(shifted.ncc_mean)?, get(noise.ncc_mean)?);
        println!("identical runs:         ncc {}  detail {}%  matched {}", fmt_f(sn, 4), fmt_f(get(same.detail_pct)?, 1), same.matched);
        println!("edit shifted by (1,0):  ncc {}  detail {}%", fmt_f(s1n, 4), fmt_f(get(shifted1.detail_pct)?, 1));
        println!("edit shifted by (3,2):  ncc {}  detail {}%", fmt_f(s2n, 4), fmt_f(get(shifted.detail_pct)?, 1));
        println!("noise-only edit:        ncc {}  detail {}%", fmt_f(nn, 4), fmt_f(get(noise.detail_pct)?, 1));
        println!("one frame missing:      matched {}  ref_only {}", partial.matched, partial.ref_only);
        let mut indexes: Vec<i64> = same.frames.iter().map(|f| f.index).collect();
        indexes.sort();
        let checks: Vec<(&str, bool)> = vec![
            ("stamps decode to the right frame indices", indexes == (first_index as i64..first_index as i64 + frames as i64).collect::<Vec<_>>()),
            ("identical runs give ncc ~1", (sn - 1.0).abs() < 1e-3),
            ("identical runs give detail ~100%", (get(same.detail_pct)? - 100.0).abs() < 0.5),
            ("a shifted edit gives clearly lower ncc", s2n < sn - 0.3),
            ("a 1 px shift scores between identical and a 3 px shift", s2n < s1n && s1n < sn - 0.05),
            ("a noise-only edit gives ncc ~0", nn.abs() < 0.05),
            ("a missing frame is skipped and counted", partial.matched == frames - 1 && partial.ref_only == 1),
        ];
        let failed = checks.iter().any(|(_, ok)| !ok);
        for (name, ok) in &checks {
            println!("  {} {name}", if *ok { "ok  " } else { "FAIL" });
        }
        Ok(u8::from(failed))
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_matches_python() {
        // Reference values from CPython: random.Random(1234).
        let hex = |v: Vec<u8>| v.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(hex(Mt::new(1234).randbytes(8)), "b97f69f75edf35c7");
        let mut r = Mt::new(1234);
        assert_eq!(hex(r.randbytes(7)), "b97f69f7df35c7");
        assert_eq!(r.below(256), 225);
        assert_eq!(r.range(16, 216), 45);
        assert_eq!(r.random(), 0.007491470058587191);
        assert_eq!(r.range(6, 30), 24);
    }
}
