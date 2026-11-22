//! The bits of Pillow the scripts use: 8-bit PNG reading and writing, crop, paste and the fixed-point
//! separable resize (Pillow's Resample.c), so Lanczos and bicubic scaling match it exactly.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub type Fail = String;

#[derive(Clone)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

#[derive(Clone, Copy)]
pub enum Filter {
    Bicubic,
    Lanczos,
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Bicubic => 2.0,
            Filter::Lanczos => 3.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            Filter::Bicubic => {
                let a = -0.5;
                let x = x.abs();
                if x < 1.0 {
                    (((a + 2.0) * x - (a + 3.0)) * x) * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * a
                } else {
                    0.0
                }
            }
            Filter::Lanczos => {
                if (-3.0..3.0).contains(&x) {
                    sinc(x) * sinc(x / 3.0)
                } else {
                    0.0
                }
            }
        }
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// Per output index: (first input index, integer coefficients).
fn coeffs(in_size: usize, out_size: usize, filter: Filter) -> Vec<(usize, Vec<i32>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ss = 1.0 / filterscale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f64 + 0.5) * scale;
            let xmin = ((center - support + 0.5) as i64).max(0);
            let xmax = (((center + support + 0.5) as i64).min(in_size as i64) - xmin).max(0);
            let mut k: Vec<f64> = (0..xmax).map(|x| filter.eval((x as f64 + xmin as f64 - center + 0.5) * ss)).collect();
            let ww: f64 = k.iter().sum();
            if ww != 0.0 {
                for v in &mut k {
                    *v /= ww;
                }
            }
            let q = k
                .iter()
                .map(|&v| if v < 0.0 { (-0.5 + v * (1u32 << PRECISION_BITS) as f64) as i32 } else { (0.5 + v * (1u32 << PRECISION_BITS) as f64) as i32 })
                .collect();
            (xmin as usize, q)
        })
        .collect()
}

fn clip8(v: i32) -> u8 {
    (v >> PRECISION_BITS).clamp(0, 255) as u8
}

/// `Image.resize((ow, oh), filter)` of an 8-bit image with `ch` interleaved channels.
pub fn resize_u8(src: &[u8], ch: usize, w: usize, h: usize, ow: usize, oh: usize, filter: Filter) -> Vec<u8> {
    let mut cur = src.to_vec();
    let (mut cw, mut chh) = (w, h);
    if ow != w {
        let kx = coeffs(w, ow, filter);
        let mut out = vec![0u8; ow * h * ch];
        for y in 0..h {
            for (x, (xmin, k)) in kx.iter().enumerate() {
                for c in 0..ch {
                    let mut ss = 1i32 << (PRECISION_BITS - 1);
                    for (i, kv) in k.iter().enumerate() {
                        ss += cur[(y * w + xmin + i) * ch + c] as i32 * kv;
                    }
                    out[(y * ow + x) * ch + c] = clip8(ss);
                }
            }
        }
        cur = out;
        cw = ow;
    }
    if oh != h {
        let ky = coeffs(h, oh, filter);
        let mut out = vec![0u8; cw * oh * ch];
        for (y, (ymin, k)) in ky.iter().enumerate() {
            for x in 0..cw {
                for c in 0..ch {
                    let mut ss = 1i32 << (PRECISION_BITS - 1);
                    for (i, kv) in k.iter().enumerate() {
                        ss += cur[((ymin + i) * cw + x) * ch + c] as i32 * kv;
                    }
                    out[(y * cw + x) * ch + c] = clip8(ss);
                }
            }
        }
        cur = out;
        chh = oh;
    }
    let _ = chh;
    cur
}

impl Rgb {
    pub fn filled(w: usize, h: usize, c: [u8; 3]) -> Rgb {
        Rgb { w, h, data: c.iter().copied().cycle().take(w * h * 3).collect() }
    }

    /// `img.crop((x0, y0, x1, y1))`; the part outside the image is black.
    pub fn crop(&self, x0: i64, y0: i64, x1: i64, y1: i64) -> Result<Rgb, Fail> {
        if x1 < x0 {
            return Err("right is less than left".into());
        }
        if y1 < y0 {
            return Err("lower is less than upper".into());
        }
        let (nw, nh) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut out = Rgb::filled(nw, nh, [0, 0, 0]);
        for y in 0..nh as i64 {
            for x in 0..nw as i64 {
                let (sx, sy) = (x0 + x, y0 + y);
                if sx >= 0 && sy >= 0 && (sx as usize) < self.w && (sy as usize) < self.h {
                    let s = (sy as usize * self.w + sx as usize) * 3;
                    let d = (y as usize * nw + x as usize) * 3;
                    out.data[d..d + 3].copy_from_slice(&self.data[s..s + 3]);
                }
            }
        }
        Ok(out)
    }

    pub fn resize_lanczos(&self, ow: i64, oh: i64) -> Result<Rgb, Fail> {
        self.resize(ow, oh, Filter::Lanczos)
    }

    pub fn resize(&self, ow: i64, oh: i64, filter: Filter) -> Result<Rgb, Fail> {
        if ow <= 0 || oh <= 0 {
            return Err("height and width must be > 0".into());
        }
        let (ow, oh) = (ow as usize, oh as usize);
        if (ow, oh) == (self.w, self.h) {
            return Ok(self.clone());
        }
        Ok(Rgb { w: ow, h: oh, data: resize_u8(&self.data, 3, self.w, self.h, ow, oh, filter) })
    }

    /// `sheet.paste(img, (x, y))`, clipped to the sheet.
    pub fn paste(&mut self, img: &Rgb, x: i64, y: i64) {
        for iy in 0..img.h as i64 {
            for ix in 0..img.w as i64 {
                let (dx, dy) = (x + ix, y + iy);
                if dx >= 0 && dy >= 0 && (dx as usize) < self.w && (dy as usize) < self.h {
                    let s = (iy as usize * img.w + ix as usize) * 3;
                    let d = (dy as usize * self.w + dx as usize) * 3;
                    self.data[d..d + 3].copy_from_slice(&img.data[s..s + 3]);
                }
            }
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), Fail> {
        write_rgb_png(path, self.w, self.h, &self.data)
    }

    /// `img.convert("L")`: Pillow's integer luma.
    pub fn luma_at(&self, x: usize, y: usize) -> u8 {
        let p = (y * self.w + x) * 3;
        ((self.data[p] as u32 * 19595 + self.data[p + 1] as u32 * 38470 + self.data[p + 2] as u32 * 7471 + 0x8000) >> 16) as u8
    }
}

pub fn write_rgb_png(path: &Path, w: usize, h: usize, data: &[u8]) -> Result<(), Fail> {
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| format!("{}: {e}", path.display()))?;
    writer.write_image_data(data).map_err(|e| format!("{}: {e}", path.display()))
}

/// `Image.open(path).convert("RGB")` for PNGs: palette and low bit depths expanded, alpha dropped, 16-bit
/// channels cut to their high byte.
pub fn read_rgb_png(path: &Path) -> Result<Rgb, Fail> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut dec = png::Decoder::new(BufReader::new(file));
    dec.set_transformations(png::Transformations::EXPAND);
    let mut reader = dec.read_info().map_err(|e| format!("{}: {e}", path.display()))?;
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or("image too large")?];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let bytes_per = if info.bit_depth == png::BitDepth::Sixteen { 2 } else { 1 };
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(format!("{}: palette not expanded", path.display())),
    };
    let mut data = Vec::with_capacity(w * h * 3);
    for i in 0..w * h {
        let s = i * channels * bytes_per;
        let sample = |c: usize| buf[s + c * bytes_per];
        if channels <= 2 {
            data.extend_from_slice(&[sample(0); 3]);
        } else {
            data.extend_from_slice(&[sample(0), sample(1), sample(2)]);
        }
    }
    Ok(Rgb { w, h, data })
}
