//! The self-test bench-iq carried over from scripts/bench-iq.py, as `--self-test` and as unit tests. The scenes are float64 like the script's;
//! numpy's random generator is not reproduced, any uniform noise serves the same checks.

use std::fs;

use super::*;

const H: usize = 120;
const W: usize = 160;

macro_rules! check {
    ($cond:expr, $($msg:tt)+) => {
        if $cond {
        } else {
            return Err(format!($($msg)+));
        }
    };
}

/// Uniform [0, 1) noise (splitmix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A textured scene (luma 0.05..0.6) whose texture sits well above the 0.05 x geomean floor.
fn scene(rng: &mut Rng) -> Img<f64> {
    Img { h: H, w: W, d: (0..H * W).map(|_| 0.05 + 0.55 * rng.next()).collect() }
}

fn scaled(a: &Img<f64>, k: f64) -> Img<f64> {
    a.like(a.d.iter().map(|v| v * k).collect())
}

/// numpy's `roll`.
fn roll(a: &Img<f64>, shift: usize, axis: usize) -> Img<f64> {
    let mut d = vec![0.0; a.d.len()];
    for y in 0..a.h {
        for x in 0..a.w {
            let (ty, tx) = if axis == 0 { ((y + shift) % a.h, x) } else { (y, (x + shift) % a.w) };
            d[ty * a.w + tx] = a.d[y * a.w + x];
        }
    }
    a.like(d)
}

fn changes(ls: &[Img<f64>]) -> Result<Vec<f64>, String> {
    let pairs = pair_changes(&log_frames(ls, floor_of(&ls[0]))).map_err(|e| e.to_string())?;
    Ok(pairs.iter().map(|p| trimmed_mean(&p.d)).collect())
}

fn cat(parts: &[&[Img<f64>]]) -> Vec<Img<f64>> {
    parts.iter().flat_map(|p| p.iter().cloned()).collect()
}

fn moving_scene(base: &Img<f64>) -> Vec<Img<f64>> {
    (0..WINDOW as usize)
        .map(|i| {
            let mut y = base.clone();
            for r in 10..34 {
                for c in 10 + 8 * i..42 + 8 * i {
                    y.d[r * W + c] = 1.0;
                }
            }
            y
        })
        .collect()
}

fn still_scene(base: &Img<f64>) -> Vec<Img<f64>> {
    (0..WINDOW).map(|_| base.clone()).collect()
}

fn err(e: IqError) -> String {
    e.to_string()
}

pub(super) fn flicker_measure() -> Result<(), String> {
    let base = scene(&mut Rng(1));
    check!(flicker(&still_scene(&base)).map_err(err)? == 0.0, "identical frames: no flicker");
    // A global luma wobble of +-2%: every pair changes by log2(1.02 / 0.98) stops minus what the floor takes off; no pixel is
    // a mover, so the 5% exclusion keeps a uniform change as it is.
    let wobble: Vec<Img<f64>> = (0..WINDOW).map(|i| scaled(&base, if i % 2 == 1 { 1.02 } else { 0.98 })).collect();
    let f0 = floor_of(&wobble[0]);
    let diffs: Vec<f64> = base.d.iter().map(|v| ((v * 1.02 + f0).log2() - (v * 0.98 + f0).log2()).abs()).collect();
    let want = np_mean(&diffs);
    let got = flicker(&wobble).map_err(err)?;
    check!((got - want).abs() < 2e-3 * want, "wobble {got} vs {want}");
    check!(percent(got) > 1.5 && percent(got) < 4.0, "wobble percent {}", percent(got));
    // Without the floor's damping it is exactly the wobble.
    let v = (1.02_f64 / 0.98).log2();
    check!((trimmed_mean(&[v; 100]) - v).abs() < 1e-12, "trimmed mean of a uniform change");
    // A bright block (4% of the pixels) moving across an otherwise still scene is excluded.
    check!(((24 * 32) as f64 / (H * W) as f64) < EXCLUDE, "block share");
    let moving = flicker(&moving_scene(&base)).map_err(err)?;
    check!(moving < 1e-3, "moving block flicker {}", percent(moving));
    Ok(())
}

pub(super) fn detail_measure() -> Result<(), String> {
    let base = scene(&mut Rng(2));
    let still = still_scene(&base);
    let one = detail(&still, &still).map_err(err)?.ok_or("no static pixels")?;
    check!((one - 1.0).abs() < 1e-12, "identical runs give x1.00, got {one}");
    let blur = base.like(
        (0..H * W)
            .map(|i| {
                let (y, x) = (i / W, i % W);
                let at = |yy: usize, xx: usize| base.d[yy * W + xx];
                (at(y, x) + at((y + H - 1) % H, x) + at((y + 1) % H, x) + at(y, (x + W - 1) % W) + at(y, (x + 1) % W)) / 5.0
            })
            .collect(),
    );
    let blurred = detail(&vec![blur.clone(); WINDOW as usize], &still).map_err(err)?.ok_or("no static pixels")?;
    check!(blurred < 0.9, "a blurred run has less detail, got {blurred}");
    let sharp = base.like(base.d.iter().zip(&blur.d).map(|(b, bl)| (b + 0.3 * (b - bl)).max(1e-3)).collect());
    let sharpened = detail(&vec![sharp; WINDOW as usize], &still).map_err(err)?.ok_or("no static pixels")?;
    check!(sharpened > 1.1, "a sharpened run has more detail, got {sharpened}");
    // The moving block is not static in the moving run: the detail ratio ignores it.
    let ratio = detail(&moving_scene(&base), &still).map_err(err)?.ok_or("no static pixels")?;
    check!((ratio - 1.0).abs() < 1e-12, "the moving block is ignored, got {ratio}");
    Ok(())
}

pub(super) fn anchoring() -> Result<(), String> {
    let mut rng = Rng(3);
    let base = scene(&mut rng);
    let still = still_scene(&base);
    // The window starts SETTLE frames after a cut, here into a black frame then the shot.
    let other = scene(&mut rng);
    let black = Img { h: H, w: W, d: vec![1e-4; H * W] };
    let shot_of = |n: i64| -> Vec<Img<f64>> { (0..n).map(|i| scaled(&other, 1.0 + 0.001 * (i % 2) as f64)).collect() };
    let shot = shot_of(SETTLE + WINDOW);
    let b2 = [black.clone(), black.clone()];
    let got = anchor(&changes(&cat(&[&still, &b2, &shot]))?, WINDOW, 1);
    check!(got == Some((WINDOW + SETTLE, true)), "window after the cut: {got:?}");
    let got = anchor(&changes(&cat(&[&still, &b2[..1], &shot[..SETTLE as usize]]))?, WINDOW, 1);
    check!(got.is_none(), "the series ends too soon: {got:?}");
    let got = anchor(&changes(&still)?, WINDOW, 1);
    check!(got == Some((0, false)), "no cut: from the start, and said so: {got:?}");
    // Frame generation: settle counts rendered frames, the window whole cycles.
    check!(settle(4) == 4 * SETTLE && settle(1) == SETTLE, "settle");
    check!([1, 2, 3, 4].map(window_frames) == [9, 9, 10, 9], "window_frames {:?}", [1, 2, 3, 4].map(window_frames));
    let long_shot = shot_of(settle(2) + WINDOW);
    let got = anchor(&changes(&cat(&[&still, &b2, &long_shot]))?, WINDOW, 2);
    check!(got == Some((WINDOW + settle(2), true)), "fg 2: {got:?}");
    let pan: Vec<Img<f64>> = (0..4).map(|i| roll(&base, 3 * i, 1)).collect();
    let got = anchor(&changes(&cat(&[&pan, &still]))?, WINDOW, 1);
    check!(got == Some((0, false)), "a camera that stops is not a cut: {got:?}");
    Ok(())
}

pub(super) fn series_lookup() -> Result<(), String> {
    // A run whose series was not moved into it is found through its launch log, spaces included.
    let tmp = std::env::temp_dir().join(format!("xtask-bench-iq-{}-{}", std::process::id(), std::thread::current().name().unwrap_or("t").replace("::", "-")));
    let _ = fs::remove_dir_all(&tmp);
    let result = (|| -> Result<(), String> {
        let tmp = tmp.to_string_lossy().to_string();
        let run = join(&tmp, "run");
        let series = join(&join(&tmp, "elsewhere dir"), "series-123");
        fs::create_dir_all(&run).map_err(|e| e.to_string())?;
        fs::create_dir_all(&series).map_err(|e| e.to_string())?;
        check!(series_dir(&run).is_none(), "no series, no log");
        let log = format!(
            "[neural-forge-layer] [capture] capturing the next 40 presents into /nonexistent/series-1\n[neural-forge-layer] [capture] capturing the next 40 presents into {series}\n[neural-forge-layer] [present] 60.0 fps\n"
        );
        fs::write(join(&run, "launch.log"), log).map_err(|e| e.to_string())?;
        check!(series_dir(&run).as_deref() == Some(series.as_str()), "log series: {:?}", series_dir(&run));
        let moved = join(&run, "series-456");
        fs::create_dir_all(&moved).map_err(|e| e.to_string())?;
        check!(series_dir(&run).as_deref() == Some(moved.as_str()), "the moved series wins over the log");
        Ok(())
    })();
    let _ = fs::remove_dir_all(&tmp);
    result
}

/// What the Rust port needs beyond the script's checks: numpy's float32 kernels, which are bit-compared against values
/// numpy gave (generated with numpy 2.3.5 on an AVX2/FMA3 machine).
pub(super) fn numpy_kernels() -> Result<(), String> {
    let ln: [(u32, u32); 6] = [
        (0x3f48_6945, 0xbe7a_ae7a),
        (0x3f80_0000, 0),
        (0x3a9e_2b9c, 0xc0d7_08d8),
        (0x4000_0000, 0x3f31_7218),
        (0x3f00_0000, 0xbf31_7218),
        (0x3dcc_cccd, 0xc013_5d8e),
    ];
    for (x, want) in ln {
        let got = np_ln_f32(f32::from_bits(x)).to_bits();
        check!(got == want, "ln {x:#x}: {got:#x} vs {want:#x}");
    }
    let exp: [(u32, u32); 5] =
        [(0xbf00_0000, 0x3f1b_4598), (0xc040_0000, 0x3d4b_ed86), (0xbf9e_064b, 0x3e94_f8cd), (0xc0fc_cccd, 0x39c2_6059), (0xba83_126f, 0x3f7f_be7f)];
    for (x, want) in exp {
        let got = np_exp_f32(f32::from_bits(x)).to_bits();
        check!(got == want, "exp {x:#x}: {got:#x} vs {want:#x}");
    }
    check!(py_sum(&[0.1; 10]) == 1.0, "compensated sum");
    check!(fixed(0.125, 2) == "0.12" && fixed(2.5, 0) == "2", "ties to even");
    Ok(())
}

pub(super) fn run() -> Result<(), String> {
    flicker_measure()?;
    detail_measure()?;
    anchoring()?;
    series_lookup()?;
    numpy_kernels()
}

#[cfg(test)]
mod tests {
    #[test]
    fn flicker_measure() {
        super::flicker_measure().unwrap();
    }
    #[test]
    fn detail_measure() {
        super::detail_measure().unwrap();
    }
    #[test]
    fn anchoring() {
        super::anchoring().unwrap();
    }
    #[test]
    fn series_lookup() {
        super::series_lookup().unwrap();
    }
    #[test]
    fn numpy_kernels() {
        super::numpy_kernels().unwrap();
    }
}
