//! The self-test bench-report carried over from scripts/bench-report.py, as `--self-test` and as unit tests, plus checks of the Python behaviour the
//! port stands in for (strptime, timedelta, repr, median).

use std::fs::{self, File};
use std::io::BufWriter;

use super::*;
use crate::bench_iq::{pyish::py_repr_str, WINDOW};

macro_rules! check {
    ($cond:expr, $($msg:tt)+) => {
        if $cond {
        } else {
            return Err(format!($($msg)+));
        }
    };
}

pub(super) fn rates() -> Result<(), String> {
    let log = "[neural-forge-layer] [present] 66.0 fps (0.0/s composited by the effect) over 5.0s\n\
        [neural-forge-layer] [present] 30.0 fps (29.0/s composited by the effect) over 5.0s\n\
        [neural-forge-layer] [preupscale] mode=model extent=1485x836 (padded 1486x836) holds=300 hold_ms median=10.20 capture_gpu_ms median=0.68 writeback_gpu_ms median=0.52 misses=0 (total 3) holds_per_s=65.8\n\
        [neural-forge-layer] [preupscale] mode=model extent=1485x836 (padded 1486x836) holds=600 hold_ms median=10.10 capture_gpu_ms median=0.68 writeback_gpu_ms median=0.52 misses=0 (total 3) holds_per_s=66.2\n\
        [neural-forge-layer] [preupscale] mode=model extent=1485x836 (padded 1486x836) holds=900 hold_ms median=10.10 capture_gpu_ms median=0.68 writeback_gpu_ms median=0.52 misses=0 (total 3) holds_per_s=40.0\n";
    check!(held_per_second(log)? == Some(65.8), "held_per_second {:?}", held_per_second(log));
    check!(held_per_second("[present] 61.0 fps (61.0/s composited by the effect) over 5.0s")?.is_none(), "no preupscale lines");
    check!(rate_text(Some(61.0), None) == "composited/s 61.0", "rate_text without holds");
    check!(engaged(Some(3.99)) && engaged(Some(1.97)) && !engaged(Some(1.01)) && !engaged(None), "engaged");
    check!(rate_text(Some(29.0), Some(65.8)).starts_with("held before upscaler/s 65.8 (composited/s 29.0"), "rate_text with holds");
    Ok(())
}

/// A scratch directory under the system temp dir, removed when done.
fn scratch(name: &str) -> String {
    std::env::temp_dir().join(format!("xtask-bench-report-{}-{name}", std::process::id())).to_string_lossy().to_string()
}

fn write_png(path: &str, w: u32, h: u32, pixels: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().and_then(|mut wr| wr.write_image_data(pixels)).map_err(|e| e.to_string())
}

pub(super) fn series_text() -> Result<(), String> {
    // A run with a series of identical frames: no flicker, and no detail added over itself.
    let run = scratch("series");
    let _ = fs::remove_dir_all(&run);
    let result = (|| -> Result<(), String> {
        fs::create_dir_all(format!("{run}/series-1")).map_err(|e| e.to_string())?;
        let pixels: Vec<u8> = (0..64 * 48 * 3).map(|i| ((i * 37) % 256) as u8).collect();
        for i in 0..WINDOW {
            write_png(&format!("{run}/series-1/{i:06}-composited.png"), 64, 48, &pixels)?;
        }
        let got = iq_text(&run, Some(&run), None)?;
        check!(got == " | flicker 0.00% (no cut) detail x1.00", "iq_text {got:?}");
        let none = iq_text(&format!("{run}/series-1/none"), None, None)?;
        check!(none.is_empty(), "no series: {none:?}");
        Ok(())
    })();
    let _ = fs::remove_dir_all(&run);
    result
}

/// The Python behaviour the report leans on.
pub(super) fn python_behaviour() -> Result<(), String> {
    // strptime accepts one-digit fields and rejects what datetime() refuses.
    check!(pytime::parse_gta_stamp("26-11-12-17-58-58") == Some(1_794_506_338 * 1_000_000), "gta stamp");
    check!(pytime::parse_gta_stamp("26-1-2-3-4-5").is_some(), "short fields");
    check!(pytime::parse_gta_stamp("26-11-12-17-58-60").is_none() && pytime::parse_gta_stamp("26-02-30-1-1-1").is_none(), "refused");
    check!(pytime::parse_gta_stamp("26-11-12-17-58-58x").is_none(), "unconverted data remains");
    check!(pytime::parse_gta_stamp("69-01-01-0-0-0") < pytime::parse_gta_stamp("68-01-01-0-0-0"), "%y pivot");
    check!(pytime::parse_gpu_stamp("2026/11/12 17:58:58").is_some() && pytime::parse_csv_stamp("2026-11-12_18-00-37").is_some(), "other stamps");
    // timedelta rounds half to even.
    check!(pytime::seconds(0.75) == 750_000 && pytime::seconds(-0.0000004) == 0 && pytime::seconds(1.5) == 1_500_000, "seconds");
    check!(pytime::microseconds(2.5) == 2 && pytime::microseconds(3.5) == 4, "microseconds");
    // repr, median, ms.
    check!(py_float_repr(10.2) == "10.2" && py_float_repr(1e-5) == "1e-05" && py_float_repr(1e16) == "1e+16" && py_float_repr(300.0) == "300.0", "float repr");
    check!(py_float_repr(0.0001) == "0.0001" && py_float_repr(123456789012345.6) == "123456789012345.6", "float repr positional");
    check!(py_repr_str("a'b") == "\"a'b\"" && py_repr_str("it") == "'it'", "str repr");
    check!(median(vec![3.0, 1.0, 2.0, 10.0]) == 2.5 && median(vec![5.0]) == 5.0, "median");
    check!(ms("250us")? == Some(0.25) && ms("1.5s")? == Some(1500.0) && ms("300")?.is_none(), "ms");
    check!(ms("1.2.3").is_ok_and(|v| v.is_none()) && ms("1.2.3ms").is_err(), "ms with a bad number");
    let log = "[sync] a=1ms b=x c=3\n[sync] a=2ms b=x c=4ms\n[sync] a=4ms b=y c=zz\n";
    check!(sync_medians(log)? == "2 lines: a=3.00 b=x/y c=4.0/zz", "sync_medians {:?}", sync_medians(log));
    check!(sync_medians("nothing")? == "-", "no sync lines");
    Ok(())
}

pub(super) fn run() -> Result<(), String> {
    rates()?;
    series_text()?;
    python_behaviour()
}

#[cfg(test)]
mod tests {
    #[test]
    fn rates() {
        super::rates().unwrap();
    }
    #[test]
    fn series_text() {
        super::series_text().unwrap();
    }
    #[test]
    fn python_behaviour() {
        super::python_behaviour().unwrap();
    }
}
